---
description: "從 Prime 的 persistent Python、Hermes 的 personal agent service，到 OpenClaw 的 Gateway control plane，沿著一次長任務比較三個 agent runtime 如何安排控制流、狀態和權限。"
date: "2026-09-15"
updated: "2026-09-15"
language: "zh-Hant"
image: "/assets/blog/agent-runtime/overview.png"
tags:
  - Architecture
  - Agents
  - Security
---

# Prime、Hermes、OpenClaw：三個 Agent Runtime 的架構選擇

*2026-09-15 · Agent Systems / Runtime / Security*

<figure id="agent-runtime-overview" class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/overview.png" width="1300" height="680" alt="Prime Agent、Hermes Agent 與 OpenClaw 2.0 的手繪 runtime 架構比較" loading="eager" decoding="async">
  <figcaption><strong>圖 1.</strong> 三個 runtime 如何接住模型提出的工作、執行工具並保存後續需要的 state。整理自 <a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a> 和 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

## 什麼是 Agent Harness？

以一次程式修復任務為例。使用者把一個 repository、失敗測試和修改要求交給 agent。模型先決定要讀哪些檔案，工具回傳內容後再提出修改，接著執行測試；測試結果回來，模型可能繼續修正，也可能結束。

模型在每一輪提出下一段文字或 action。讀檔、寫檔和執行測試則由外部程式完成，結果還要被整理成下一輪能使用的 context。任務中斷時，系統也要知道哪些工具已經完成、哪些檔案已經改過，以及下一次應該從哪裡接續。

把模型呼叫、工具執行、結果交接、session 保存、錯誤恢復和權限檢查接起來的程式，稱為 agent harness。它持有整段工作的執行環境。

這裡的 runtime 指 harness 的執行組織。對一個長任務，要追 request 從哪裡進來、哪個 component 持有 loop、state 寫到哪裡、工具在哪個 process 執行，以及 policy 在 action 到達工具前如何介入。

## 三個 project 的出發點

Prime Agent 由 Prime Intellect 開發。它的 README 把 coding、research 和 general long-running work 放在同一個定位裡，並用兩個抽象組織工作：RLM 把 context 當成可以由 Python 操作的變數，Continual Harness 則把 prompt、memory、skill description 和可重用的 subagent specification 留成 durable state。[^prime-readme]

Hermes Agent 由 Nous Research 開發。它從 personal agent 的使用方式出發，把 CLI 和 messaging gateway 接到同一套 agent，並把 provider、session search、memory、skills、cron 和 terminal backend 放在同一個可自行部署的服務裡。[^hermes-readme]

OpenClaw 來自 OpenClaw project 和社群。<code>v2026.8.1</code> 的 README 把 assistant 放在使用者已有的 devices 和 chat channels 旁邊，Gateway 負責接起模型、工具、訊息通道和 companion apps；文件另外把 Gateway、內建 agent runtime 和 plugin harness 分成不同層。[^openclaw-readme] [^openclaw-runtime]

這三個 project 面對的工作單位因此不同。Prime 讓一個 coding／research session 持續操作，Hermes 維持一個人每天反覆使用的 agent service，OpenClaw 則維持一個可能跨 channel、device、plugin 和 automation 的 Gateway system。

版本固定為 Prime Agent commit <code>1fc1adb6</code>、Hermes Agent commit <code>afe06f2</code>，以及 OpenClaw <code>v2026.8.1</code> release。這個比較把該 pinned release 稱為 OpenClaw 2.0，current main 留在比較範圍之外。[^openclaw-release]

## 一次 Agent 任務需要哪些執行步驟

把剛才的程式修復任務拆開，可以得到一條反覆回到模型的路徑：

~~~text
request
  -> model call
  -> action
  -> tool execution
  -> result
  -> next context
  -> model call
~~~

每次 tool execution 都會增加一個交接點。工具可能只回傳幾行測試結果，也可能修改檔案、啟動長時間 process，或把工作交給另一個 agent。下一輪需要的 context 可能留在記憶體、session transcript、workspace 檔案或資料庫裡。

如果測試執行十秒，模型等待期間，任務的 context 和工具狀態仍然要被保留。如果工具執行到一半失敗，系統要分辨失敗發生在模型呼叫、工具本身、結果寫入，還是下一輪 context 組裝。runtime 的邊界，就是在這些交接點上形成的。

後面的比較固定追蹤四個問題：

| 要追的部分 | 具體問題 |
| --- | --- |
| loop | 哪個 component 決定下一次 model call 和 tool call？ |
| state | 哪些 context、工作產物和 child result 能跨過一次 turn？ |
| execution | action 最後在哪個 process、container 或 remote worker 執行？ |
| admission | 哪個 policy 或 approval 可以讓 action 通過，哪個只能記錄身份？ |

## 模型產生 action 後，由哪一層接手？

control surface 指模型用來組織下一步工作的程式介面；使用者看到的 UI 不在這個定義裡。三個 runtime 把這個介面放在不同位置：

| 比較面向 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| model-facing surface | persistent Python REPL／RLM | AIAgent conversation loop | OpenClaw embedded runtime |
| loop 的主要 owner | Python kernel 加上 TypeScript host | AIAgent 和 gateway service | Gateway 加上 embedded agent runner |
| 工作如何延伸 | Python state、skills、<code>rlm.spawn(...)</code> | tools、skills、delegation、background task | native tools、plugins、nodes、automation |
| 主要收益 | 長任務可以用程式保存工作流 | provider、入口和日常服務集中 | routing、session、tool 和 policy 有共同 owner |
| 主要成本 | Python state、child lifecycle 和 host permission | service scope、memory correctness 和 backend scope | Gateway、plugin、node 和 channel scope |

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/control-surface.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 三種 per-session control loop 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 5.</strong> 三個 per-session loop 保存的單位不同：Prime 保存 Python cell 和 child session，Hermes 保存 tool call 和 delegation，OpenClaw 保存 intake、context、native tool 和 persistence。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

Prime 的 RLM 文件把 persistent Python kernel 放在模型和工具之間。模型使用的內建 model tool 是 <code>ipython</code>；讀檔、編輯、跑命令、轉換結果、呼叫 skills 和建立 subagent 都從這個 kernel 展開。Python 變數、import、函式和 task handle 可以跨過後續 tool call，context 不需要每次都重新組成一個巨大訊息。[^prime-rlm]

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/prime.png" width="1300" height="680" alt="Prime Agent 的 persistent Python RLM、工具和 child agent 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 2.</strong> Prime 把 persistent Python／RLM 放在 model-facing surface。檔案、shell、skills、MCP 和 child agent 都從這個工作面接出去。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>。</figcaption>
</figure>

<code>rlm.spawn(...)</code> 也沿著這個介面建立 child session。呼叫先完成 task admission，回傳的是 child handle；child 的結果要透過 <code>agent_message</code> 或檔案回到 parent。這讓 parent 可以在同一個 Python 工作面安排多個獨立工作，但 child 的生命週期和結果合併不能當成一般同步函式處理。[^prime-rlm]

Prime 的 TypeScript host 仍然持有 provider call、session persistence、child lifecycle、scheduling 和 safety policy。persistent Python 是模型可以編程的控制環境；provider、session 和安全策略仍由 host 管理。RLM 文件明確說明，Python kernel 執行 model-generated Python 和 project commands 時使用 worker 的 operating-system permissions；worker 和 kernel 改善生命週期隔離與恢復，權限隔離則需要外部 sandbox 或受限環境。

這個分工讓 Prime 適合需要反覆讀資料、轉換結果和分派子任務的工作。代價是恢復時要同時確認 Python state、檔案副作用、child registry 和 host process 是否一致；程式自由度越高，resume 需要重建的狀態也越多。

Hermes 把控制流收在 AIAgent 的 conversation loop。CLI 和 messaging gateway 是兩個入口，gateway 可以把 Telegram、Discord、Slack、WhatsApp、Signal 和 CLI 的工作送進同一個服務；AIAgent 這一層組 prompt、解析 provider、建立 tool surface，再依序處理 tool result 和下一輪 model call。[^hermes-readme] [^hermes-loop]

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/hermes.png" width="1300" height="680" alt="Hermes Agent 的 AIAgent loop、provider resolver、tool registry 和 memory 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 3.</strong> Hermes 把 CLI 和 messaging gateway 接到 AIAgent loop。provider、tool registry、terminal backend、memory、skills 和 cron 由同一個 personal service 維持。來源：<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>。</figcaption>
</figure>

Hermes 的擴充單位是 tool、skill、delegation 或 background task。Terminal backend 可以是 local、Docker、SSH、Singularity、Modal、Daytona 或 Vercel Sandbox；這些 backend 決定實際命令在哪裡執行。模型獲得的是 service 提供的工具表面，工作流狀態則由 session、memory、skills 和 background machinery 接住。[^hermes-readme]

這種安排適合每天從不同入口使用同一個 assistant。provider switching、session search、memory loading 和 scheduled automation 不需要分散到多個獨立 runtime。相應的維護問題集中在 service：session 是否寫入、FTS5 索引是否更新、memory 是否取回過期內容，以及 terminal backend 是否和預期的 workspace 對齊。

OpenClaw 把 Gateway 放在入口和 agent runtime 之間。Gateway 是 sessions、tools、events 和 channel connections 的 control plane；內建 agent runtime 則在 <code>src/agents/</code> 和 <code>@openclaw/agent-core</code> 中持有 attempt loop、model/provider wiring、compaction、transcript 和 session wiring。plugin 可以註冊其他 harness，但那是 runtime selection 的擴充，不會改變 Gateway 作為入口和協調 owner 的位置。[^openclaw-readme] [^openclaw-runtime] [^openclaw-runtimes]

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/openclaw.png" width="1300" height="680" alt="OpenClaw 2.0 的 Gateway、session routing、policy、plugins、automation 和 workspace 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 4.</strong> OpenClaw 將 channels、CLI、nodes、plugins 和 automation 接到 Gateway，再由 embedded runtime 處理 agent turn。來源：<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">Gateway docs</a> 和 <a href="https://github.com/openclaw/openclaw/blob/v2026.8.1/docs/agent-runtime-architecture.md">agent runtime architecture</a>。</figcaption>
</figure>

Gateway 的集中 ownership 讓 OpenClaw 可以在同一個位置處理 session routing、channel delivery、node capability 和 tool policy。這對多入口系統很有用，因為同一個 session 是否能被另一個 channel、node 或 automation 看見，不必由每個 plugin 各自決定。

集中也帶來另一個結果：Gateway 成為高價值的 shared trust boundary。plugin 是否在同一個 process、node 能暴露哪些 command、session context 是否跨 channel 流動，都要和 Gateway 的 policy 一起檢查。名字叫 Gateway、plugin 或 node，不能單獨證明 process isolation 已經存在。

## 工作狀態怎麼留下來？

Agent state 至少有三種。第一種是下一次 model call 需要的 context；第二種是工具產生的工作產物，例如檔案、測試結果和 child output；第三種是可以跨過當前任務留下來的 memory、skill 或排程。三者的生命週期不一定相同。

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/state.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 persistent state 和 memory 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 6.</strong> 三個 runtime 保存的 state：Prime 以 Python namespace、session artifact 和 Continual Harness 為中心；Hermes 以 session database、Markdown memory 和 skills 為中心；OpenClaw 以 transcript、workspace、memory index 和 Gateway-owned state 為中心。來源：<a href="https://arxiv.org/abs/2605.09998">Continual Harness</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/blob/v2026.8.1/docs/concepts/memory-architecture.md">OpenClaw memory architecture</a>。</figcaption>
</figure>

| state 類型 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| turn／session | Python kernel、session artifact、daemon-backed session | AIAgent session、gateway transcript、SQLite session store | Gateway session、transcript、embedded runner state |
| 工作上下文 | Python variables、parsed results、task handles | conversation context、loaded skills、tool results | assembled context、compaction state、workspace files |
| 長期資料 | Continual Harness 的 prompt、memory、skill 和 subagent specification | <code>MEMORY.md</code>、<code>USER.md</code>、skills、FTS5 session search | <code>MEMORY.md</code>、<code>USER.md</code>、daily notes、SQLite memory index |
| 背景工作 | retained child、heartbeat、schedule、persistent goal | delegation、background task、cron | automation、standing intent、node／plugin run |
| resume 要確認的東西 | kernel state、child registry、檔案副作用 | session write、index、memory scope、terminal backend | session routing、transcript、memory provenance、plugin state |

Prime 的 state 直接貼近工作流。Python namespace 可以保留解析結果和 task handle，Continual Harness 再保存可以跨過一次 chat window 的 supplemental state。這讓長任務不必把所有中間資料重新塞進 prompt；resume 時則要分辨 live kernel state 和已寫入檔案是否都還成立。[^prime-rlm] [^continual-harness]

Hermes 的 state 主要服務一個長期使用的 personal service。session search 讓舊對話可以被找回，<code>MEMORY.md</code> 和 <code>USER.md</code> 提供較穩定的個人脈絡，skills 則把重複的操作留在可載入的能力中。這種設計方便跨入口使用，但資料寫入、索引和取回成了同一條 request path 上的可靠性問題。

OpenClaw 的 state 由 session transcript、workspace memory 和 Gateway 管理的資料組成。內建 memory architecture 將可讀的 Markdown 檔案和 SQLite index 分開，並在 recall 時處理 provenance、scope 和 admission。這讓 memory 可以被檢查和刪除，也讓 channel、agent、workspace 和 session 的來源必須一起保存。[^openclaw-memory]

三種 state 都可以抽象成：

~~~text
write
  -> organize or index
  -> retrieve
  -> inject into the next model call
~~~

差異在於 write 的 owner 和 retrieve 的時機。Prime 的 live Python state 可以直接被下一個 cell 使用；Hermes 需要 service 把 session、memory 和 skills 組回 AIAgent；OpenClaw 則要把 channel、session、workspace 和 memory policy 一起放進 Gateway 的 context assembly。

## 工具執行的權限在哪裡結束？

一個 action 要走過三個不同檢查點：

1. 模型提出 action。
2. policy、approval 或 pairing 決定 action 是否能進入執行路徑。
3. 某個 process、container 或 remote worker 實際執行 action。

這三步可能由不同 component 負責。approval 記錄「允許這個 action」，process isolation 需要另外驗證；pairing 確認「這個 device 或 sender 被承認」，command scope 仍要另外檢查。

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/security.png" width="1734" height="907" alt="Prime、Hermes、OpenClaw 的 execution、approval 和 isolation 邊界手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 7.</strong> execution、approval／admission 和 child／plugin boundary 分開檢查。生命週期隔離、OS／container sandbox 和身份授權各自回答不同問題。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md">Prime Agent README</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway docs</a>。</figcaption>
</figure>

Prime 的 Python kernel 執行 model-generated Python 和 project commands，使用 worker 的 operating-system permissions。Prime 的 worker、kernel 和 daemon 讓 session 能夠持續與恢復，RLM 文件卻明確把它們稱為 durable control environment，而非 security sandbox。檢查 Prime 的部署時，要沿著 kernel、shell bridge、host process 和 user environment 追檔案、網路與 credential。

Hermes 把 terminal backend 放在 tool path 的末端。README 同時列出 local、Docker、SSH、Singularity、Modal、Daytona 和 Vercel Sandbox，security 文件則把 command approval、DM pairing 和 container isolation 分開列出。Hermes 的 execution scope 取決於目前選用的 backend、mounted secrets、working directory 和 approval mode；AIAgent 的 approval request 只涵蓋其中一段。[^hermes-security]

OpenClaw 把 session permission mode、tool policy、exec approval、sandbox 和 node pairing 分成多個控制面。Gateway 文件指出，工具在主 session 預設可於 host 執行，sandbox 需要另外設定；node pairing 只會控制 device handshake 和 declared capability surface，node command 還要通過正常 command policy。檢查 OpenClaw 時，應分別確認 Gateway auth、channel identity、node capability、plugin process 和 sandbox workspace。[^openclaw-security] [^openclaw-pairing]

| 檢查層 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| execution | Python kernel、shell bridge、host process | 選定的 terminal backend | Gateway host、sandbox 或 node backend |
| admission | host safety policy、extension／skill path | command approval、tool policy、sender pairing | permission mode、exec policy、channel／node pairing |
| isolation | worker／kernel 的 lifecycle boundary；sandbox 需要另行配置 | local、container 或 remote backend 的實際隔離 | sandbox、host execution、plugin 和 node 的分層配置 |
| review 重點 | OS permissions、skill、credential、child result | backend、approval、mounted secret、working directory | Gateway auth、channel scope、node command、plugin 和 sandbox |

## 哪些工作可以平行？

平行化發生在 task level，不會消除單一回答的 token dependency。以 research task 為例，讀三批互不依賴的文件可以交給三個 child 或 worker；每個 child 自己的 model call 仍然沿著 autoregressive path 生成 token。

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/parallelism.png" width="1300" height="680" alt="Agent task-level parallelism 與 autoregressive decoder dependency 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 8.</strong> parent 可以把獨立工作分派給 child A、B、C，再收集結果；每個 response 的 token path 仍按序生成。來源：Prime 的 <a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">RLM</a>、Hermes 的 <a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">delegation</a>、OpenClaw 的 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">automation path</a>。</figcaption>
</figure>

Prime 用 <code>rlm.spawn(...)</code> 在 Python workspace 中建立 child session；Hermes 使用 delegation 和 background task；OpenClaw 則可以由 automation、plugin 或 node 啟動其他工作。這些介面負責多個工作分派；單一回答的 token 仍按 autoregressive path 生成。

如果有 <code>n</code> 個彼此獨立的子任務，順序執行時間可以寫成：

$$
T_{\text{serial}} = \sum_{i=1}^{n} T_i
$$

當資源足夠、子任務確實獨立，而且結果合併成本可接受時，平行執行時間才可能接近：

$$
T_{\text{parallel}} \approx \max_i(T_i) + T_{\text{dispatch}} + T_{\text{merge}}
$$

Prime 的 dispatch cost 出現在 child admission、context split 和 result message；Hermes 需要維持 background scope、session ownership 和 terminal environment；OpenClaw 還要把 plugin、node、channel 和 Gateway policy 放進同一條 lifecycle。平行化縮短的是 critical path，也同時增加了要追蹤的 state 和 trust surface。

## 設計取捨與適用情境

| 工作負載 | workload 形狀 | control surface | state owner | execution boundary | 適合的部署情境 | 需要承擔的成本 |
| --- | --- | --- | --- | --- | --- | --- |
| Prime Agent | 長時間 coding、research、反覆處理資料 | persistent Python／RLM | Python namespace、session artifact、Continual Harness | worker、kernel、shell bridge 和 host | 需要把工作流寫成程式並保留中間結果 | Python state recovery、child lifecycle、host permission |
| Hermes Agent | 個人長期使用、跨 CLI 和 messaging 的日常工作 | AIAgent service | session DB、FTS5、Markdown memory、skills | local、container 或 remote terminal backend | 需要 provider、memory、skills、cron 和多入口集中管理 | service scope、memory retrieval、terminal backend |
| OpenClaw 2.0 | 多 channel、device、plugin、automation | Gateway 加上 embedded runtime | transcript、workspace、memory index、Gateway state | host、sandbox、plugin 和 node | 需要由一個 control plane 統一 routing、session 和 policy | Gateway trust boundary、channel scope、plugin／node policy |

選 Prime 時，主要得到的是一個可以被模型持續操作的 programming surface。它適合中間結果多、子任務多、需要保留工作流的 coding 或 research；部署者要負責把 Python、shell、skills 和 host permissions 放進可恢復、可審查的環境。

選 Hermes 時，主要得到的是一個以 personal service 為中心的整合面。它適合每天從不同入口回到同一個 assistant；部署者要把 provider、session database、memory files、cron 和 terminal backend 當成同一個 service 的 state 來維護。

選 OpenClaw 時，主要得到的是 Gateway 對入口、session、tool、node 和 policy 的共同 ownership。它適合 channel 和外部元件很多的 self-hosted system；部署者要把 Gateway auth、channel scope、plugin trust、node capability 和 sandbox 設定放在同一份 review 裡。

部署 review 可以沿著四條線進行：

| review 問題 | 要留下的證據 |
| --- | --- |
| 下一個 action 由誰接手？ | loop owner、tool registry、runtime selection |
| 任務中斷後從哪裡恢復？ | session artifact、workspace、memory index、child／background state |
| action 最後在哪裡執行？ | process、container、remote worker、working directory |
| 誰可以讓它通過？ | policy、approval、pairing、credential scope |

四條線對應到三個 runtime 最容易出現的故障：工作恢復時遺失中間 state、記憶取回錯誤的 context，以及模型拿到超出預期的權限。

[^prime-readme]: [Prime Agent README at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md). 用於 Prime 的 coding、research、long-running work、RLM 和 Continual Harness 定位。
[^prime-rlm]: [Prime Agent RLM programming model at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md). 用於 persistent Python、child admission、child lifecycle、host bridge 和 trust model。
[^continual-harness]: [Continual Harness](https://arxiv.org/abs/2605.09998). 用於 durable prompt、memory、skill 和 subagent state 的 runtime framing。
[^hermes-readme]: [Hermes Agent README at commit afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md). 用於 Hermes 的 personal-agent scope、gateway、provider、memory、skills、cron、delegation 和 terminal backend。
[^hermes-loop]: [Hermes conversation loop at commit afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/agent/conversation_loop.py). 用於 AIAgent 的多輪 tool-call loop 與 tool result 交接。
[^hermes-security]: [Hermes security guide at commit afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/website/docs/user-guide/security.md). 用於 command approval、pairing、container isolation 和 terminal security scope。
[^openclaw-readme]: [OpenClaw README at v2026.8.1](https://github.com/openclaw/openclaw/blob/v2026.8.1/README.md). 用於 devices、chat channels、Gateway、models、tools、nodes 和 plugins 的 project scope。
[^openclaw-release]: [OpenClaw v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1). 這個比較把此 pinned release 稱為 OpenClaw 2.0。
[^openclaw-runtime]: [OpenClaw agent runtime architecture at v2026.8.1](https://github.com/openclaw/openclaw/blob/v2026.8.1/docs/agent-runtime-architecture.md). 用於 embedded runner、agent core、session wiring、tool definitions、harness registry 和 runtime selection。
[^openclaw-runtimes]: [OpenClaw agent runtimes at v2026.8.1](https://github.com/openclaw/openclaw/blob/v2026.8.1/docs/concepts/agent-runtimes.md). 用於 provider、model、agent runtime、harness 和 Gateway ownership 的分層。
[^openclaw-memory]: [OpenClaw memory architecture at v2026.8.1](https://github.com/openclaw/openclaw/blob/v2026.8.1/docs/concepts/memory-architecture.md). 用於 Markdown memory、SQLite index、provenance、recall 和 memory admission。
[^openclaw-security]: [OpenClaw gateway security documentation at v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway). 用於 session permission、tool policy、sandbox、exec approval 和 Gateway security boundary。
[^openclaw-pairing]: [OpenClaw node pairing at v2026.8.1](https://github.com/openclaw/openclaw/blob/v2026.8.1/docs/gateway/pairing.md). 用於 device pairing、node capability approval 和 command policy 的邊界。
