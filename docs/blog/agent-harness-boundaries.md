---
description: "從 Prime 的 persistent Python、Hermes 的 personal agent service，到 OpenClaw 的 Gateway control plane，拆解三個 agent runtime 如何處理控制流、狀態與權限。"
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
  <figcaption><strong>圖 1.</strong> 三個 runtime 的 model-facing control surface、工具路徑和 durable state。手繪圖整理自 <a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a> 和 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

一個聊天模型通常完成一次 completion。

它收到 prompt，生成 token，再把文字交回呼叫端。

Agent 的工作會繼續往下走。

模型產生 action 之後，系統要執行工具，把結果放回下一輪 context，保存 session，處理失敗，還要決定某個 action 能不能真的碰到檔案、網路或 credential。

模型周圍負責這些工作的程式、設定和執行環境，通常稱為 agent harness。

harness 描述模型如何接到外部世界。

runtime 描述這套 harness 如何長時間運作：request 從哪裡進來，哪一層持有 loop，state 寫到哪裡，工具在哪個 process 裡執行，以及 policy 在什麼位置攔截 action。

Prime Agent、Hermes Agent 和 OpenClaw 2.0 都處理這一層。

三個 project 的來源和目標不同。

Prime Agent 由 Prime Intellect 開發，README 從 coding、research 和 long-running work 談起。[^prime-readme]

Hermes Agent 由 Nous Research 開發，定位是可以自行部署的 personal agent，從 CLI、Telegram、Discord 或 ACP 接收工作。[^hermes-readme]

OpenClaw 由 OpenClaw Foundation 和社群維護，<code>v2026.8.1</code> 把 assistant 放在 chat channels、devices、plugins 和 automation 旁邊，由 Gateway 接住整個系統。[^openclaw-release]

版本固定為 Prime Agent commit <code>1fc1adb6</code>、Hermes Agent commit <code>afe06f2</code>，以及 OpenClaw <code>v2026.8.1</code> release。

文中所說的 OpenClaw 2.0 指這個 release。

## 一張表放在同一個座標系

三個 project 都有 model、tools、memory 和 loop。

差異集中在這些元件由哪一層持有，以及它們服務的工作單位。

| 比較面向 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| 作者／來源 | Prime Intellect | Nous Research | OpenClaw Foundation 和社群 |
| pinned version | <code>1fc1adb6</code> | <code>afe06f2</code> | <code>v2026.8.1</code> release |
| 原始工作負載 | coding、research、long-running work | self-hosted personal assistant | 多 channel、device、plugin、automation 的 self-hosted system |
| 工作單位 | 可以持續操作的 coding／research session | 一個人每天使用的 personal agent service | 由 Gateway 管理的多入口 agent system |
| request 入口 | coding／research request | CLI、Telegram、Discord、ACP、gateway | channel、CLI、paired node |
| model-facing control surface | persistent Python REPL／RLM | AIAgent core | Gateway session loop |
| loop owner | parent session 和 Python workspace | AIAgent core、prompt builder、provider resolver、tool registry | Gateway 的 session routing、agent loop 和 policy |
| tool execution | Python worker／kernel、files、shell、skills、MCP | terminal、web、MCP backend | native tools、plugins、nodes |
| child／background work | <code>rlm.spawn(...)</code>、child agent | delegation、cron、background task | automation、node、plugin path |
| durable state | Python namespace、workspace、Continual Harness state、session artifacts | session DB、SQLite／FTS5、<code>MEMORY.md</code>、<code>USER.md</code>、skills | workspace Markdown、SQLite／FTS5、retrieval、Gateway state |
| state 如何回到模型 | 同一個 persistent workspace 或下一次 session resume | session search、memory loading、prompt builder | session routing、workspace context、Gateway context assembly |
| action admission | parent／worker path、host bridge | approval pattern、service policy、terminal backend | pairing、Gateway policy、approval、plugin／node registration |
| isolation boundary | worker、kernel、host process 和部署設定 | local、container、remote terminal backend | Gateway、plugin、node 的 process 與 sandbox 設定 |
| task-level parallelism | <code>rlm.spawn(...)</code> | delegation、background work | automation、node、plugin |
| 主要架構貢獻 | 把模型放進可程式化、可持續的工作面 | 把 model loop、memory、skills 和多入口服務放在一起 | 把 session routing、policy 和外部元件收進 Gateway control plane |
| 主要成本 | workspace stale state、child lifecycle、host permission | memory retrieval、terminal scope、background job | cross-channel scope、plugin trust、Gateway 成為高價值 process |
| 適合的工作 | 長時間讀資料、寫程式、跑驗證 | 每天從不同入口使用同一個 assistant | 同時管理多入口、device、plugin 和 automation |

表中的 control surface 是比較核心。

它指模型產生下一個 action 時，實際面對的程式介面。

Prime 把這個介面做成 persistent Python／RLM。

Hermes 把它收在 AIAgent core 的 service loop。

OpenClaw 把它放進 Gateway 管理的 session loop。

同一個「呼叫工具」動作，落在三個位置之後，能看到的 state、能取得的 credential 和失敗後的恢復方式都會改變。

## 三條 request path

Prime 的入口進入 parent model，再往下進 persistent Python REPL／RLM。

Python 工作面可以讀檔案、跑 shell、載入 skills 或 MCP，也可以用 <code>rlm.spawn(...)</code> 拆出 child agent。

工作產物和 durable session state 留在同一個可重新接上的環境裡。

Hermes 的入口先進 gateway，再進 AIAgent core。

core 會組 prompt、解析 provider、查 tool registry，接著把 action 交給 terminal、web 或 MCP backend。

session database、FTS5、Markdown memory 和 skills 會在後續 request 被重新載入。

OpenClaw 的 channel、CLI 或 paired node 先進 Gateway WebSocket。

Gateway 做 session routing，再把 context、model、native tools、plugins 和 policy 串成一次 agent run。

結果寫回 workspace、SQLite／FTS5 或 retrieval layer，必要時再送回原本的 channel、node 或 automation。

~~~text
Prime:
request -> parent model -> persistent Python / RLM
       -> files / shell / skills / MCP / rlm.spawn(...)
       -> workspace / durable session

Hermes:
CLI / gateway / ACP -> AIAgent core
       -> prompt builder -> provider resolver -> tool registry
       -> terminal / web / MCP
       -> session DB / memory / skills

OpenClaw:
channel / CLI / node -> Gateway WebSocket
       -> session routing -> policy -> agent loop
       -> context -> model -> native tools / plugins / nodes
       -> workspace / SQLite / retrieval
~~~

這三條 path 的分界，決定 runtime 對「下一步」的解釋。

Prime 把下一步寫成 Python 工作流。

Hermes 把下一步當成 service loop 裡的一次 tool call。

OpenClaw 把下一步放在 Gateway 所有的 routing、session 和 policy 決策之後。

## Prime：persistent Python 是工作面

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/prime.png" width="1300" height="680" alt="Prime Agent 的 persistent Python RLM、工具和 child agent 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 2.</strong> Prime 的 model-facing surface 是 persistent Python／RLM；files、shell、skills、MCP 和 child agent 都從這個工作面接出去。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>。</figcaption>
</figure>

Prime 的 RLM 文件把 context 放進可以由 Python 操作的資料結構。

模型可以讀資料，把中間結果留在變數裡，再決定下一段程式、工具呼叫或 child-agent 工作。[^prime-rlm]

~~~text
request
  -> parent model
  -> persistent Python REPL / RLM
       -> files / shell / skills / MCP
       -> rlm.spawn(...)
       -> workspace / durable harness state
~~~

一般 tool-calling loop 把一次工具呼叫拆成幾個 host 端步驟：模型產生工具名稱和參數，host 執行工具，再把結果包回下一輪訊息。

RLM 將這些步驟放進可持續操作的 programming surface。

一個 research session 可以先讀一批文件，把解析結果留在 Python state，再把不同子問題交給 <code>rlm.spawn(...)</code>，最後把 child 結果合併成報告。

這個模型讓「下一步」具有程式結構。

條件分支、迴圈、暫存資料和子任務都可以留在工作面裡。

長任務因此少了幾次 context 重建，但 runtime 要負責更多 state。

變數可能指向過期檔案。

工具可能已經寫入檔案，parent 卻在收到結果前中斷。

child agent 可能只完成一半。

session resume 需要知道哪些 Python state、檔案副作用和 child result 已經成立。

### Prime 的 execution boundary

persistent Python 是執行介面。

sandbox 是否存在，取決於 worker、kernel、host bridge 和部署設定。

如果 Python process 能直接讀到使用者檔案、環境變數或 credential，模型寫出的程式也可能沿用同一組權限。

因此 Prime 的架構貢獻集中在 programming model。

權限邊界仍要沿著最後執行 action 的 process 追下去。

## Hermes：personal agent service 是控制中心

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/hermes.png" width="1300" height="680" alt="Hermes Agent 的 AIAgent loop、provider resolver、tool registry 和 memory 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 3.</strong> Hermes 把多個入口接到同一個 AIAgent loop，loop 再連到 provider、tool registry、terminal、web、MCP 和長期 state。來源：<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>。</figcaption>
</figure>

Hermes 的中心是長時間運作的 personal agent service。

CLI、Telegram、Discord、ACP 和其他 gateway 入口，把工作送進同一個 AIAgent core。

core 內的 prompt builder、provider resolver 和 tool registry 負責準備一次 model call。

~~~text
CLI / Telegram / Discord / ACP
  -> gateway
  -> AIAgent core
       -> prompt builder
       -> provider resolver
       -> tool registry
  -> terminal / web / MCP
  -> session and memory
~~~

這個位置讓 Hermes 可以把 provider switching、session search、memory、skills 和 cron 放在同一個 service 裡。

模型不必知道 request 來自哪個 channel。

channel 也不必各自複製一套 agent loop。

Hermes 的長期狀態由幾條路徑組成：

~~~text
write -> index -> retrieve -> inject into the next model call
~~~

寫入失敗時，使用者以為已經保存的內容根本不存在。

索引沒有更新時，內容存在卻找不到。

取回結果太寬時，過期或不相關的記憶會進入 prompt。

注入過多時，歷史資料會吃掉當前任務的 context budget。

<code>MEMORY.md</code>、<code>USER.md</code> 和 skills 也有 scope 問題。

同一份個人設定要能跨 session 重用，又要避免不同 user、channel 或 deployment 互相污染。

Hermes 的 terminal execution 可以接 local process、container 或 remote backend。

approval pattern 控制 action 能否進入執行路徑。

真正的 process isolation 仍由 terminal backend 和部署環境決定。

同一個 AIAgent core 接上本機 shell 和受限 container，信任邊界完全不同。

## OpenClaw：Gateway 管理整個 control plane

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/openclaw.png" width="1300" height="680" alt="OpenClaw 2.0 的 Gateway WebSocket、session routing、policy、plugins、automation 和 workspace 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 4.</strong> OpenClaw 把 channels、CLI、nodes、plugins 和 automation 接到 Gateway；workspace、SQLite／FTS5 和 retrieval 由 Gateway path 保存與取回。來源：<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">Gateway docs</a> 和 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1">v2026.8.1 source</a>。</figcaption>
</figure>

OpenClaw 的 request 先處理入口和 session，再進 model loop。

Gateway 要先知道 request 屬於哪個 session、哪個 user、哪個 node，以及這次 action 能使用哪些 policy。

~~~text
channel / CLI / node
  -> Gateway WebSocket
  -> session routing and policy
  -> agent loop
       -> context
       -> model
       -> native tools
  -> workspace / SQLite / retrieval
~~~

這個 control plane 把 channel coordination、session ownership、policy enforcement 和 plugin／node 接入放在一起。

cron 和 automation 可以在沒有即時聊天的情況下啟動工作。

plugin 和 node 則把外部能力帶進 Gateway。

多入口系統的困難在 scope。

Telegram、Web UI 和 paired device 可以共用一個 assistant，也可以各自擁有獨立 session。

共用 session 時，Gateway 要限制哪些 context 可以跨入口流動，哪些 action 只能由特定 node 或使用者核准。

plugin 和 node 的結果也要回到正確的 channel。

pairing 能確認來源和身份。

approval 能控制某個 action 是否進入執行路徑。

這兩件事都不能單獨描述 plugin 是否隔離，或 node 最後能碰到哪些檔案、網路和 credential。

在固定的 <code>v2026.8.1</code> release 裡，plugin execution、sandbox 設定和 node 權限仍要分開檢查。[^openclaw-security]

## Control surface：三個 loop 序列化不同的東西

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/control-surface.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 三種 per-session control loop 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 5.</strong> 三個 project 的 per-session loop：Prime 序列化 Python cell、host request 和 child session；Hermes 序列化 tool call、delegation 和 summary；OpenClaw 序列化 intake、context、native tools 和 persist。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

Prime 的一個 session 以 Python cell 為起點。

Python cell 觸發 host request，host request 可能建立 child session。

這條線把模型可編程的工作面和外部 process 接在一起。

Hermes 的一個 session 從 tool call 開始。

<code>delegate_task</code> 把工作交給另一個 agent 或 background path，結果再以 summary 回到主要對話。

這條線把 delegation 收在 personal service 的 loop 裡。

OpenClaw 的一個 session 從 intake 開始。

Gateway 組合 context 和 model，再執行 native tools，最後把結果 persist。

這條線把每個入口的工作收進 Gateway-owned run。

三者的「session」都可以長時間存在，session 內部實際被序列化的物件不同。

Prime 序列化 programming state。

Hermes 序列化 tool／product loop 和 personal memory。

OpenClaw 序列化 Gateway 管理的 request、policy、plugin 和 persistence。

## State：持久化資料決定工作會留下什麼

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/state.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 persistent state 和 memory 手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 6.</strong> 三種 state path：Prime 以 persistent Python namespace 和 Continual Harness 為中心；Hermes 以 session DB、Markdown memory 和 skills 為中心；OpenClaw 以 workspace、retrieval 和 plugin／context engine 為中心。來源：<a href="https://arxiv.org/abs/2605.09998">Continual Harness</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1">OpenClaw source</a>。</figcaption>
</figure>

三個 runtime 都把有用資料保存到 model weights 之外。

Prime 保存 Python workspace、prompts、memories、skill descriptions、child specs 和工作產物。

Continual Harness 將 supplemental prompt、memory、skill description 和可重用的 subagent specification 保存成 durable state，讓工作規則跨過一次 chat window。[^continual-harness]

Hermes 保存 session history、SQLite／FTS5 index、<code>MEMORY.md</code>、<code>USER.md</code> 和 skills。

OpenClaw 保存 workspace Markdown、SQLite／FTS5 和 retrieval state。

這些資料會影響下一輪輸入，模型參數維持原狀。

三個 state lifecycle 都可以寫成：

~~~text
write -> index or organize -> retrieve -> inject
~~~

Prime 的 retrieve 多半發生在同一個 persistent workspace，或下一次 session resume。

Hermes 需要 session search、personal context 和 skill loading。

OpenClaw 需要 session routing、workspace context 和 hybrid retrieval。

真正需要驗證的是 state scope。

Prime 要確認 workspace resume 時不會帶入 stale variable 或重做已完成的副作用。

Hermes 要確認個人 memory、session index 和 channel scope 能正確對齊。

OpenClaw 要確認跨入口共享的 context 沒有把不該流動的資料送到另一個 channel 或 node。

## Task-level parallelism 和 decoder-level dependency

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/parallelism.png" width="1300" height="680" alt="Agent task-level parallelism 與 autoregressive decoder dependency 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 7.</strong> parent 可以把獨立工作 fan-out 給 child A、B、C，再收集結果；單一回答的 token path 仍沿著 t1、t2、t3、t4、t5 依序生成。來源：Prime 的 <a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">RLM</a>、Hermes 的 <a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">delegation</a>、OpenClaw 的 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">automation path</a>。</figcaption>
</figure>

Agent project 裡的 <code>rlm.spawn(...)</code>、delegation、background task 和 automation 都會增加 task-level parallelism。

它們把獨立的 research、coding、maintenance 或 scheduled work 分派給不同 worker、child session 或 node。

單一 autoregressive sequence 的 token dependency 仍然存在。

如果有 $n$ 個彼此獨立的子任務，理想化的順序執行時間接近：

$$
T_{serial} = \sum_{i=1}^{n} T_i
$$

資源足夠、子任務真的獨立，而且 merge 成本可接受時，平行執行才可能接近：

$$
T_{parallel} \approx \max_i(T_i) + T_{dispatch} + T_{merge}
$$

Prime 的 <code>rlm.spawn(...)</code> 把 fan-out 接到 persistent Python workspace。

Hermes 的 delegation 把子任務接到 personal agent service。

OpenClaw 的 automation、node 和 plugin 把背景工作接到 Gateway control plane。

三者都能縮短多任務的 wall-clock time。

單一回答的第 $t+1$ 個 token 仍依賴第 $t$ 個 token。

## Execution、Admission、Isolation

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/security.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 execution、approval 和 isolation 邊界手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 8.</strong> execution、approval／admission 和 child／plugin boundary 分開檢查。三個 project 的 lifecycle、policy 和 process isolation 落在不同位置。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

安全分析要沿著 action 的實際路徑走。

model output 只表示模型提出了一個 action。

接下來要確認三件事：

1. action 最後在哪個 process、container 或 remote worker 執行。
2. 哪個 component 允許它通過 approval、pairing 或 policy。
3. 執行 process 能看到哪些檔案、credential、網路和 session。

Prime 要追 Python worker／kernel、shell bridge、host process 和 user environment。

Hermes 要追 terminal backend、approval path、mounted secret、provider config 和 deployment。

OpenClaw 要追 Gateway、plugin、node、channel identity、pairing 和 policy。

worker 名稱不能直接代表 sandbox。

container 名稱也不能直接代表 credential 已經隔離。

pairing 名稱則不能直接代表 plugin 擁有獨立 process。

這些元件需要沿著 process boundary 和 credential path 實際驗證。

## 三個 project 各自改變了哪一層

Prime 改變的是模型操作電腦的 programming surface。

它把長時間工作需要的 context、暫存資料、工具組合和 child task 放進 persistent Python／RLM。

Hermes 改變的是 personal assistant 的 service boundary。

它把多入口、provider、memory、skills、terminal backend、delegation 和 cron 收進同一個可長期運作的服務。

OpenClaw 改變的是多入口 agent system 的 control plane。

它把 session routing、channels、nodes、plugins、automation 和 policy 放在 Gateway path 內。

評估部署時，固定標出三個 owner：model action 交給哪個 loop，state 由哪個 component 寫入、索引、取回和注入，以及 tool action 最後在哪個 process、credential scope 和 policy 下執行。

這三條線會直接指出 session resume、memory scope 和 credential isolation 的測試位置。

[^prime-readme]: [Prime Agent README at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md). 用於 project scope、long-running work、coding 和 research 定位。
[^prime-rlm]: [Prime Agent RLM programming model at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md). 用於 persistent Python surface、context variables、child-agent lifecycle 和 host bridge。
[^continual-harness]: [Continual Harness](https://arxiv.org/abs/2605.09998). 用於 durable prompt、memory、skill 和 subagent state 的 runtime framing；這些資料與 model-weight update 分開。
[^hermes-readme]: [Hermes Agent README at commit afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md). 用於 personal-agent scope、gateway、memory/search、skills、cron、delegation 和 terminal backend。
[^openclaw-release]: [OpenClaw v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1). 此比較把這個 pinned release 稱為 OpenClaw 2.0。
[^openclaw-security]: [OpenClaw gateway documentation at v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway). 用於 Gateway routing、policy、pairing、approval 和 deployment-bound security observations。
