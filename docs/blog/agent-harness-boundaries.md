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

聊天模型處理一次 completion：prompt 進來，token 生成，文字回去。Agent 要把這個 completion 接成一段持續工作的流程。

模型產生 action 後，系統要執行工具，把結果放回下一輪 context，保存 session，處理失敗，還要決定它能不能碰到檔案、網路或 credential。

模型周圍負責這些工作的程式、設定和執行環境，通常稱為 agent harness。這個 harness 決定模型如何接到外部世界。

runtime 描述 harness 如何長時間運作：request 從哪裡進來、哪個 component 持有 loop、state 寫到哪裡、工具在哪個 process 執行，以及 policy 在什麼位置攔截 action。Prime 把中心放到 persistent Python／RLM，Hermes 放到 AIAgent service，OpenClaw 放到 Gateway control plane。

Prime Agent 由 Prime Intellect 開發，README 從 coding、research 和 long-running work 談起；Hermes Agent 由 Nous Research 開發，定位是可以自行部署的 personal agent；OpenClaw 由 OpenClaw Foundation 和社群維護，<code>v2026.8.1</code> 把 assistant 放在 chat channels、devices、plugins 和 automation 旁邊，由 Gateway 接住整個系統。[^prime-readme] [^hermes-readme] [^openclaw-release]

版本固定為 Prime Agent commit <code>1fc1adb6</code>、Hermes Agent commit <code>afe06f2</code>，以及 OpenClaw <code>v2026.8.1</code> release；所有優勢與代價都以這三個 source snapshot 為準。

## 三個 runtime 的定位

Prime、Hermes 和 OpenClaw 都有 model、tools、memory 和 loop。真正不同的是，哪一層接住模型的下一個 action，以及哪一層負責把工作留下來。

| 比較面向 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| 作者／來源 | Prime Intellect | Nous Research | OpenClaw Foundation 和社群 |
| 原始工作負載 | coding、research、long-running work | self-hosted personal assistant | 多 channel、device、plugin、automation 的 self-hosted system |
| 工作單位 | 可以持續操作的 coding／research session | 一個人每天使用的 personal agent service | 由 Gateway 管理的多入口 agent system |
| control surface | persistent Python REPL／RLM | AIAgent core | Gateway session loop |
| state | Python namespace、workspace、Continual Harness state | session DB、SQLite／FTS5、<code>MEMORY.md</code>、<code>USER.md</code>、skills | workspace Markdown、SQLite／FTS5、retrieval、Gateway state |
| 工具與背景工作 | files、shell、MCP、<code>rlm.spawn(...)</code> | terminal、web、MCP、delegation、cron | native tools、plugins、nodes、automation |
| 主要優勢 | 工作流可編程，中間結果留在同一個工作面 | 入口、provider、memory、skills 和 cron 共用一個 service | session routing、policy 和外部元件由 Gateway 統一管理 |
| 主要代價 | workspace stale state、child lifecycle、host permission | memory retrieval、terminal scope、background job | cross-channel scope、plugin trust、Gateway 成為高價值 trust boundary |
| 適合的工作 | 長時間讀資料、寫程式、跑驗證 | 每天從不同入口使用同一個 assistant | 同時管理多入口、device、plugin 和 automation |

control surface 是這張比較的中心。模型發出同一個工具 action，Prime 讓它進 persistent Python，Hermes 讓它進 AIAgent core，OpenClaw 讓它先經過 Gateway 的 session routing 和 policy。這個位置決定模型能看到哪些 state、工具能取得哪些 credential，以及失敗後由誰負責恢復。

## 三條 request path

Prime 的入口進入 parent model，再往下進 persistent Python REPL／RLM。工作流、中間結果和 child task 都留在同一個可以重新接上的環境裡。

Hermes 的入口先進 gateway，再進 AIAgent core。core 組 prompt、解析 provider、查 tool registry，接著把 action 交給 terminal、web 或 MCP backend；session database、FTS5、Markdown memory 和 skills 服務後續 request。

OpenClaw 的 channel、CLI 或 paired node 先進 Gateway WebSocket。Gateway 做 session routing，再把 context、model、native tools、plugins 和 policy 串成一次 agent run；結果寫回 workspace、SQLite／FTS5 或 retrieval layer。

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

三條 path 的差異在於「下一步」由誰保存。Prime 保存一段可以繼續執行的 Python 工作流，Hermes 保存一次 service loop 的狀態，OpenClaw 保存 Gateway 對 request、session 和 policy 的決策。

## Prime：persistent Python 是工作面

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/prime.png" width="1300" height="680" alt="Prime Agent 的 persistent Python RLM、工具和 child agent 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 2.</strong> Prime 的 model-facing surface 是 persistent Python／RLM。files、shell、skills、MCP 和 child agent 都從這個工作面接出去。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>。</figcaption>
</figure>

Prime 的 RLM 文件把 context 放進可以由 Python 操作的資料結構。模型可以讀資料，把中間結果留在變數裡，再決定下一段程式、工具呼叫或 child-agent 工作；這是 Prime 最重要的 programming surface。[^prime-rlm]

~~~text
request
  -> parent model
  -> persistent Python REPL / RLM
       -> files / shell / skills / MCP
       -> rlm.spawn(...)
       -> workspace / durable harness state
~~~

一般 tool-calling loop 把一次工具呼叫拆成幾個 host 端步驟：模型產生工具名稱和參數，host 執行工具，再把結果包回下一輪訊息。RLM 把這些步驟放到可以持續操作的 Python 工作面裡。

一次 research session 可以先讀一批文件，把解析結果留在 Python state，再用 <code>rlm.spawn(...)</code> 拆分子問題，最後合併 child 結果。條件、迴圈、暫存資料和子任務都能留在同一個工作面裡。

這種設計減少長任務的 context 重建，也把更多可靠性責任交給 workspace。變數可能指向過期檔案，工具可能已經寫入檔案但 parent 還沒收到結果，child agent 也可能只完成一半。

session resume 必須知道哪些 Python state、檔案寫入和 child result 已經成立。這是 Prime 的 programming freedom 所換來的 recovery 責任。

### Prime 的 execution boundary

persistent Python 是 Prime 的執行介面。sandbox 是否存在，取決於 worker、kernel、host bridge 和部署設定。

如果 Python process 能讀到使用者檔案、環境變數或 credential，模型寫出的程式也可能沿用同一組權限。Prime 的 architecture contribution 在 programming model，部署時則必須沿著最後執行 action 的 process 追查權限。

## Hermes：personal agent service 是控制中心

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/hermes.png" width="1300" height="680" alt="Hermes Agent 的 AIAgent loop、provider resolver、tool registry 和 memory 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 3.</strong> Hermes 把多個入口接到同一個 AIAgent loop。provider、tool registry、terminal、web、MCP 和長期 state 都收在 personal service。來源：<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>。</figcaption>
</figure>

Hermes 的中心是長時間運作的 personal agent service。CLI、Telegram、Discord、ACP 和其他 gateway 入口都送進同一個 AIAgent core。

core 內的 prompt builder、provider resolver 和 tool registry 負責準備一次 model call。這些元件把 provider、工具和對話狀態收進同一個 service，讓日常維護有明確的 owner。

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

這個 service 可以把 provider switching、session search、memory、skills 和 cron 放在同一個位置。Hermes 的優勢是日常使用集中；代價是模型的工作流被收在 service code 裡，客製化通常要透過 tool、skill 或 delegation path 擴充。

Hermes 的長期狀態由 session、搜尋索引、Markdown context 和 skills 組成。它保存 personal service 的使用脈絡；Python workspace 則可以直接繼續執行工作。

~~~text
write -> index -> retrieve -> inject into the next model call
~~~

寫入失敗會讓使用者以為資料已保存，索引失敗會讓存在的內容消失在搜尋結果裡，取回過寬則會把過期記憶塞進 prompt。這些 memory failure 是 Hermes 最主要的 reliability 成本。

<code>MEMORY.md</code>、<code>USER.md</code> 和 skills 的 scope 必須和 session 對齊。Hermes 的 terminal execution 可以接 local process、container 或 remote backend，替換執行後端是它的另一個優點。

approval pattern 只決定 action 能不能進入執行路徑，真正的 process isolation 仍由 terminal backend 和部署環境決定。AIAgent core 接上本機 shell 和受限 container 後，信任邊界並不相同。

## OpenClaw：Gateway 管理整個 control plane

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/openclaw.png" width="1300" height="680" alt="OpenClaw 2.0 的 Gateway WebSocket、session routing、policy、plugins、automation 和 workspace 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 4.</strong> OpenClaw 把 channels、CLI、nodes、plugins 和 automation 接到 Gateway。workspace、SQLite／FTS5 和 retrieval 都由 Gateway path 保存與取回。來源：<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">Gateway docs</a> 和 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1">v2026.8.1 source</a>。</figcaption>
</figure>

OpenClaw 的 request 先處理入口和 session，再進 model loop。Gateway 把 user、node、channel 和 policy 的 ownership 放在同一個 control plane。

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

這個 control plane 把 channel coordination、session ownership、policy enforcement 和 plugin／node 接入放在一起。cron 和 automation 可以在沒有即時聊天的情況下啟動工作，OpenClaw 因此適合長期 background orchestration。

多入口的難題是 scope。OpenClaw 要處理 Telegram、Web UI 和 paired device 之間的共享或隔離。共用 session 時，Gateway 必須限制 context 能跨哪些入口流動，並把 plugin、node 的結果送回正確 channel。

pairing 能確認來源和身份，approval 能控制 action 是否進入執行路徑，但兩者都不能單獨證明 plugin 或 node 已經隔離。固定在 <code>v2026.8.1</code> 時，OpenClaw 仍要分開檢查 plugin execution、sandbox 和 node 權限。[^openclaw-security]

## Control surface：三個 loop 序列化不同的東西

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/control-surface.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 三種 per-session control loop 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 5.</strong> 三個 project 的 per-session loop：Prime 序列化 Python cell、host request 和 child session；Hermes 序列化 tool call、delegation 和 summary；OpenClaw 序列化 intake、context、native tools 和 persist。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

Prime 的 session 從 Python cell 開始。cell 觸發 host request，也可能建立 child session；被序列化的是 programming state。

Hermes 的 session 從 tool call 開始。<code>delegate_task</code> 把工作交給另一個 agent，再用 summary 回到主要對話；service code 因此能觀察整個 loop。

OpenClaw 的 session 從 intake 開始。Gateway 組合 context、model、native tools 和 persistence，把每個入口的工作收進同一個 Gateway-owned run。

三個 session 的差異很直接：Prime 把可編程性放在 session 裡，Hermes 把可觀察性放在 service 裡，OpenClaw 把 ownership 放在 Gateway 裡。對應的成本是 recovery、service state 和 shared trust surface。

## State：持久化資料決定工作會留下什麼

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/state.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 persistent state 和 memory 手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 6.</strong> 三種 state path：Prime 以 persistent Python namespace 和 Continual Harness 為中心；Hermes 以 session DB、Markdown memory 和 skills 為中心；OpenClaw 以 workspace、retrieval 和 plugin／context engine 為中心。來源：<a href="https://arxiv.org/abs/2605.09998">Continual Harness</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1">OpenClaw source</a>。</figcaption>
</figure>

Prime 把工作資料放在 Python workspace、prompts、memories、skills、child specs 和工作產物裡。Continual Harness 再把 supplemental prompt、memory、skill description 和 subagent specification 保存成可以跨過一次 chat window 的 durable state。[^continual-harness]

Hermes 把長期狀態放在 session history、SQLite／FTS5、<code>MEMORY.md</code>、<code>USER.md</code> 和 skills。這些資料服務的是 personal assistant 的連續使用。

OpenClaw 把 workspace Markdown、SQLite／FTS5 和 retrieval state 接到 Gateway。這些資料服務的是跨 channel、node 和 automation 的 context assembly。

Prime、Hermes 和 OpenClaw 的 state lifecycle 都可以寫成：

~~~text
write -> index or organize -> retrieve -> inject
~~~

Prime 的 retrieve 多半發生在 persistent workspace 或 session resume。要測的是 stale variable、重複副作用，以及 child result 是否能正確恢復。

Hermes 需要 session search、personal context 和 skill loading。要測的是 memory 寫入、索引、取回和 channel scope 是否對齊。

OpenClaw 需要 session routing、workspace context 和 hybrid retrieval。要測的是 context 是否跨到錯誤的 channel 或 node。

## Task-level parallelism 和 decoder-level dependency

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/parallelism.png" width="1300" height="680" alt="Agent task-level parallelism 與 autoregressive decoder dependency 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 7.</strong> parent 可以把獨立工作 fan-out 給 child A、B、C，再收集結果；單一回答的 token path 仍沿著 t1、t2、t3、t4、t5 依序生成。來源：Prime 的 <a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">RLM</a>、Hermes 的 <a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">delegation</a>、OpenClaw 的 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">automation path</a>。</figcaption>
</figure>

Agent-level parallelism 發生在 decoder 之外。只要工作可以拆成互不依賴的子任務，runtime 就能把它們分派給不同 worker、child session 或 node。

Prime 用 <code>rlm.spawn(...)</code> 從 Python workspace 做 fan-out。Hermes 用 delegation 和 background task；OpenClaw 用 automation、node 和 plugin。

三個 runtime 都保留單一回答的 autoregressive dependency。平行化改善多個工作之間的 wall-clock time，同一個 response 的 token 仍按順序生成。

如果有 $n$ 個彼此獨立的子任務，順序執行時間可以先寫成：

$$
T_{serial} = \sum_{i=1}^{n} T_i
$$

資源充足、子任務真的獨立，而且 merge 成本可接受時，平行執行時間才可能接近：

$$
T_{parallel} \approx \max_i(T_i) + T_{dispatch} + T_{merge}
$$

Prime 的成本在 child lifecycle 和結果合併。Hermes 的成本在 background scope。OpenClaw 的成本在 plugin／node trust，以及 Gateway 對背景工作的管理。

## Execution、Admission、Isolation

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/security.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 execution、approval 和 isolation 邊界手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 8.</strong> execution、approval／admission 和 child／plugin boundary 分開檢查。三個 project 的 lifecycle、policy 和 process isolation 落在不同位置。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

model output 只表示模型提出了一個 action。安全邊界要沿著 action 的實際路徑追下去，分開看 execution、admission 和 isolation。

三個檢查點是：

1. action 最後在哪個 process、container 或 remote worker 執行。
2. 哪個 component 允許它通過 approval、pairing 或 policy。
3. 執行 process 能看到哪些檔案、credential、網路和 session。

Prime 要追 Python worker／kernel、shell bridge、host process 和 user environment。

Hermes 要追 terminal backend、approval path、mounted secret、provider config 和 deployment。

OpenClaw 要追 Gateway、plugin、node、channel identity、pairing 和 policy。worker 名稱不能直接代表 sandbox，container 名稱不能直接代表 credential 已經隔離，pairing 名稱也不能直接代表 plugin 擁有獨立 process。

Prime 的 programming freedom 帶來 host permission 風險。Hermes 的 service convenience 帶來 backend scope 風險。OpenClaw 的 integration breadth 帶來更大的 shared trust surface。

## 怎麼選

需要一個可以長時間讀資料、寫程式、跑驗證的工作面時，Prime 的 persistent Python／RLM 最直接。它把控制流交給模型，也把 workspace recovery、child lifecycle 和 host permission 留給部署者。

需要每天從 CLI 或聊天入口使用同一個 personal assistant 時，Hermes 的 service boundary 比較合適。它把 provider、memory、skills、terminal 和 cron 收在一起，代價是 memory correctness 和 backend scope 需要持續維護。

需要把多個 channel、device、plugin 和 automation 接在同一個系統裡時，OpenClaw 的 Gateway 比較合適。它集中 session routing 和 policy，但 Gateway、plugin、node 和 channel identity 會形成更大的 trust surface。

部署 review 時，先把三個 owner 標出來：model action 交給哪個 loop，state 由哪個 component 寫入與取回，以及 tool action 最後在哪個 process、credential scope 和 policy 下執行。這三條線就是 session resume、memory scope 和 credential isolation 的測試邊界。

[^prime-readme]: [Prime Agent README at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md). 用於 project scope、long-running work、coding 和 research 定位。
[^prime-rlm]: [Prime Agent RLM programming model at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md). 用於 persistent Python surface、context variables、child-agent lifecycle 和 host bridge。
[^continual-harness]: [Continual Harness](https://arxiv.org/abs/2605.09998). 用於 durable prompt、memory、skill 和 subagent state 的 runtime framing；這些資料與 model-weight update 分開。
[^hermes-readme]: [Hermes Agent README at commit afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md). 用於 personal-agent scope、gateway、memory/search、skills、cron、delegation 和 terminal backend。
[^openclaw-release]: [OpenClaw v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1). 此比較把這個 pinned release 稱為 OpenClaw 2.0。
[^openclaw-security]: [OpenClaw gateway documentation at v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway). 用於 Gateway routing、policy、pairing、approval 和 deployment-bound security observations。
