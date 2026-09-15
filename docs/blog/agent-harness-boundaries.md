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

先把模型本身和 agent 系統分開看。聊天模型的基本迴路是 prompt 進來、token 生成、completion 回去；Prime Agent、Hermes Agent 和 OpenClaw 2.0 真正增加的，是 completion 之後還能不能繼續工作。

這個「繼續」包含的不只 API 數量。模型產生 action 後，Prime、Hermes 和 OpenClaw 都要把 action 交給工具，把結果放回下一輪 context，保存 session，處理失敗，還要決定它能不能碰到檔案、網路或 credential。

模型周圍負責這些工作的程式、設定和執行環境，通常稱為 agent harness。Prime 把 harness 的中心放到 persistent Python／RLM，Hermes 放到 AIAgent service，OpenClaw 放到 Gateway control plane；三者的架構差異從這裡開始。

runtime 是 harness 在長時間工作中的運作方式：request 從哪裡進來、哪個 component 持有 loop、state 寫到哪裡、工具在哪個 process 執行，以及 policy 在什麼位置攔截 action。Prime、Hermes 和 OpenClaw 都需要這些元件，但 ownership 落在不同層。

Prime Agent 由 Prime Intellect 開發，README 從 coding、research 和 long-running work 談起；Hermes Agent 由 Nous Research 開發，定位是可以自行部署的 personal agent；OpenClaw 由 OpenClaw Foundation 和社群維護，<code>v2026.8.1</code> 把 assistant 放在 chat channels、devices、plugins 和 automation 旁邊，由 Gateway 接住整個系統。[^prime-readme] [^hermes-readme] [^openclaw-release]

版本固定為 Prime Agent commit <code>1fc1adb6</code>、Hermes Agent commit <code>afe06f2</code>，以及 OpenClaw <code>v2026.8.1</code> release；所有優勢與代價都以這三個 source snapshot 為準。

## 一張表放在同一個座標系

Prime、Hermes 和 OpenClaw 都有 model、tools、memory 和 loop；差異在於這些元件由哪一層持有，以及各自把哪種工作當成基本單位。

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
| 核心優勢 | Python 可以表達條件、迴圈、資料處理和 child workflow；中間結果留在同一個工作面 | 開箱即用的 personal service；入口、provider、memory、skills 和 cron 共用一個 core | 入口、session、node、plugin、automation 和 policy 有同一個 Gateway owner |
| 核心弱點 | stateful workspace 讓 resume、side effect 和 host permission 變成使用者要承擔的工程問題 | tool registry 和 service policy 收斂了控制流；memory correctness、terminal scope 和 background job 仍要維護 | Gateway、plugin 和 node 形成較大的 trust surface；identity、session scope 和部署隔離需要一起管理 |
| 主要成本 | workspace stale state、child lifecycle、host permission | memory retrieval、terminal scope、background job | cross-channel scope、plugin trust、Gateway 成為高價值 process |
| 適合的工作 | 長時間讀資料、寫程式、跑驗證 | 每天從不同入口使用同一個 assistant | 同時管理多入口、device、plugin 和 automation |

真正拉開 Prime、Hermes 和 OpenClaw 差距的是 control surface。它指模型產生下一個 action 時實際面對的程式介面：Prime 把它做成 persistent Python／RLM，Hermes 把它收在 AIAgent core 的 service loop，OpenClaw 把它放進 Gateway 管理的 session loop。

同一個「呼叫工具」動作，交給 Prime 的 workspace、Hermes 的 service 或 OpenClaw 的 Gateway 後，模型能看到的 state、工具能取得的 credential，以及失敗後的恢復方式都會改變。

## 優勢和代價放在一起看

Prime 的優勢在 programming surface。模型可以用 Python 保存中間結果，安排條件、迴圈和資料處理，再把獨立工作交給 child agent；相較 Hermes 的 tool registry 和 OpenClaw 的 Gateway loop，Prime 給模型更多工作流組合空間，也把更多可靠性責任留在 workspace。

Prime 適合長時間 coding 或 research session；Hermes 把相同需求拆到 service state，OpenClaw 則交給 Gateway 和 automation。Prime 的代價是 workspace 可能累積 stale state，工具可能留下部分副作用，child session 需要回報和合併，Python worker 還可能沿用 host 的檔案和 credential。

Hermes 的優勢在 service boundary。它把 CLI、聊天入口、provider、memory、skills、terminal 和 cron 接到同一個 AIAgent core；相較 Prime 要自己維持 programming workspace，Hermes 比較接近日常可用的 personal service，相較 OpenClaw 則把問題限制在單一使用者的 service scope。

Hermes 的集中化讓日常使用和 service-level observability 比較直接；Prime 把較多 workflow state 留在 workspace，OpenClaw 把較多責任推到 Gateway。Hermes 的代價是模型的工作流被收在 tool registry、provider resolver 和 service policy 裡，memory 的寫入、索引、取回和注入也成為 reliability 的主要來源。

OpenClaw 的優勢在 system boundary。Gateway 可以把 channel、CLI、paired node、plugin、automation、session routing 和 policy 接在同一條 path 上；它比 Prime 更適合外部元件整合，也比 Hermes 承接更廣的入口和 device scope。

OpenClaw 的代價是 Gateway 變成整個系統的高價值 trust boundary；Prime 的風險較集中在 host bridge 和 workspace，Hermes 的風險較集中在 terminal backend 和 personal service。

對 OpenClaw 來說，channel identity、session scope、plugin process、node capability 和 credential path 要同時正確；Prime 和 Hermes 的元件範圍較窄，但各自的 host、terminal、memory scope 仍需要單獨驗證。

Prime 對 Hermes 的取捨是可編程工作面換取較高的 state 和部署責任；Hermes 對 OpenClaw 的取捨是 personal service 的範圍換取較小的 control-plane 複雜度；Prime 對 OpenClaw 的取捨則是長任務連續性換取多入口整合能力。

因此，Prime 的價值集中在工作流可編程性，Hermes 的價值集中在 personal service 的完整度，OpenClaw 的價值集中在整合面和 Gateway ownership。

## 三條 request path

Prime 的入口進入 parent model，再往下進 persistent Python REPL／RLM；這條 path 把工作流和中間結果留在同一個可重新接上的環境裡，控制力高於 Hermes 的 tool registry，也比 OpenClaw 更靠近 model-facing workspace。

Hermes 的入口先進 gateway，再進 AIAgent core；core 組 prompt、解析 provider、查 tool registry，接著把 action 交給 terminal、web 或 MCP backend，session database、FTS5、Markdown memory 和 skills 則服務後續 request，日常連續性比 Prime 更產品化，入口範圍比 OpenClaw 更收斂。

OpenClaw 的 channel、CLI 或 paired node 先進 Gateway WebSocket；Gateway 做 session routing，再把 context、model、native tools、plugins 和 policy 串成一次 agent run，結果寫回 workspace、SQLite／FTS5 或 retrieval layer，相較 Prime 和 Hermes，整合範圍最大，也把最多 trust 和 scope 責任集中在 Gateway。

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

runtime 對「下一步」的解釋也不同：Prime 把它寫成 Python 工作流，Hermes 把它當成 service loop 裡的一次 tool call，OpenClaw 把它放在 Gateway 所有的 routing、session 和 policy 決策之後。

## Prime：persistent Python 是工作面

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/prime.png" width="1300" height="680" alt="Prime Agent 的 persistent Python RLM、工具和 child agent 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 2.</strong> Prime 的 model-facing surface 是 persistent Python／RLM；相較 Hermes 的 AIAgent core 和 OpenClaw 的 Gateway，files、shell、skills、MCP 和 child agent 都從更靠近模型的工作面接出去。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>。</figcaption>
</figure>

Prime 的 RLM 文件把 context 放進可以由 Python 操作的資料結構，這是三者裡最接近 programming language 的 model-facing surface；Hermes 把同一層工作收在 AIAgent core，OpenClaw 則交給 Gateway session loop。模型可以讀資料，把中間結果留在變數裡，再決定下一段程式、工具呼叫或 child-agent 工作。[^prime-rlm]

~~~text
request
  -> parent model
  -> persistent Python REPL / RLM
       -> files / shell / skills / MCP
       -> rlm.spawn(...)
       -> workspace / durable harness state
~~~

一般 tool-calling loop 把一次工具呼叫拆成幾個 host 端步驟：模型產生工具名稱和參數，host 執行工具，再把結果包回下一輪訊息；Hermes 把這個 host loop 集中在 tool registry 和 terminal backend，OpenClaw 再往上加上 Gateway routing 和 policy。

RLM 將這些步驟放進可持續操作的 programming surface，因此 Prime 的優勢是模型能直接組合控制流；Hermes 和 OpenClaw 把控制流收進 service 或 Gateway，操作面比較集中，Prime 的代價則是 state 和副作用更接近模型的工作面。

一個 research session 可以先讀一批文件，把解析結果留在 Python state，再把不同子問題交給 <code>rlm.spawn(...)</code>，最後把 child 結果合併成報告；Hermes 用 delegation 做相似的 task split，OpenClaw 則把背景工作交給 automation、node 或 plugin。

這個模型讓「下一步」具有程式結構，Prime 因此比 Hermes 和 OpenClaw 更能表達條件分支、迴圈、暫存資料和子任務。

條件分支、迴圈、暫存資料和子任務都可以留在工作面裡；Hermes 和 OpenClaw 把相同責任收進 service 或 Gateway，比較容易集中治理，Prime 則承擔較多 workflow correctness 和 workspace recovery。

長任務因此少了幾次 context 重建，但 runtime 要負責更多 state；Hermes 把這些責任分散到 session DB 和 memory lifecycle，OpenClaw 則放到 Gateway state 和 routing。

Prime 的變數可能指向過期檔案，工具可能已經寫入檔案而 parent 尚未收到結果，child agent 也可能只完成一半；Hermes 主要面對 memory retrieval 和 backend scope，OpenClaw 主要面對跨入口 state scope。

session resume 需要知道哪些 Python state、檔案副作用和 child result 已經成立，這是 Prime 相較 Hermes 的 service state 和 OpenClaw 的 Gateway state 更重的 recovery 責任。

### Prime 的 execution boundary

persistent Python 是執行介面，這讓 Prime 的 programming surface 比 Hermes 的 tool call 和 OpenClaw 的 Gateway policy 更直接。

sandbox 是否存在，取決於 worker、kernel、host bridge 和部署設定；Hermes 把這個問題交給 terminal backend，OpenClaw 則要再加上 plugin、node 和 Gateway 的 process boundary。

如果 Python process 能直接讀到使用者檔案、環境變數或 credential，模型寫出的程式也可能沿用同一組權限；Hermes 和 OpenClaw 也有同樣的風險，權限入口分別落在 terminal backend 和 Gateway-connected component。

因此 Prime 的架構貢獻集中在 programming model，Hermes 的貢獻集中在 personal service，OpenClaw 的貢獻集中在 system control plane。

三者的權限邊界都要沿著最後執行 action 的 process 追下去；Prime 看 host bridge，Hermes 看 terminal backend，OpenClaw 看 Gateway、plugin 和 node。

## Hermes：personal agent service 是控制中心

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/hermes.png" width="1300" height="680" alt="Hermes Agent 的 AIAgent loop、provider resolver、tool registry 和 memory 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 3.</strong> Hermes 把多個入口接到同一個 AIAgent loop；相較 Prime 的 workspace 和 OpenClaw 的 Gateway，provider、tool registry、terminal、web、MCP 和長期 state 都收在較完整的 personal service。來源：<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>。</figcaption>
</figure>

Hermes 的中心是長時間運作的 personal agent service，位置介於 Prime 的 persistent workspace 和 OpenClaw 的 Gateway control plane 之間。

CLI、Telegram、Discord、ACP 和其他 gateway 入口，把工作送進同一個 AIAgent core；Prime 的入口更接近 coding／research session，OpenClaw 的入口則會先經過更完整的 channel、node 和 policy routing。

core 內的 prompt builder、provider resolver 和 tool registry 負責準備一次 model call，這讓 Hermes 比 Prime 更容易集中維護，也比 OpenClaw 少一層跨入口控制面的責任。

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

這個位置讓 Hermes 可以把 provider switching、session search、memory、skills 和 cron 放在同一個 service 裡；Prime 需要把更多 workflow state 留在 workspace，OpenClaw 則要把相同能力和更廣的 channel／node scope 放進 Gateway。

模型不必知道 request 來自哪個 channel，channel 也不必各自複製一套 agent loop，這是 Hermes 相較 Prime 的日常使用優勢；OpenClaw 取得更大的整合範圍，但要同步管理更多 identity 和 session scope。

Hermes 的長期狀態由 session、搜尋索引、Markdown context 和 skills 組成；Prime 偏向 persistent Python namespace，OpenClaw 偏向 workspace、retrieval 和 Gateway state。

~~~text
write -> index -> retrieve -> inject into the next model call
~~~

寫入失敗時，使用者以為已經保存的內容根本不存在；索引沒有更新時，內容存在卻找不到；取回結果太寬時，過期或不相關的記憶會進入 prompt；注入過多時，歷史資料會吃掉當前任務的 context budget。Hermes 最直接面對這組 memory failure，Prime 會把同類問題表現成 workspace resume，OpenClaw 則可能表現成跨 channel 的 context scope。

這些 failure mode 是 Hermes 的主要弱點，Prime 會以 stale Python state 和 workspace resume 的形式出現，OpenClaw 則可能擴大成跨 channel 的 context scope 問題。

<code>MEMORY.md</code>、<code>USER.md</code> 和 skills 也有 scope 問題；Hermes 的 personal service 需要在便利的跨 session 設定和隔離之間取平衡，Prime 主要檢查 workspace 邊界，OpenClaw 則要檢查 user、channel、node 和 Gateway state 的組合。

Hermes 的 terminal execution 可以接 local process、container 或 remote backend，這讓它比 Prime 的 host-oriented Python bridge 更容易替換執行後端，也比 OpenClaw 的 plugin／node 組合簡單。

approval pattern 控制 action 能否進入執行路徑，真正的 process isolation 仍由 terminal backend 和部署環境決定；同一個 AIAgent core 接上本機 shell 和受限 container，信任邊界完全不同，OpenClaw 的 pairing 和 policy 也不能取代這個 process-level 檢查，Prime 的 host bridge 同樣需要單獨驗證。

## OpenClaw：Gateway 管理整個 control plane

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/openclaw.png" width="1300" height="680" alt="OpenClaw 2.0 的 Gateway WebSocket、session routing、policy、plugins、automation 和 workspace 手繪架構圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 4.</strong> OpenClaw 把 channels、CLI、nodes、plugins 和 automation 接到 Gateway；相較 Prime 的 session workspace 和 Hermes 的 personal service，OpenClaw 的 workspace、SQLite／FTS5 和 retrieval 由更高層的 Gateway path 保存與取回。來源：<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">Gateway docs</a> 和 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1">v2026.8.1 source</a>。</figcaption>
</figure>

OpenClaw 的 request 先處理入口和 session，再進 model loop；Prime 直接從 parent model 進 Python workspace，Hermes 從 gateway 進 AIAgent core，OpenClaw 把 session ownership 再往 Gateway 上移。

Gateway 要先知道 request 屬於哪個 session、哪個 user、哪個 node，以及這次 action 能使用哪些 policy，這是 OpenClaw 相較 Prime 和 Hermes 多出的整合責任，也是它能承接多入口系統的主要優勢。

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

這個 control plane 把 channel coordination、session ownership、policy enforcement 和 plugin／node 接入放在一起；Prime 的控制面更適合單一長任務，Hermes 的控制面更適合單一個人的 personal service。

cron 和 automation 可以在沒有即時聊天的情況下啟動工作，plugin 和 node 則把外部能力帶進 Gateway，OpenClaw 因而比 Prime 和 Hermes 更適合長期 background orchestration，也承擔更大的元件管理成本。

多入口系統的困難在 scope；Prime 主要處理 session workspace，Hermes 主要處理 personal memory，OpenClaw 還要處理 Telegram、Web UI 和 paired device 之間的共享或隔離。

共用 session 時，Gateway 要限制哪些 context 可以跨入口流動，哪些 action 只能由特定 node 或使用者核准，plugin 和 node 的結果也要回到正確的 channel；這是 OpenClaw 的系統整合優勢，同時也是它相較 Prime 和 Hermes 更大的 failure surface。

pairing 能確認來源和身份，approval 能控制某個 action 是否進入執行路徑；Hermes 也有 approval pattern，Prime 則更依賴 worker、kernel 和 host bridge 的執行路徑。

這兩件事都不能單獨描述 plugin 是否隔離，或 node 最後能碰到哪些檔案、網路和 credential，因此 OpenClaw 的 Gateway ownership 帶來更完整的 policy surface，也帶來比 Prime 和 Hermes 更高的 trust-boundary 複雜度。

在固定的 <code>v2026.8.1</code> release 裡，OpenClaw 的 plugin execution、sandbox 設定和 node 權限仍要分開檢查；Prime 的 worker／kernel 和 Hermes 的 terminal backend 也需要相同層級的 process 檢查。[^openclaw-security]

## Control surface：三個 loop 序列化不同的東西

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/control-surface.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 三種 per-session control loop 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 5.</strong> 三個 project 的 per-session loop：Prime 序列化 Python cell、host request 和 child session；Hermes 序列化 tool call、delegation 和 summary；OpenClaw 序列化 intake、context、native tools 和 persist。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

Prime 的一個 session 以 Python cell 為起點，Python cell 觸發 host request，host request 可能建立 child session；相較 Hermes 的 tool call 和 OpenClaw 的 intake，Prime 序列化的是最靠近 model-facing workspace 的 programming state。

Hermes 的一個 session 從 tool call 開始，<code>delegate_task</code> 把工作交給另一個 agent 或 background path，結果再以 summary 回到主要對話；這比 Prime 的 Python state 更容易由 service code 觀察，比 OpenClaw 的 Gateway-owned run 更聚焦在 personal assistant。

OpenClaw 的一個 session 從 intake 開始，Gateway 組合 context 和 model，再執行 native tools，最後把結果 persist；它把每個入口的工作收進同一個 Gateway-owned run，換取比 Prime 和 Hermes 更廣的 routing 和 policy 控制。

三者的「session」都可以長時間存在，session 內部實際被序列化的物件不同：Prime 序列化 programming state，Hermes 序列化 tool／product loop 和 personal memory，OpenClaw 序列化 Gateway 管理的 request、policy、plugin 和 persistence。

因此，Prime 的優勢是工作流可編程，Hermes 的優勢是 service loop 集中，OpenClaw 的優勢是跨入口 ownership；對應的弱點則是 Prime 的 recovery、Hermes 的 service state、OpenClaw 的 shared trust surface。

## State：持久化資料決定工作會留下什麼

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/state.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 persistent state 和 memory 手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 6.</strong> 三種 state path：Prime 以 persistent Python namespace 和 Continual Harness 為中心；Hermes 以 session DB、Markdown memory 和 skills 為中心；OpenClaw 以 workspace、retrieval 和 plugin／context engine 為中心。來源：<a href="https://arxiv.org/abs/2605.09998">Continual Harness</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1">OpenClaw source</a>。</figcaption>
</figure>

三個 runtime 都把有用資料保存到 model weights 之外，但保存位置形成三種優劣：Prime 保存 Python workspace、prompts、memories、skill descriptions、child specs 和工作產物，保留最多工作連續性；Hermes 保存 session history、SQLite／FTS5 index、<code>MEMORY.md</code>、<code>USER.md</code> 和 skills，偏向 personal service；OpenClaw 保存 workspace Markdown、SQLite／FTS5 和 retrieval state，偏向跨入口整合。

Continual Harness 將 supplemental prompt、memory、skill description 和可重用的 subagent specification 保存成 durable state，讓 Prime 的工作規則跨過一次 chat window；Hermes 把相似責任放在 session memory 和 skills，OpenClaw 則放在 workspace、retrieval 和 Gateway state。[^continual-harness]

這些資料會影響下一輪輸入，模型參數維持原狀；Prime 的主要代價是 workspace resume 的 stale state，Hermes 的主要代價是 memory retrieval correctness，OpenClaw 的主要代價是 cross-channel context scope。

Prime、Hermes 和 OpenClaw 的 state lifecycle 都可以寫成：

~~~text
write -> index or organize -> retrieve -> inject
~~~

Prime 的 retrieve 多半發生在同一個 persistent workspace，或下一次 session resume；Hermes 需要 session search、personal context 和 skill loading；OpenClaw 需要 session routing、workspace context 和 hybrid retrieval。

Prime 把 retrieve 做得最接近工作流，Hermes 把 retrieve 做得最接近日常記憶，OpenClaw 把 retrieve 做得最接近多入口 context assembly。

真正需要驗證的是 Prime、Hermes 和 OpenClaw 各自的 state scope。

Prime 要確認 workspace resume 時不會帶入 stale variable 或重做已完成的副作用；Hermes 要確認個人 memory、session index 和 channel scope 能正確對齊；OpenClaw 要確認跨入口共享的 context 沒有把不該流動的資料送到另一個 channel 或 node。

這三個檢查對應三種優勢的價格：Prime 的連續工作面需要更重的 recovery，Hermes 的便利記憶需要更嚴格的 retrieval scope，OpenClaw 的整合能力需要更完整的 identity 和 permission scope。

## Task-level parallelism 和 decoder-level dependency

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/parallelism.png" width="1300" height="680" alt="Agent task-level parallelism 與 autoregressive decoder dependency 的手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 7.</strong> parent 可以把獨立工作 fan-out 給 child A、B、C，再收集結果；單一回答的 token path 仍沿著 t1、t2、t3、t4、t5 依序生成。來源：Prime 的 <a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">RLM</a>、Hermes 的 <a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">delegation</a>、OpenClaw 的 <a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">automation path</a>。</figcaption>
</figure>

Agent project 裡的 <code>rlm.spawn(...)</code>、delegation、background task 和 automation 都會增加 task-level parallelism；Prime 把 fan-out 放進 Python workspace，Hermes 放進 personal service，OpenClaw 放進 Gateway orchestration。

Prime、Hermes 和 OpenClaw 都可以把獨立的 research、coding、maintenance 或 scheduled work 分派給不同 worker、child session 或 node；改善的是工作分派和 wall-clock time，優勢則落在各自的工作尺度。

Prime、Hermes 和 OpenClaw 的單一 autoregressive sequence 都保留 token dependency。

對 Prime、Hermes 和 OpenClaw 而言，如果有 $n$ 個彼此獨立的子任務，理想化的順序執行時間接近：

$$
T_{serial} = \sum_{i=1}^{n} T_i
$$

在資源足夠、子任務真的獨立，而且 merge 成本可接受的條件下，Prime、Hermes 和 OpenClaw 的平行執行時間才可能接近：

$$
T_{parallel} \approx \max_i(T_i) + T_{dispatch} + T_{merge}
$$

Prime 的 <code>rlm.spawn(...)</code> 把 fan-out 接到 persistent Python workspace，控制力最強；Hermes 的 delegation 把子任務接到 personal agent service，日常使用最集中；OpenClaw 的 automation、node 和 plugin 把背景工作接到 Gateway control plane，整合範圍最大。

三者都能縮短多任務的 wall-clock time，代價分別落在 Prime 的 child lifecycle、Hermes 的 background scope 和 OpenClaw 的 plugin／node trust。

Prime、Hermes 和 OpenClaw 都保留單一回答的 decoder dependency；它們的平行化優勢只出現在 task fan-out，不會改變同一個 response 的 token 順序。

## Execution、Admission、Isolation

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-runtime/security.png" width="1300" height="680" alt="Prime、Hermes、OpenClaw 的 execution、approval 和 isolation 邊界手繪比較圖" loading="lazy" decoding="async">
  <figcaption><strong>圖 8.</strong> execution、approval／admission 和 child／plugin boundary 分開檢查。三個 project 的 lifecycle、policy 和 process isolation 落在不同位置。來源：<a href="https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md">Prime RLM</a>、<a href="https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md">Hermes README</a>、<a href="https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway">OpenClaw Gateway</a>。</figcaption>
</figure>

安全分析要沿著 action 的實際路徑走，因為 Prime、Hermes 和 OpenClaw 把 execution、admission 和 isolation 放在不同 component。

Prime、Hermes 和 OpenClaw 的 model output 都只表示模型提出了一個 action；三者的優劣要看這個 action 之後由誰執行、誰允許，以及誰持有檔案和 credential。

檢查 Prime、Hermes 和 OpenClaw 時，三個檢查點是：

1. action 最後在哪個 process、container 或 remote worker 執行。
2. 哪個 component 允許它通過 approval、pairing 或 policy。
3. 執行 process 能看到哪些檔案、credential、網路和 session。

Prime 要追 Python worker／kernel、shell bridge、host process 和 user environment；Hermes 要追 terminal backend、approval path、mounted secret、provider config 和 deployment；OpenClaw 要追 Gateway、plugin、node、channel identity、pairing 和 policy。

worker 名稱不能直接代表 sandbox，container 名稱不能直接代表 credential 已經隔離，pairing 名稱也不能直接代表 plugin 擁有獨立 process；Prime、Hermes 和 OpenClaw 都需要沿著 process boundary 和 credential path 實際驗證。

Prime 的 programming freedom 伴隨 host permission 風險；Hermes 的 service convenience 伴隨 backend scope 風險；OpenClaw 的 integration breadth 伴隨更大的 shared trust surface。

## 三個 project 各自改變了哪一層

Prime 改變的是模型操作電腦的 programming surface，把長時間工作需要的 context、暫存資料、工具組合和 child task 放進 persistent Python／RLM；Hermes 改變的是 personal assistant 的 service boundary，把多入口、provider、memory、skills、terminal backend、delegation 和 cron 收進同一個服務；OpenClaw 改變的是多入口 agent system 的 control plane，把 session routing、channels、nodes、plugins、automation 和 policy 放在 Gateway path 內。

三者的優勢分別落在不同位置：Prime 的可編程性最高，Hermes 的日常 service 完整度最高，OpenClaw 的整合和 routing 範圍最高；對應的限制則是 Prime 的 state／host responsibility、Hermes 的 service scope、OpenClaw 的 trust 和 policy complexity。

評估 Prime、Hermes 或 OpenClaw 的部署時，固定標出三個 owner：model action 交給哪個 loop，state 由哪個 component 寫入、索引、取回和注入，以及 tool action 最後在哪個 process、credential scope 和 policy 下執行。

這三條線會直接指出 Prime、Hermes 和 OpenClaw 的 session resume、memory scope 和 credential isolation 測試位置。

[^prime-readme]: [Prime Agent README at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md). 用於 project scope、long-running work、coding 和 research 定位。
[^prime-rlm]: [Prime Agent RLM programming model at commit 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md). 用於 persistent Python surface、context variables、child-agent lifecycle 和 host bridge。
[^continual-harness]: [Continual Harness](https://arxiv.org/abs/2605.09998). 用於 durable prompt、memory、skill 和 subagent state 的 runtime framing；這些資料與 model-weight update 分開。
[^hermes-readme]: [Hermes Agent README at commit afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md). 用於 personal-agent scope、gateway、memory/search、skills、cron、delegation 和 terminal backend。
[^openclaw-release]: [OpenClaw v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1). 此比較把這個 pinned release 稱為 OpenClaw 2.0。
[^openclaw-security]: [OpenClaw gateway documentation at v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway). 用於 Gateway routing、policy、pairing、approval 和 deployment-bound security observations。
