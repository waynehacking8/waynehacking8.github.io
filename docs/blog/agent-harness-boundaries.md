---
description: "從 Prime 的 persistent Python、Hermes 的 personal agent service，到 OpenClaw 的 Gateway control plane，拆解三個 agent runtime 如何處理控制流、狀態與權限。"
date: "2026-09-15"
updated: "2026-09-15"
language: "zh-Hant"
image: "/assets/blog/agent-harness-boundaries.svg"
tags:
  - Architecture
  - Agents
  - Security
---

# Prime、Hermes、OpenClaw：三個 Agent Runtime 的架構選擇

*2026-09-15 · Agent Systems / Runtime / Security*

<figure id="agent-runtime-architecture" class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-harness-boundaries.svg" width="1200" height="630" alt="Prime、Hermes 與 OpenClaw 的 agent runtime 架構比較" loading="eager" decoding="async">
  <figcaption><strong>圖 1.</strong> 自製架構圖。三個 project 都把模型接到工具和持久化 state，但 control、state 和 trust 落在不同的層。</figcaption>
</figure>

一個聊天模型只需要完成一次推理。

它收到 prompt，生成 token，再把文字交回呼叫端。

Agent 要處理的是另一種工作：模型完成一輪推理之後，系統還要把工具結果、session state、權限和下一輪 context 接回來。

模型外部的程式碼、設定和執行環境承擔這些工作。

這一層通常叫做 agent harness。`harness` 描述模型如何接到外部世界；`runtime` 描述實際運作的服務，以及它如何管理 request、process、state 和 policy。

Prime Agent、Hermes Agent 和 OpenClaw 2.0 都在做這一層，但三個 project 面對的工作並不相同。

Prime 從長時間的 coding 和 research session 出發。

Hermes 從每天使用的 self-hosted personal assistant 出發。

OpenClaw 則從多個 chat channel、device、plugin 和 automation 的整合出發。

因此，三個 runtime 對同一個問題給了三個答案：模型產生下一個 action 之後，哪一層接手？工作狀態放在哪裡？工具和外部元件最後承擔誰的權限？

比較版本固定為 Prime Agent commit `1fc1adb6`、Hermes Agent commit `afe06f2`，以及 OpenClaw `v2026.8.1` release。「OpenClaw 2.0」僅指這個 release，`current main` 不在比較範圍內。

## Agent runtime 的比較單位

聊天模型和 agent runtime 的責任不同。

| 層 | 一次 completion | Agent runtime |
| --- | --- | --- |
| 控制流 | 模型產生輸出後結束 | runtime 可能把工具結果送回模型，繼續下一輪 |
| 工具 | 呼叫端自行處理工具 | runtime 宣告工具、執行 action，並把結果放回 session |
| 狀態 | 通常由呼叫端保存對話 | runtime 保存工作資料、session、memory 或 workspace |
| 權限 | 由外部應用程式決定 | runtime 需要處理 approval、policy、process 和 isolation |
| 失敗處理 | 呼叫端自行重試 | runtime 可以恢復 session、重跑工具或等待 child task |

模型仍然負責產生 token 和選擇下一個 action。

模型負責產生 token 和選擇 action；檔案、process 和跨 session state 由模型外部的執行層管理。

比較軸是三個 project 如何把模型接成可以持續工作的系統。

## 三個 project 的工作單位

三個 project 都有 model、tools、memory 和 agent loop，功能名稱相同，工作單位卻不同。

從 README 的開場來看，它們的工作單位其實不同。

| Project | 作者／來源 | 主要工作 | 工作單位 | 架構中心 |
| --- | --- | --- | --- | --- |
| Prime Agent | Prime Intellect | coding、research、long-running work | 可以持續操作的 session | persistent Python／RLM workspace |
| Hermes Agent | Nous Research | self-hosted personal assistant | 一個人每天使用的 agent service | AIAgent core、gateway、memory、skills |
| OpenClaw 2.0 | OpenClaw Foundation 和社群 | 多 channel、device、plugin、automation | 多入口的 self-hosted system | Gateway control plane |

Prime Agent 的 README 把 coding、research 和 long-running work 放在一起談。[^prime-readme]

它以可以留在工作環境裡繼續做事的 session 為核心抽象。

Hermes Agent 把自己定位成可以自行部署的 personal agent，從 CLI、Telegram、Discord 等入口接收工作，再由同一個服務處理 provider、memory、skills、session search、delegation 和 cron。[^hermes-readme]

OpenClaw 把 assistant 放在使用者的 devices 和 chat channels 旁邊，Gateway 再把 channels、nodes、plugins、automation 和 agent session 接起來。[^openclaw-release]

三個起點直接決定架構。

Prime 先解決「一段工作怎麼留在同一個可操作的環境裡」。

Hermes 先解決「一個人怎麼每天從不同入口使用同一個 assistant」。

OpenClaw 先解決「多個入口和外部元件怎麼由一個 self-hosted control plane 統一管理」。

比較從工作負載和架構 owner 開始。

## 三條 request path

圖 1 以三欄表示三個 runtime。

每一欄都沿著同一條路徑閱讀：request 從哪裡進來，哪一層呼叫模型，模型產生的 action 由誰執行，結果和長期 state 最後放在哪裡。

| Request path | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| 入口 | coding／research request | CLI、messaging gateway、ACP | channel、CLI、node |
| Model-facing control surface | persistent Python／RLM REPL | AIAgent core | Gateway session loop |
| 工具路徑 | files、shell、skills、MCP、child agent | terminal、web、MCP | native tools、plugins、nodes |
| 長期 state | Python workspace、harness state、session artifacts | SQLite／FTS5、`MEMORY.md`、`USER.md`、skills | workspace Markdown、SQLite／FTS5、retrieval |
| request 完成後 | 留在可重新接上的工作環境 | 回到可搜尋的 personal service | 回到 channel、node 或 automation path |

左邊的 Prime 把工作面放在模型旁邊。

模型進入 persistent Python／RLM REPL，再從這個 programming surface 連到 files、shell、skills、MCP 和 child agent。

中間的 Hermes 把模型放進 AIAgent core。

入口、provider、tool registry 和 service state 都由 personal agent service 統一管理。

右邊的 OpenClaw 把 Gateway 放在更高的位置。

channels、nodes 和 plugins 先進入 Gateway，Gateway 再負責 session routing、policy、agent loop 和 persistence。

圖 1 把三個 ownership 差異放在同一張圖裡：Prime 的控制面接近 model-facing workspace，Hermes 的控制面接近 personal agent service，OpenClaw 的控制面接近 Gateway control plane。

## Agent runtime 的四個責任

Agent runtime 的抽象 loop：

~~~text
request
  -> load session and context
  -> call the model with the available tools
  -> inspect the model's next action
  -> execute a tool or return a final answer
  -> persist the result
  -> continue, pause, or stop
~~~

三個 project 都要處理同一組責任，但放置位置不同。

| Runtime 責任 | 要回答的問題 | Prime | Hermes | OpenClaw |
| --- | --- | --- | --- | --- |
| Control flow | tool result 回來後是否繼續？誰結束 session？ | REPL／RLM 和 parent session | AIAgent core | Gateway session loop |
| Tool execution | action 在哪裡變成 process 或 request？ | worker／kernel、host bridge | terminal backend、web、MCP | native tools、plugin、node |
| State lifecycle | 哪些內容寫回、索引、取回、注入？ | workspace、durable harness state | database、Markdown memory、skills | workspace、SQLite／retrieval |
| Admission and trust | 誰能讓 action 進入執行路徑？ | worker、kernel、host process | approval、backend、deployment | pairing、policy、Gateway、plugin |

四個責任構成三個 runtime 的比較軸。

同一個 tool，如果由 persistent REPL、AIAgent service 或 Gateway policy 管理，能看到的 state、能使用的 credential 和失敗後的 recovery 都可能不同。

## Prime：persistent Python workspace

Prime 的主要設計選擇，是讓模型在一個持續存在的 Python control environment 裡組織工作。

Prime Agent 的 RLM 文件直接把 context 放進可由 Python 操作的資料結構。模型可以讀取資料，將中間結果保留在變數裡，再呼叫工具或建立 child agent。[^prime-rlm]

Prime 的 request path 分成四層：

| Prime component | 在 request path 裡做什麼 | 產生的 state | 主要代價 |
| --- | --- | --- | --- |
| Parent model | 讀取目前 context，選擇下一個程式或 action | model output、task plan | 仍受 autoregressive generation 限制 |
| Persistent Python／RLM | 提供可持續操作的 programming surface | 變數、中間結果、控制流 | stateful failure 比一次性 tool call 複雜 |
| Files、shell、skills、MCP | 把程式碼或 action 接到外部世界 | 檔案、命令結果、外部回應 | 實際權限取決於 host bridge |
| `rlm.spawn(...)` 和 child agent | 把獨立子問題拆出去執行 | child handle、結果、工作產物 | 需要處理 lifecycle、merge 和副作用 |

Prime request path：

~~~text
request
  -> parent model
  -> persistent Python REPL / RLM
       -> files / shell / skills / MCP
       -> rlm.spawn(...)
       -> workspace and durable harness state
~~~

Prime 的差異在於 Python 變成模型可以持續操作的 control surface。

在一般 tool-calling loop 裡，模型產生一個工具名稱和參數，host 執行工具，再把結果包成下一輪訊息。

在 RLM programming model 裡，模型可以把資料處理、工具呼叫和 child-agent 管理寫進同一個工作流程。

一個研究任務可能先讀取一批文件，把解析結果放在 workspace，再把不同子問題交給 `rlm.spawn(...)`，最後把 child 的結果合併成一份報告。

這種寫法把「下一步要叫哪個工具」改成「下一段程式要怎麼繼續跑」。

| 工作需求 | 一般 tool-calling loop | Prime 的 RLM workspace |
| --- | --- | --- |
| 讀取大量資料 | 每輪把需要的結果回填 context | 可把資料處理結果留在 Python state |
| 反覆修改程式 | 多次呼叫工具，再由 host 組合訊息 | 在同一個 programming surface 裡繼續執行 |
| 拆分子任務 | 由外部 orchestration code 管理 | 可由 `rlm.spawn(...)` 接到 parent workflow |
| session resume | 依賴外部保存與重新注入 | workspace 和 durable state 成為工作的一部分 |
| 失敗恢復 | 通常從最近一次 request 重試 | 要處理變數、檔案副作用和 child lifecycle |

對長任務而言，這會減少每一輪重新建立 context 的成本。

模型可以留下文字、變數、檔案、子任務結果、生成的程式碼和驗證記錄。

### Prime 的 execution boundary

persistent Python 讓模型有更強的 programming surface，也把 runtime state 帶進了模型的控制流程。

Python REPL 是執行介面；sandbox 需要另外配置。

| Prime 的層 | 它可能隔離什麼 | 仍需另外確認的邊界 |
| --- | --- | --- |
| worker／kernel lifecycle | process 的啟動、停止、重新連接 | 最小 host permission |
| Python／shell bridge | 模型如何觸發外部 action | credential 不會被看到 |
| workspace | 工作資料與中間結果 | workspace 不會包含敏感檔案 |
| child agent | 子問題的工作流程 | child 沒有額外副作用 |

如果 Python process 能直接看到使用者的檔案、環境變數或 credential，模型寫出的程式就可能沿用同一組權限。

Prime 的主要進步在 programming model；isolation model 仍取決於 worker、host bridge 和部署設定。

### Prime 的 state management

persistent workspace 會保留 useful state，也會保留 stale state。

變數可能指向過期檔案，child agent 可能只完成一半，某次工具呼叫可能已經寫入檔案但沒有把結果正確回報給 parent。

Prime runtime 需要處理 workspace checkpoint、child 回報、session resume，以及 failure 後的副作用重做。

這些問題在一次性問答裡不明顯，在數十分鐘或數小時的 coding／research session 裡會變成主要的可靠性成本。

## Hermes：personal agent service

Hermes 的出發點是把模型放進一個長期運作的 personal agent service。

Hermes 先建立長期運作的 personal agent service，再把入口、provider、memory、skills、terminal backend 和 cron 接到服務上。

Hermes 的核心 component：

| Hermes component | 在 request path 裡做什麼 | 對使用者的意義 | 主要代價 |
| --- | --- | --- | --- |
| CLI、Telegram、Discord、ACP | 接收不同入口的工作 | 入口可以換，assistant service 不必換 | identity 和 session 要一致 |
| Gateway | 把入口送進同一個服務 | channel 不必各自實作 agent loop | gateway 本身成為長期運作的 process |
| AIAgent core | 組 prompt、解析 provider、管理 tool registry | model loop 集中在一個地方 | loop state 和 provider failure 集中到 core |
| terminal、web、MCP | 執行模型選出的 action | assistant 可以真的操作外部系統 | backend 決定實際權限 |
| memory、skills、cron | 保存 context、提供能力、定時啟動 | service 可以每天持續使用 | state scope 和 background job 要管理 |

Hermes request path：

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

入口和模型回合因此脫鉤。CLI 和 messaging channel 的 request 都由同一個 AIAgent core 組 context、選 provider、載入工具並執行 loop。

這和 Prime 的 model-facing REPL 是不同的取捨。

Prime 讓模型在工作面裡組合控制流。

Hermes 把控制流收在 service 裡，模型在 service 提供的 tool registry 內選擇下一個 action。

這種設計比較適合「每天都要用」的 assistant。

服務可以集中處理 provider switching、session search、memory、skills 和 cron。

模型不必知道訊息是從哪個 channel 進來，也不必為每次互動重新建立整套個人設定。

### Hermes 的 service state path

Hermes 的 memory 由 session、搜尋索引、Markdown context、skills 和 scheduled task 組成。

| State component | 保存什麼 | 何時會被用到 | 如果設計錯誤會怎樣 |
| --- | --- | --- | --- |
| session history | 過去的互動和工具結果 | 恢復或搜尋舊 session | 對話接不起來 |
| SQLite／FTS5 | 可搜尋的索引 | 找回相關訊息 | 資料存在但查不到 |
| `MEMORY.md`、`USER.md` | 個人偏好和長期 context | 新 session 的 prompt 建構 | 偏好遺失或跨使用者混用 |
| skills | 可重複使用的操作規則 | tool 或 workflow 載入 | 每次都重新描述同一個流程 |
| cron | 沒有即時訊息時啟動的工作 | scheduled task | background action 缺少清楚的 scope |

Hermes memory path：

~~~text
write -> index -> retrieve -> inject into the next model call
~~~

每一步都有自己的 failure mode。

寫入失敗，模型以為已經記住的內容其實不存在。

索引沒有更新，內容存在但搜尋不到。

取回結果不對，下一輪 context 會混入錯誤或過期資料。

注入過多，模型的 context budget 會被歷史內容吃掉。

「有 memory」只表示系統保存了某種資料；可靠性取決於誰寫入、誰能讀取、搜尋怎麼建立，以及不同 session 是否共享同一份資料。

### Hermes 的 terminal boundary

Hermes 可以把 terminal execution 放在 local process、container 或 remote backend。

approval pattern 可以要求某些指令先取得允許。它管理 action 是否進入執行路徑；process isolation 仍取決於 terminal backend。

| 問題 | Hermes 要看哪一層 |
| --- | --- |
| shell action 在哪裡執行？ | terminal backend |
| 哪些 action 需要確認？ | approval pattern 和 agent policy |
| process 能看到什麼？ | local、container 或 remote deployment 的設定 |
| credential 從哪裡來？ | backend、環境變數、mounted secret 和 provider 設定 |

同一個 AIAgent core，如果接的是使用者本機 shell，和接的是受限 container，風險模型並不相同。

## OpenClaw：Gateway control plane

OpenClaw 面對的是系統整合問題。

request 可能來自不同 channel、CLI 或 paired node。

Gateway 必須先找到對應的 session，再處理 routing、context、policy、model、native tool 和結果保存。[^openclaw-security]

OpenClaw request path components：

| OpenClaw component | 在 request path 裡做什麼 | Gateway 的作用 | 主要代價 |
| --- | --- | --- | --- |
| channel、CLI、node | 帶入訊息或外部事件 | 所有入口可以使用同一套 session 邏輯 | identity 和 pairing 變複雜 |
| Gateway WebSocket | 接收、轉送、維持連線 | 讓入口和 agent runtime 解耦 | Gateway 成為高價值 process |
| session routing | 找到正確的 agent 和 session | 不同入口可以共享或隔離 context | scope 錯誤會跨入口傳播 |
| policy、approval、pairing | 決定誰能觸發哪些 action | 管理 system-level admission | isolation 仍取決於 execution boundary |
| plugins、nodes、automation | 接入外部能力和背景工作 | agent 可以跨 channel 和 device 做事 | 外部元件擴大 trust boundary |
| workspace、SQLite、retrieval | 保存和取回 context | assistant 可以長期運作 | state scope 要和 session scope 對齊 |

OpenClaw request path：

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

Gateway 同時負責 session ownership、routing、channel coordination、policy enforcement 和 plugin／node 的接入。

cron 和 automation 可以在沒有即時聊天的情況下啟動工作。

plugin 和 node 則把外部能力帶進這個 control plane。

Gateway 還要處理多入口 identity、session sharing、node action 和 plugin result routing：不同入口是否屬於同一個使用者、哪些入口可以共用 session、哪一個 node 能執行 native action，以及 plugin 的結果應該回到哪個 channel。

### OpenClaw 的 routing 與 policy

多入口系統最難維持的是 identity、session 和 permission 的一致性。

如果 Telegram、Web UI 和 paired device 各自維護一套 session，使用者會得到三個互相不認識的 assistant。

共用 session 時，Gateway 必須知道哪些 context 可以跨入口流動，哪些 action 只能由特定 node 或使用者核准。

OpenClaw 把這些決定集中在 Gateway，換來一個比較清楚的 control plane。

代價是 Gateway 變成高價值的信任邊界。

| 信任對象 | Gateway 要控制什麼 | 仍需另外驗證的邊界 |
| --- | --- | --- |
| channel | 來源、identity、session mapping | 來源已驗證就代表 action 安全 |
| node | pairing、可用能力、回傳路徑 | node 只會執行低風險工作 |
| plugin | 載入方式、執行位置、可見資料 | plugin 一定與 Gateway 隔離 |
| automation | 觸發條件、context、重試 | background job 沒有使用者就不需要 policy |
| workspace／retrieval | 可讀範圍、寫入範圍、session scope | 所有 assistant context 都可以共享 |

pairing 和 approval 控制進入路徑；native plugin 是否隔離，仍要看它的 process boundary。

在固定的 `v2026.8.1` release 裡，plugin 的執行位置、sandbox 設定和 node 權限需要分開檢查。[^openclaw-security]

## Control、State 與 Trust

Control、State、Trust 三軸如下。

| 軸 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| Control owner | persistent Python／RLM 和 parent session | AIAgent core、provider resolver、tool registry | Gateway session loop、routing、policy |
| State owner | Python workspace、Continual Harness、工作產物 | session database、FTS5、Markdown memory、skills | workspace、SQLite／FTS5、hybrid retrieval |
| Trust owner | worker／kernel、host bridge、使用者 process | terminal backend、approval、部署環境 | Gateway、plugin、node、pairing、policy |
| 主要工作尺度 | 一段可持續的 coding／research session | 一個人每天使用的 service | 多 channel、多 node 的 system |
| 主要 failure mode | stale state、child lifecycle、host permission | backend scope、memory retrieval、background job | session scope、plugin trust、cross-channel policy |

### Control

Prime 把 model-facing control surface 放在 persistent Python／RLM。

模型可以在這個工作面裡保存資料、呼叫工具、管理 child agent，控制流更接近模型本身。

Hermes 把 loop 收在 AIAgent core。

模型提出 action，service 再透過 provider resolver、tool registry 和 terminal backend 把 action 變成執行。

OpenClaw 把更高層的控制權放在 Gateway。

Gateway 先處理 session、routing 和 policy，模型回合只是 Gateway 管理的一段執行流程。

### State

Prime 的 state 以 workspace 和 durable harness state 為中心。

Hermes 的 state 以 service database、search index、Markdown memory、skills 和 scheduled task 為中心。

OpenClaw 的 state 以 Gateway 管理的 workspace、session store 和 retrieval layer 為中心。

三者都會把資料保存到模型外面，但 state 的 scope 不同。

| State scope | Prime | Hermes | OpenClaw |
| --- | --- | --- | --- |
| 當前工作 | 變數、檔案、工具結果 | session history 和當前 prompt | current session context |
| 跨 session | durable prompt、memory、skills、subagent spec | `MEMORY.md`、`USER.md`、搜尋 index | workspace Markdown、retrieval state |
| 背景工作 | child agent | delegation、cron | automation、node、plugin |
| 最需要驗證的事 | 能否 resume 並避免 stale state | 能否正確 retrieve 和 inject | context 是否跨入口誤共享 |

Prime 的 state scope 對應 session resume。

Hermes 的 state scope 對應個人長期使用。

OpenClaw 的 state scope 對應跨入口 context sharing。

### Trust

Prime 要檢查 Python worker、kernel、host bridge 和使用者 process 的關係。

Hermes 要檢查 terminal backend、approval path 和部署環境。

OpenClaw 要檢查 Gateway、plugin、node、pairing 和 policy 的關係。

| 層 | 判斷問題 | Prime | Hermes | OpenClaw |
| --- | --- | --- | --- | --- |
| Execution | action 實際在哪個 process、container 或 remote worker 執行？ | Python worker／kernel、shell bridge | terminal backend | native tool、plugin、node |
| Admission | 哪個 component 可以讓 action 通過 approval 或 policy？ | parent／worker path | approval pattern、service policy | Gateway policy、pairing、approval |
| Isolation | process 能看到哪些檔案、credential、網路和 session？ | host process 與 worker 的設定 | local、container、remote deployment | Gateway、plugin、node 的部署設定 |

approval 開關只能證明 action 進入了某種 admission path；isolation 要看實際 process。

worker 或 container 的名稱不足以描述 host bridge，必須檢查部署設定。

要判斷一個 agent 的實際風險，必須把 action 從 model output 一路追到最後的 process 和 credential。

## Task-level 與 decoder-level parallelism

Agent project 常常同時出現 child agent、delegation、background task 和 automation。

這些功能增加 task-level parallelism；同一個 autoregressive sequence 仍使用 token-level 的序列生成。

| Parallelism 層級 | 它平行化什麼 | Prime | Hermes | OpenClaw | decoder 狀態 |
| --- | --- | --- | --- | --- | --- |
| Task-level | 獨立的 research、coding、maintenance 或 background task | `rlm.spawn(...)` | delegation、background work | automation、node、plugin path | 單一回答的 token dependency |
| Service-level | 不同入口或 session 的工作 | session／child runtime | 多入口 personal service | 多 channel、多 node Gateway | 單一模型回合的 decoder |
| Decoder-level | 同一個 sequence 的 token generation | autoregressive | autoregressive | autoregressive | 第 $t+1$ 個 token 仍依賴第 $t$ 個 token |

對一個必須依序產生 token 的回答來說，第 $t+1$ 個 token 仍然依賴第 $t$ 個 token。

把工作拆成多個 child，改變的是工作分派；decoder dependency 仍然存在。

如果有 $n$ 個彼此獨立的子任務，理想化的順序執行時間接近：

$$
T_{serial} = \sum_{i=1}^{n} T_i
$$

在資源足夠、子任務真的獨立，而且 merge 成本可接受時，平行執行才可能接近：

$$
T_{parallel} \approx \max_i(T_i) + T_{dispatch} + T_{merge}
$$

`child agent` 主要改善工作分派和 wall-clock time。

Prime 的 `rlm.spawn(...)`、Hermes 的 delegation，以及 OpenClaw 的 automation 都屬於工作層的拆分。

它們改善的是工作分派、背景執行或整體 wall-clock time。

單一回答沿著原本的 decoder path 生成 token，autoregressive dependency 維持。

## State lifecycle

Agent memory 來自 runtime 對外部資料的寫入和取回。

這和更新 model weights 是兩件事。

| State lifecycle | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| Write | workspace、prompt、memory、skill、child result、工作產物 | session、`MEMORY.md`、`USER.md`、skills | workspace Markdown、session、automation result |
| Index | workspace／harness 可重用 state | SQLite／FTS5 | SQLite／FTS5、retrieval layer |
| Retrieve | 同一個工作環境、session resume、child result | session search、personal context、skill loading | session routing、workspace context、hybrid retrieval |
| Inject | Python state、下一段 RLM control flow | prompt builder、AIAgent core | Gateway context assembly、model call |
| 最大風險 | stale variable、partial side effect | 找錯記憶、context 過長 | session scope 錯誤、跨入口資料外洩 |

Prime 保存 Python workspace、Continual Harness 的 prompts、memories、skill descriptions、child specs 和工作產物。

Continual Harness 將 supplemental prompt、memory、skill description 和可重用的 subagent specification 保存成 durable state，讓工作規則可以跨過一次 chat window。這些資料屬於 runtime state；model weights 維持不變。[^continual-harness]

Hermes 保存 session history、SQLite／FTS5 index、`MEMORY.md`、`USER.md` 和 skills。

OpenClaw 保存 workspace Markdown、SQLite／FTS5 和 retrieval state。

這些資料都可能影響下一輪輸入，模型參數則維持原狀。

比較 memory feature 時，要追的是寫入權、搜尋範圍、session scope、context budget 和資料刪除方式。

## Execution、Admission 與 Isolation

三個 project 都讓模型接觸外部工具，安全分析要沿著 tool execution path 展開。

同一個 `shell` 名稱，在本機 process、container、遠端 worker 和受 policy 控制的 node 上，代表的風險完全不同。

| Security 問題 | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| Action 在哪裡執行？ | Python worker／kernel、host bridge、shell | local、container 或 remote terminal backend | native tool、plugin、node、Gateway path |
| 執行允許 | parent／worker 的執行路徑 | approval pattern、service policy | Gateway policy、pairing、approval |
| 哪裡保存權限？ | user process、worker、host environment | backend、deployment、mounted secret、provider config | Gateway、plugin、node、channel identity |
| isolation 的主要未知數 | REPL 是否可碰 host | backend 是否真的受限 | plugin／node 是否獨立隔離 |
| 單一訊號不足以證明的事情 | worker 自動形成 sandbox | approval 自動形成 isolation | pairing 自動形成 plugin sandbox |

部署時，我會沿著四條線追一次：process、credential、plugin 和 host permission。

先確認 action 的實際 process。

再確認 process 如何取得 credential。

接著確認 plugin 或 node 能呼叫哪些外部能力。

最後才判斷 policy 和 isolation 是否真的形成邊界。

這四步能定位 security boundary。

## 系統層貢獻

這三個 project 的改變發生在模型和外部世界之間，涵蓋 programming model、personal service 和 system control plane。

| Project | 系統層抽象 | 對使用者的直接影響 | 仍需另外處理的問題 |
| --- | --- | --- | --- |
| Prime | programming model | 模型可以在 persistent Python workspace 裡組織 context、工具和 child task | host permission、state recovery、decoder dependency |
| Hermes | personal agent service | 一個人可以從多入口長期使用同一個 assistant | terminal isolation、memory correctness、background scope |
| OpenClaw | system control plane | 多 channel、node、plugin 和 automation 可以由 Gateway 統一管理 | plugin isolation、session scope、cross-channel policy |

Prime 重新安排模型操作電腦的介面。

Hermes 重新安排一個人每天使用 assistant 的服務邊界。

OpenClaw 重新安排多入口 agent system 的控制與整合邊界。

三者各自改變不同的系統抽象。

decoder benchmark 和單輪回答品質無法涵蓋這些 runtime 差異。

## Workload fit

| 工作負載 | 較接近的設計 | 適配原因 | 需要承擔的代價 |
| --- | --- | --- | --- |
| 長時間讀資料、寫程式、跑驗證 | Prime | persistent Python／RLM 可以把工作環境和中間結果留在同一個 session | workspace state、child lifecycle 和 host permission 必須自己管好 |
| 從 CLI 或聊天入口每天使用 personal assistant | Hermes | service 統一管理 provider、memory、skills、search 和 cron | terminal backend 和 deployment 決定實際隔離程度 |
| 同時接多個 channel、device、plugin 和 automation | OpenClaw | Gateway 統一管理 session routing、policy 和外部元件 | Gateway 變成高價值 trust boundary，scope 錯誤會跨入口傳播 |

工作負載和 runtime 的架構中心需要一起評估。

評估時確認三件事：模型位於哪個 loop，session state 寫到哪裡，以及 tool action 最後在哪個 process 和權限下執行。

這三個位置比工具數量更能描述系統架構。

[^prime-readme]: [Prime Agent README at commit `1fc1adb6`](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md). Used for the project scope, long-running work, and coding／research positioning.
[^prime-rlm]: [Prime Agent RLM programming model at commit `1fc1adb6`](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md). Used for the persistent Python surface, context variables, child-agent lifecycle, and host bridge.
[^continual-harness]: [Continual Harness](https://arxiv.org/abs/2605.09998). Used for the durable prompt, memory, skill, and subagent-state framing; it separates that runtime state from model-weight updates.
[^hermes-readme]: [Hermes Agent README at commit `afe06f2`](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md). Used for the personal-agent scope, gateways, memory/search, skills, cron, delegation, and terminal backends.
[^openclaw-release]: [OpenClaw `v2026.8.1`](https://github.com/openclaw/openclaw/tree/v2026.8.1). This is the pinned release called OpenClaw 2.0 in this comparison.
[^openclaw-security]: [OpenClaw gateway documentation at `v2026.8.1`](https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway). Used for Gateway routing, policy, pairing, approval, and deployment-bound security observations.
