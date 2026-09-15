---
description: "把 Prime Agent、Hermes Agent 和 OpenClaw 2.0 放到同一個問題上比較：模型外面的 harness 由誰持有控制流、狀態與信任邊界。"
date: "2026-09-15"
language: "zh-Hant"
image: "/assets/blog/agent-harness-boundaries.svg"
tags:
  - Architecture
  - Agents
  - Security
---

# Agent Harness 的三種邊界：Prime、Hermes 與 OpenClaw 2.0

*2026-09-15 · Agent Architecture / Runtime / Security*

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/agent-harness-boundaries.svg" width="1200" height="630" alt="Prime、Hermes 與 OpenClaw 的 agent harness 架構比較" loading="eager" decoding="async">
  <figcaption>自製示意圖：比較三個 project 把 control、state 和 trust 放在哪裡。</figcaption>
</figure>

很多人把 agent 簡化成「模型會自己用工具」。

這個說法沒有錯，但它把最重要的部分藏掉了。

模型產生一次 tool call 之後，誰把結果送回下一輪，誰保存中間狀態，誰決定這個工具能不能碰到 host，這些都不在模型權重裡。

這一層通常被叫做 agent harness。

本文把 Prime Agent、Hermes Agent 和 OpenClaw 2.0 放在同一個問題上比較：模型外面的系統，到底由哪一層接住工作。

版本固定在 Prime Agent 的 commit <code>1fc1adb6</code>、Hermes Agent 的 commit <code>afe06f2</code>，以及 OpenClaw 的 <code>v2026.8.1</code> release。

本文中的 OpenClaw 2.0 指這個 release，不把會繼續變動的 current main 混進來。

## TL;DR

- **Prime Agent** 把 model-facing control surface 放在 persistent Python／RLM workspace。它的工作單位是一個可以持續執行、保存變數、呼叫 child agent 的 coding／research session。
- **Hermes Agent** 把 loop 收在 AIAgent core 和 personal gateway。它的工作單位是一個每天可以從 CLI 或 messaging channel 使用的 personal assistant service。
- **OpenClaw 2.0** 把 routing、session、policy、plugin 和 automation 收到 Gateway。它的工作單位是一個連接多個 channel、device 和外部元件的 self-hosted system。
- 三者都有 task-level parallelism，但都沒有改寫 autoregressive decoder。child、background job 和 automation 拆的是工作，不是同一個回答的 token dependency。
- 安全性要沿著 execution、approval／admission 和 isolation 分開看。工具名稱本身不能告訴你 host 到底被誰控制。

## Harness 到底補了哪一層

單獨呼叫一個模型時，流程很短。

輸入 prompt，模型產生輸出，這一輪就結束。

Agent 需要把這個回合接成一個可以繼續工作的 process。

它要處理 request 從哪裡進來、該用哪個 model、目前有哪些工具、工具結果要回到哪個 session、哪些資料要保存，以及下一輪是否允許繼續執行。

因此我在這篇文章裡把 harness 拆成三個 owner 問題。

| 問題 | 看的是什麼 |
| --- | --- |
| Control | 模型可以直接操作哪一種工作面，下一步由哪個 loop 決定。 |
| State | 一個 session 結束後，哪些資料還能被下一輪找回來。 |
| Trust | tool、child agent 和 plugin 最後承擔的是哪個 process 的權限。 |

這不是一張 feature checklist。

同一個功能放在不同 owner 手上，行為就會不一樣。

## 三個 project 原本要服務的工作

Prime Agent 由 Prime Intellect 開發。

它的 README 從 coding、research 和 long-running work 開始寫，核心概念是 Recursive Language Model，也就是把 context 當成變數，把工具和 recursive subagent 當成 Python 裡的 function call。[^prime-readme]

Hermes Agent 由 Nous Research 開發。

它把自己定位成 self-hosted personal agent，入口包含 CLI 和多種 messaging platform，並把 provider、memory、skills、session search 與 cron 放進同一個長期運作的服務。[^hermes-readme]

OpenClaw 來自 OpenClaw Foundation 和社群。

它把 assistant 放在 devices 與 chat channels 旁邊，Gateway 再把 channels、nodes、plugins、automation 和 agent session 接起來。[^openclaw-release]

所以三者不是同一種產品的三個版本。

Prime 先把長任務的工作環境做出來。

Hermes 先把每天使用的 personal assistant service 做出來。

OpenClaw 先把多入口、多元件的 self-hosted control plane 做出來。

## 三個控制面的差別

先把三條 request path 壓成一張表。

| | Prime Agent | Hermes Agent | OpenClaw 2.0 |
| --- | --- | --- | --- |
| Model-facing control | persistent Python／RLM REPL | AIAgent core 的 agent loop | Gateway 管理的 session run |
| Durable state | Python workspace、harness state、session artifacts | SQLite／FTS5、MEMORY.md、USER.md、skills | workspace Markdown、SQLite／FTS5、hybrid retrieval |
| 主要工作單位 | 長時間 coding／research session | 個人長期使用的 assistant service | 多 channel、多 node 的 self-hosted system |
| 主要信任邊界 | worker／kernel 與 host process | terminal backend、approval 和部署設定 | Gateway、plugin、node 與 policy |

這個差異可以用一句話概括。

Prime 讓 model 往工作環境裡走。

Hermes 讓 model 進入一個可以長期使用的 personal agent service。

OpenClaw 讓 model 進入一個由 Gateway 統一管理的 system。

## Prime：把工作面做成 persistent Python

Prime 的主要設計選擇，是讓 model 在一個持續存在的 Python control environment 裡做事。

請求進入 parent model 之後，model 面對的核心工具不是一排互相獨立的 action，而是一個 persistent Python REPL。

~~~text
request
  -> parent model
  -> persistent Python REPL
       -> files / shell / skills / MCP
       -> rlm.spawn(...)
       -> durable harness state
~~~

RLM 把 context 放進 Python 變數。

模型可以先讀資料，把解析結果留在變數裡，再跑 shell、呼叫 skill、建立 child agent，最後把幾個中間結果組成下一步。

<code>rlm.spawn(...)</code> 也不是單純把另一個 prompt 丟出去。

它在這個 programming model 裡是一個可以被程式控制的 child-agent admission call，parent 可以保留 handle，之後再透過 message 或檔案接收結果。

Continual Harness 負責另一半。[^continual-harness]

它把 supplemental prompt、memory、skill description 和可重用的 subagent specification 保存成 durable state，讓這些工作規則可以跨過一次 chat window。

這個組合真正改變的是工作單位。

一個 coding／research session 可以反覆讀資料、改程式、跑驗證，再回到同一個 workspace。

它不需要每一輪都把同一組 context 重新拼回 prompt。

Prime 的代價也在同一個位置。[^prime-rlm]

Python REPL 是 model-facing programming surface，不是安全 sandbox。

worker 和 kernel 可以改善 lifecycle isolation、recovery 和 reattach，但模型產生的 Python 或 project command 仍可能沿用使用者的 host 權限。

所以 Prime 的核心進步是 programming model。

它把模型和電腦之間的介面，從「選一個工具」改成「在持續存在的工作面裡寫控制流」。

## Hermes：把 loop 收進 personal agent service

Hermes 的出發點不同。

它要處理的是一個人每天反覆使用的 assistant，而不是把每個 session 暴露成一個可程式化的 REPL。

CLI、messaging gateway、ACP、batch 和 API 都會進同一個 AIAgent core。

core 組 prompt、解析 provider、維護 tool registry，再把 terminal、web、MCP 等能力接到模型回合裡。

~~~text
CLI / Telegram / Discord / ACP
  -> gateway
  -> AIAgent core
       -> prompt builder
       -> provider resolver
       -> tool registry
  -> terminal / web / MCP
  -> session + memory
~~~

這種設計讓入口和執行核心分開。

使用者可以從 CLI 開始，也可以從 Telegram 或 Discord 進來，但工作最後都回到同一個 agent loop。

長期使用需要的 state 也放在 service 旁邊。

SQLite／FTS5 保存 session 和搜尋索引，MEMORY.md 與 USER.md 保存個人 context，skills 提供可以重複使用的操作能力，cron 則讓服務在沒有即時互動時啟動工作。

Hermes 的 tool execution 可以落在 local、container 或 remote terminal backend。

這個選擇很重要，因為 personal agent 最後還是要碰到檔案、shell 和網路。

approval pattern 可以限制某些指令，但真正的隔離程度仍然取決於 terminal backend、部署位置和設定。

Hermes 的工程重點不是讓模型自己持有整個控制流。

它把模型放在 AIAgent core 裡，再用 gateway、memory、skills、provider switching 和 cron 把一次互動延長成一個每天可用的服務。

## OpenClaw：Gateway 是 control plane

OpenClaw 面對的問題更接近系統整合。

request 可能從不同 channel、CLI 或 paired node 進來。

Gateway 要先找到對應的 session，再處理 routing、context、policy、model、native tool 和結果保存。

~~~text
channel / CLI / node
  -> Gateway WebSocket
  -> session routing + policy
  -> agent loop
       -> context
       -> model
       -> native tools
  -> workspace / SQLite / retrieval
~~~

因此 Gateway 不是單純的訊息轉發層。

它同時是 session owner、routing layer、channel coordinator 和 plugin host。

cron、automation、nodes 和 plugins 都掛在這個 control plane 周邊。

workspace Markdown、SQLite／FTS5 和 hybrid retrieval 則把 assistant 的長期 context 放到模型外面。

這個架構適合需要跨 channel 維持同一套 assistant 行為的情境。

同一個 user 可能從不同入口進來，Gateway 仍要決定它們是否屬於同一個 session、使用哪一個 agent、允許哪些工具，以及結果要送回哪裡。

它也把信任邊界集中到 Gateway 周邊。

pairing 和 approval 可以限制誰能進入或觸發某些動作，但它們不會自動把 native plugin 變成隔離程序。

在固定的 <code>v2026.8.1</code> release 裡，plugin 的執行位置、sandbox 設定和 node 權限需要分開檢查。[^openclaw-security]

OpenClaw 的主要工程貢獻是 control-plane architecture。

它處理的是多入口、多 session、多 plugin 和 automation 如何放進同一個 self-hosted runtime。

## Parallelism：task-level，不是 decoder-level

Agent project 很容易讓人誤以為模型本身也變平行了。

其實要分開看兩種 parallelism。

第一種是 task-level parallelism。

Prime 可以用 <code>rlm.spawn(...)</code> 啟動 child agent，Hermes 可以 delegate task，OpenClaw 可以用 automation 或 background path 拆開工作。

這些做法能讓幾個彼此獨立的 research、coding 或 maintenance 工作同時跑。

第二種是 decoder-level parallelism。

同一個 autoregressive sequence 的下一個 token，仍然要依賴前一個 token 的生成結果。

多開幾個 child 不會消除這個 dependency。

所以三個 project 的平行化改善的是工作分派、背景執行和整體吞吐。

它們沒有改變單一回答的 decoder path，也沒有讓 autoregressive generation 變成真正的 token-level parallel generation。

這是 agent runtime 和模型架構的分界。

前者決定很多工作怎麼被接起來，後者才會改變 token 怎麼生成。

## State 不是 model weights

Agent 看起來有記憶，通常是因為 harness 把資料保存到模型外面。

Prime 留下的是 Python workspace、Continual Harness 的 prompts、memories、skill descriptions、child specs 和工作產物。

這些 state 讓同一個 long-running session 接著做。

Hermes 保存 session history、SQLite／FTS5、MEMORY.md、USER.md 和 skills。

這些資料服務的是 personal assistant：找回舊對話、保留個人偏好、沿用工具設定，再由 cron 啟動下一次工作。

OpenClaw 把 workspace Markdown、SQLite／FTS5 和 hybrid retrieval 放在 Gateway 周邊。

Gateway 再決定哪些 context 進入這次 session，以及結果要保存到哪個外部 state。

三者都可以讓下一輪工作接續前一輪。

但它們做的事情是保存、搜尋和重新注入 context，不是更新 model weights。

如果要判斷一個 agent 的「記憶」有多可靠，應該追 state 的寫入、索引、取回和權限，而不是只看 UI 上有沒有一個 memory feature。

## 安全性要拆成 execution、approval 和 isolation

這三個詞常被放在同一個 security label 裡，但它們回答不同問題。

| 層 | 問題 |
| --- | --- |
| Execution | 指令實際在哪個 process、container 或 remote worker 執行。 |
| Approval／admission | 誰可以讓這個 action 進入執行路徑。 |
| Isolation | 這個 process 最後能碰到多少 host 資源、credential 和其他 session。 |

Prime 的 persistent workspace 和 daemon 很適合長任務，但 worker／kernel 的 lifecycle management 不等於 sandbox。

模型生成的 Python 或 shell 仍可能沿用 user process 的權限。

Hermes 可以把 terminal 放在 local、Docker、SSH、Singularity、Modal、Daytona 或 Vercel Sandbox 等 backend，並用 approval pattern 控制某些指令。

實際邊界取決於你選了哪一個 backend，以及 deployment 如何配置。

OpenClaw 的 Gateway 管理 pairing、approval 和 policy。

plugin、node 和 Gateway process 之間的信任關係，則要另外檢查。

尤其不能把「使用者已配對」直接解讀成「所有 plugin 都被隔離」。

部署時我會沿著四條線走一次：process、credential、plugin 和 host permission。

只看到 approval 開關，還不足以推導出 isolation。

## 這算不算本質進步

如果把本質進步定義成改變 decoder、training objective 或 token generation complexity，這三個 project 都不屬於那一類。

它們的改變發生在模型外面。

Prime 的改變比較接近 programming model。

它讓模型在 persistent Python workspace 裡組織 context、工具和 child task，這會直接改變 long-running coding／research 的工作方式。

Hermes 的改變主要在 personal agent service。

它把入口、provider、memory、skills、search、cron 和 terminal backend 收在同一個每天可用的服務裡，重點是可持續使用和部署彈性。

OpenClaw 的改變主要在 control plane。

它把 channel、node、session、plugin、automation 和 policy 放到 Gateway 這個系統邊界裡，重點是多入口和外部元件的整合。

所以三者的「新」不是同一種新。

Prime 比較像在重新定義模型如何操作電腦。

Hermes 比較像把 agent 變成一個可以長期使用的個人服務。

OpenClaw 比較像在做 agent system 的 control plane。

把它們排成模型能力排行榜，會漏掉真正的差異。

## 怎麼選

研究或寫程式時，如果工作需要在同一個 session 裡反覆讀資料、改程式、跑驗證，Prime 的 RLM／persistent Python 對應這個工作。

如果你要的是每天從 CLI 或聊天入口使用的 personal assistant，還要保留 memory、skills、provider switching 和 cron，Hermes 的 AIAgent service 對應這個工作。

如果系統要同時接多個 channel、device、plugin 和 automation，再由一個 self-hosted control plane 統一管理，OpenClaw 的 Gateway 對應這個工作。

這三個 project 都在 agent runtime 上做了完整的工程選擇。

它們沒有替 decoder 提供一個新的生成機制。

真正需要比較的是：工作由哪一層持有，state 由哪一層保存，權限又由哪一個 process 承擔。

[^prime-readme]: [Prime Agent README @ 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/README.md). Used for the product scope, RLM, Continual Harness, long-running sessions, and trust warning.
[^prime-rlm]: [Prime Agent RLM programming model @ 1fc1adb6](https://github.com/PrimeIntellect-ai/prime-agent/blob/1fc1adb6e8062bf871a9b59705c1d15468e589f0/packages/coding-agent/docs/rlm.md). Used for the persistent Python surface, child-agent lifecycle, durable state, and host bridge.
[^continual-harness]: [Continual Harness](https://arxiv.org/abs/2605.09998). Used for the harness-state framing; this article does not treat it as a model-weight update.
[^hermes-readme]: [Hermes Agent README @ afe06f2](https://github.com/NousResearch/hermes-agent/blob/afe06f21f45f476c25034c4529818d9a2f9fdf1c/README.md). Used for the personal-agent scope, gateways, memory/search, skills, cron, delegation, and terminal backends.
[^openclaw-release]: [OpenClaw v2026.8.1](https://github.com/openclaw/openclaw/tree/v2026.8.1). This is the pinned release called OpenClaw 2.0 in this comparison.
[^openclaw-security]: [OpenClaw gateway and security documentation](https://github.com/openclaw/openclaw/tree/v2026.8.1/docs/gateway). Used for Gateway routing, policy, pairing, approval, and deployment-bound security observations.
