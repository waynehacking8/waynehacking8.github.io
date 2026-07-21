---
description: "Kimi K3 2.8T 架構解析：KDA 混合線性注意力、Attention Residuals、Stable LatentMoE，以及與 Inkling 975B 的設計分歧。"
date: "2026-07-20"
language: "zh-Hant"
image: "https://kimi-file.moonshot.cn/prod-chat-kimi/kfs/4/2/2026-07-17/d9cs7176rtp4tqfofnsg?x-tos-process=image%2Fauto-orient%2C1%2Fstrip%2Fignore-error%2C1"
tags:
  - Architecture
  - Linear Attention
  - MoE
---

# Kimi K3 2.8T：KDA、AttnRes 與 896 選 16 的 MoE

*2026-07-20 · LLM Architecture / Linear Attention / Agentic Coding*

<figure class="pb-article-hero">
  <img src="/assets/blog/kimi-k3-cover.webp" alt="Moonshot AI 官方 Kimi K3 主視覺" loading="eager" decoding="async">
  <figcaption>官方平台主視覺 · <a href="https://www.kimi.com/blog/kimi-k3">Source: Moonshot AI</a></figcaption>
</figure>

Moonshot AI 發布 **Kimi K3** API：2.8T 總參數、原生視覺、1M-token context，定位在長周期 coding、knowledge work 與 reasoning。它把超長序列與超深網路的問題分別交給 **Kimi Delta Attention（KDA）** 與 **Attention Residuals（AttnRes）**。[^k3-overview]

!!! note "發布狀態（2026-07-20）"
    Kimi K3 已可在 Kimi.com、Kimi Work、Kimi Code 與 API 使用；官方文件表示完整模型權重將在 **2026-07-27 前**釋出。本文不把尚未發生的權重釋出寫成已完成事件，架構細節也以現有官方資訊為限。[^k3-quickstart]

## TL;DR

- **2.8T parameters**，官方稱為首個 3T 級開放模型。
- 使用 **KDA hybrid linear attention** 與 **AttnRes**。
- Stable LatentMoE：896 個 experts 中啟用 16 個。
- 原生視覺理解與 1M context。
- 官方稱相較 K2 約有 **2.5× overall scaling efficiency**；這是廠商數據，完整技術報告仍待發布。
- API 已上線；完整權重截至本文日期仍是預告狀態。

## 1. KDA：用 channel-wise gating 更新長序列狀態

標準 full attention 的序列長度成本快速上升；MLA 主要壓縮 KV 表示，而 KDA 走的是 hybrid linear attention 路線。Moonshot 先前公開的 Kimi Linear 將 Delta Attention 擴展為具有細粒度 gating 的 recurrent-style state update，讓模型能按特徵維度控制資訊寫入、保留與遺忘。[^kda-paper]

KDA 把大量 token-to-token 的歷史互動壓進可更新狀態，再與其他 attention 路徑混合。對 1M context，它提供比每一步都回看完整歷史更可擴展的計算形態。

目前 K3 的完整 layer ratio、kernel 細節與消融實驗要等 technical report；因此，網路流傳的精確 6.3× 解碼速度不能在缺乏同硬體、同 batch 與同輸出設定下直接泛化到所有部署。

### Linear attention 維護的狀態

標準 causal attention 在第 $t$ 個 token 需要讀取先前所有 key/value；線性或 recurrent-style attention 則維護一個固定形狀的狀態 $S_t$，概念上可寫成：

$$
S_t = G_t \odot S_{t-1} + U_t,
\qquad y_t = q_t^\top S_t.
$$

$G_t$ 是遺忘／保留門，$U_t$ 是當前 token 寫入的更新。KDA 的 channel-wise gating 重點，是不同 feature channels 不必共用同一個衰減速率。這將「回看全部歷史」改成「更新摘要狀態」，但也帶來新的 kernel 問題：狀態更新有序列依賴，training 需要可平行化 scan，decode 則要讓 state 常駐在高速記憶體中。架構上的線性複雜度，只有在 kernel fusion 與 memory layout 做好時才會變成實際吞吐。

## 2. AttnRes：殘差不再只連上一層

傳統 residual connection 通常把上一層狀態與當前 transformation 相加。AttnRes 則讓深層網路可依內容從先前層的輸出中選擇與聚合資訊，目標是改善非常深的模型裡，訊號跨層傳遞與梯度路徑逐漸稀釋的問題。[^attnres-paper]

KDA 與 AttnRes 分別處理兩個方向的資訊流：

- KDA 處理**序列方向**的長距離資訊流。
- AttnRes 處理**網路深度方向**的跨層資訊流。

官方公布的 2.5× overall scaling efficiency 是 KDA、AttnRes、訓練與資料 recipe 的合計結果，不能單獨歸因於 AttnRes，也不等同於訓練時間縮短 60%。

## 3. Stable LatentMoE：896 選 16

K3 將 MoE 稀疏度推到 **896 experts、每次啟用 16 個**，搭配 Stable LatentMoE。超大總參數提供更大的模型容量，稀疏啟用則控制每個 token 實際執行的計算量。

16/896 只描述 active experts。部署仍需儲存完整權重，並負擔跨 GPU／節點 routing、memory bandwidth 與負載平衡成本。完整 serving 成本要等權重、量化方案與 vLLM 等 runtime 的實測。

### 三個不能混為一談的 MoE 數字

評估 896 選 16 時，至少要分開看三個 budget：

1. **Capacity**：2.8T 總參數決定 checkpoint 與可容納的專家知識容量。
2. **Active compute**：每 token 啟用 16 個 experts，決定主要矩陣乘法量，但還要加 attention、router 與 shared components。
3. **Communication**：token dispatch/all-to-all 取決於 expert placement、batch 組成與負載平衡；它可能在 FLOPs 尚未飽和前先受網路與 HBM bandwidth 限制。

提高稀疏度也會增加 routing 與負載平衡壓力。若 token 分布偏向少數 experts，capacity factor 會造成 padding 或 dropped／rerouted tokens；若 experts 橫跨節點，all-to-all latency 也會成為 decode 的尾延遲來源。

## 4. 與 Inkling 975B 的路線差異

| 面向 | Thinky Inkling | Moonshot Kimi K3 |
|---|---|---|
| 總參數 | 975B | 2.8T |
| 長序列主軸 | 5:1 sliding-window / global attention | KDA hybrid linear attention |
| 深層訊號 | short convolution + residual stream | Attention Residuals |
| MoE | 256 routed + 2 shared；6 routed active | 896 experts；16 active |
| 多模態 | 文字、圖片、音訊 | 原生視覺（官方目前重點） |
| 定位 | 可客製的通用多模態底座 | 長周期 coding 與 knowledge work |

兩者都在離開「每層 full attention＋RoPE＋同一套 optimizer」的單一路線，但方法不同。Inkling 降低全域 attention 的出現頻率，再用卷積補局部混合；K3 把序列記憶改造成 KDA state update，並用 AttnRes 改造深度方向的訊號傳遞。

## 5. API 與部署策略

K3 API 提供固定 1M context，官方文件表示不因 context 長度採階梯式單價；cache hit、cache miss 與 output 仍分別計價。它也提供 automatic context caching、tool calls、structured output、dynamic tool loading，以及 low/high/max 的 reasoning effort 設定。[^k3-pricing]

截至 7 月 20 日，只有 API 已上線；權重與完整 technical report 尚未公開，因此還沒有可驗證的 vLLM 吞吐、量化品質或多節點 routing 結果。

## 6. Moonshot 主打 long-horizon coding

Moonshot 展示的 long-horizon coding 任務包括理解大型 repository、協調 terminal tools，以及用視覺回饋處理 frontend、game development 與 CAD。官方 kernel optimization case study 讓模型在最多 24 小時的 sandbox 內反覆 profile、改寫與 benchmark GPU kernels。

Inkling 主打可控制的推理成本與文字／圖片／音訊通用底座；K3 的產品展示則集中在長工作流與大型工程任務。兩者的輸入模態與服務場景不同，單張 benchmark 表不足以比較部署取捨。

## 榜單的比較限制

官方自己承認 K3 整體能力仍落後最強 proprietary models。Arena、GDPval 或 kernel benchmark 各自測量不同能力，而且 vendor-reported 結果可能使用不同 harness、thinking budget 與工具權限。Frontend Arena 的單項名次、Artificial Analysis 指標或 hallucination rate，都應附上測試版本與方法，不能拼成一個「全面勝過某模型」的結論。

## 權重發布後還缺哪些資料

權重與 technical report 發布後，仍需補上精度格式、單節點／多節點吞吐、expert communication、長 context 品質和量化退化，才能判斷 2.8T 架構的實際部署成本。

[^k3-overview]: [Moonshot AI, “Kimi K3”](https://www.kimi.com/blog/kimi-k3). Used for the announced architecture and product positioning.
[^k3-quickstart]: [Kimi K3 Quickstart & Specifications](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart). Used for availability, context length, and release-state claims.
[^kda-paper]: [“Kimi Linear: An Expressive, Efficient Attention Architecture”](https://arxiv.org/abs/2510.26692). Used for the KDA mechanism; K3-specific layer ratios remain undisclosed.
[^attnres-paper]: [“Attention Residuals”](https://arxiv.org/abs/2603.15031). Used for the cross-layer residual mechanism.
[^k3-pricing]: [Kimi API pricing](https://platform.kimi.ai/docs/pricing/chat-k3). Used for cache and token-pricing distinctions.
