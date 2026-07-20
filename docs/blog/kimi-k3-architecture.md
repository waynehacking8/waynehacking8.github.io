---
description: "Kimi K3 2.8T 架構解析：KDA 混合線性注意力、Attention Residuals、Stable LatentMoE，以及與 Inkling 975B 的設計分歧。"
date: "2026-07-20"
image: "https://kimi-file.moonshot.cn/prod-chat-kimi/kfs/4/2/2026-07-17/d9cs7176rtp4tqfofnsg?x-tos-process=image%2Fauto-orient%2C1%2Fstrip%2Fignore-error%2C1"
---

# Kimi K3 2.8T：用 KDA 與 AttnRes 推進 3T 級開放模型

*2026-07-20 · LLM Architecture / Linear Attention / Agentic Coding*

<figure class="pb-article-hero">
  <img src="https://kimi-file.moonshot.cn/prod-chat-kimi/kfs/4/2/2026-07-17/d9cs7176rtp4tqfofnsg?x-tos-process=image%2Fauto-orient%2C1%2Fstrip%2Fignore-error%2C1" alt="Moonshot AI 官方 Kimi K3 主視覺" loading="eager" decoding="async">
  <figcaption>官方平台主視覺 · Source: Moonshot AI</figcaption>
</figure>

Moonshot AI 發布 **Kimi K3** API：2.8T 總參數、原生視覺、1M-token context，定位在長周期 coding、knowledge work 與 reasoning。它把超長序列與超深網路的問題分別交給 **Kimi Delta Attention（KDA）** 與 **Attention Residuals（AttnRes）**。

!!! note "發布狀態（2026-07-20）"
    Kimi K3 已可在 Kimi.com、Kimi Work、Kimi Code 與 API 使用；官方文件表示完整模型權重將在 **2026-07-27 前**釋出。本文不把尚未發生的權重釋出寫成已完成事件，架構細節也以現有官方資訊為限。

## TL;DR

- **2.8T parameters**，官方稱為首個 3T 級開放模型。
- 使用 **KDA hybrid linear attention** 與 **AttnRes**。
- Stable LatentMoE：896 個 experts 中啟用 16 個。
- 原生視覺理解與 1M context。
- 官方稱相較 K2 約有 **2.5× overall scaling efficiency**；這是廠商數據，完整技術報告仍待發布。
- API 已上線；完整權重截至本文日期仍是預告狀態。

## 1. KDA：把長序列狀態更新做得更細

標準 full attention 的序列長度成本快速上升；MLA 主要壓縮 KV 表示，而 KDA 走的是 hybrid linear attention 路線。Moonshot 先前公開的 Kimi Linear 將 Delta Attention 擴展為具有細粒度 gating 的 recurrent-style state update，讓模型能按特徵維度控制資訊寫入、保留與遺忘。

這種設計的核心不是「完全不需要 attention」，而是把大量 token-to-token 的歷史互動壓進可更新狀態，再與其他 attention 路徑混合。對 1M context，它提供比每一步都回看完整歷史更可擴展的計算形態。

目前 K3 的完整 layer ratio、kernel 細節與消融實驗要等 technical report；因此，網路流傳的精確 6.3× 解碼速度不能在缺乏同硬體、同 batch 與同輸出設定下直接泛化到所有部署。

## 2. AttnRes：殘差不再只連上一層

傳統 residual connection 通常把上一層狀態與當前 transformation 相加。AttnRes 則讓深層網路可依內容從先前層的輸出中選擇與聚合資訊，目標是改善非常深的模型裡，訊號跨層傳遞與梯度路徑逐漸稀釋的問題。

可以把它理解成兩種不同維度的記憶：

- KDA 處理**序列方向**的長距離資訊流。
- AttnRes 處理**網路深度方向**的跨層資訊流。

官方稱 KDA、AttnRes、訓練與資料 recipe 合計，讓 K3 相較 K2 的 overall scaling efficiency 約提升 2.5×。這是整體結果，不能只歸因於 AttnRes，也不能直接解讀為訓練時間必然縮短 60%。

## 3. Stable LatentMoE：896 選 16

K3 將 MoE 稀疏度推到 **896 experts、每次啟用 16 個**，搭配 Stable LatentMoE。超大總參數提供更大的模型容量，稀疏啟用則控制每個 token 實際執行的計算量。

但 active FLOPs 低，不代表部署免費。完整權重仍需要龐大儲存與記憶體容量，跨 GPU／跨節點的 expert routing 也可能受通訊、memory bandwidth 與負載平衡限制。真正的 serving 成本要等權重、量化方案與 vLLM 等 runtime 的實測，而不能只從「16/896」推導。

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

K3 API 提供固定 1M context，官方文件表示不因 context 長度採階梯式單價；cache hit、cache miss 與 output 仍分別計價。它也提供 automatic context caching、tool calls、structured output、dynamic tool loading，以及 low/high/max 的 reasoning effort 設定。

這裡也要區分兩件事：API 能用，不等於開放權重部署已成熟。截至 7 月 20 日，權重與完整 technical report 尚未公開；任何 vLLM 吞吐、量化品質或多節點 routing 結論，都應在實際 artifact 發布後再驗證。

## 6. 核心場景：長鏈條系統工程

Moonshot 把 K3 的核心戰場放在 long-horizon coding：理解大型 repository、協調 terminal tools，並在視覺回饋下處理 frontend、game development 與 CAD。官方 kernel optimization case study 甚至讓模型在最多 24 小時的 sandbox 內反覆 profile、改寫與 benchmark GPU kernels。

這和 Inkling 的差異很鮮明：Inkling 強調可控制的推理成本與文字／圖片／音訊通用底座；K3 更像是為長工作流與大型工程任務設計的 agentic model。兩者不是單純用一張 benchmark 表就能互相取代。

## 如何理性閱讀目前的榜單

官方自己承認 K3 整體能力仍落後最強 proprietary models。Arena、GDPval 或 kernel benchmark 各自測量不同能力，而且 vendor-reported 結果可能使用不同 harness、thinking budget 與工具權限。Frontend Arena 的單項名次、Artificial Analysis 指標或 hallucination rate，都應附上測試版本與方法，不能拼成一個「全面勝過某模型」的結論。

## 結語

K3 的技術訊號比 2.8T 這個 headline 更重要：KDA 解序列長度、AttnRes 解模型深度、極稀疏 MoE 解容量與 active compute 的分離。如果完整權重與 technical report 如期釋出，真正值得看的將是系統層數據——精度格式、單節點／多節點吞吐、expert communication、長 context 品質與量化後退化，而不只是參數量。

## References

- [Kimi K3 官方技術文章](https://www.kimi.com/blog/kimi-k3)
- [Kimi K3 Quickstart & Specifications](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Kimi K3 Pricing](https://platform.kimi.ai/docs/pricing/chat-k3)
- [Kimi Linear / KDA paper](https://arxiv.org/abs/2510.26692)
- [Attention Residuals paper](https://arxiv.org/abs/2603.15031)
