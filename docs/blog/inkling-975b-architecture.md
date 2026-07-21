---
description: "Thinking Machines Lab Inkling 975B 架構解析：5:1 滑動視窗、相對位置編碼、短卷積、聯合正規化 MoE，以及 Muon/Adam 混合優化。"
date: "2026-07-20"
language: "zh-Hant"
image: "https://thinkingmachines.ai/news/introducing-inkling/images/cover-social-inkling-post.png"
tags:
  - Architecture
  - MoE
  - Multimodal
---

# Inkling 975B：5:1 混合注意力如何重新分配算力

*2026-07-20 · LLM Architecture / MoE / Multimodal*

<figure class="pb-article-hero">
  <img src="/assets/blog/inkling-cover.png" width="1200" height="630" alt="Thinking Machines Lab Inkling 官方發布主視覺" loading="eager" decoding="async">
  <figcaption>官方發布主視覺 · <a href="https://thinkingmachines.ai/news/introducing-inkling/">Source: Thinking Machines Lab</a></figcaption>
</figure>

Thinking Machines Lab 在 2026 年 7 月 15 日發布首個開放權重模型 **Inkling**。模型總參數為 975B，每個 token 啟用 41B，支援最高 1M tokens context，並以 45T 個文字、圖片、音訊與影片 tokens 預訓練。官方將它定位為可微調的原生多模態基礎模型，並用稀疏啟用與混合注意力控制推理成本。[^inkling-release][^inkling-card]

## TL;DR

- **975B total / 41B active** 的 MoE Transformer，1M context。
- 注意力層採 **5:1 sliding-window / global** 混合，而不是每層都做全域注意力。
- 捨棄 RoPE，採用 **relative positional embedding**。
- 在 attention 投影與 residual branch 加入 **short convolution**。
- 256 個 routed experts、2 個 shared experts，每個 token 啟用 6 個 routed experts；兩類 expert 的分數會一起正規化。
- 大型矩陣使用 **Muon**，其他參數使用 **Adam**。
- 視覺與音訊採 encoder-free 輸入，再與文字 tokens 共同處理。

## 1. 用 5:1 混合注意力降低全域計算頻率

MLA 等方法的主要價值是壓縮 KV cache；這和「是否每層都執行全域注意力」是兩個不同維度。Inkling 選擇更直接的結構取捨：每六個 attention layers 中，五層只看局部滑動視窗，一層保留全域視野，並使用 8 個 KV heads。

每六層中仍有一層交換跨區段資訊，其餘五層把大部分計算限制在鄰近 token。對長序列而言，這套配置降低了全域 attention 的密度；它處理的是全域計算出現的頻率，和縮小 cache 表示是不同問題。

官方目前只公布架構比例與 benchmark，尚無完整消融實驗，因此不能把所有速度差異都歸因於 5:1 attention，也不能泛化成它必然比所有壓縮注意力更快。

### 把 5:1 的節省寫成成本式

令序列長度為 $n$、滑動視窗寬度為 $w$。先忽略 head dimension 與常數，一層全域 attention 的 score 計算量近似 $n^2$，局部層則近似 $nw$。六層一組時：

$$
C_{\text{Inkling}} \propto 5nw+n^2,
\qquad
\frac{C_{\text{Inkling}}}{C_{\text{all-global}}}
=\frac{5nw+n^2}{6n^2}
=\frac{1}{6}+\frac{5w}{6n}.
$$

當 $n \gg w$，attention-score 這一項會漸近接近全域六層的六分之一，不會降成零。全域層仍保留 $O(n^2)$ 成本；端到端延遲還包含 QKV projection、MoE、通訊與記憶體搬移，因此上式不代表整個模型會快六倍。

## 2. 相對位置編碼取代 RoPE

Inkling 使用相對位置 embedding，而不是目前 Llama、Qwen 等模型常見的 RoPE。官方表示，在這套混合滑動視窗架構中，相對位置表示對更長序列有更好的外推表現。

直覺上，RoPE 把位置資訊旋轉進 query/key；相對位置方法則直接表達 token 間的相對距離。後者和固定局部視窗很自然地搭配：局部 attention 最在意的本來就是「相隔幾格」，全域層再負責跨視窗整合。

Inkling 的選擇只適用於這套混合滑動視窗架構；位置編碼仍需和 attention topology 一起評估，不能脫離模型其他結構單獨比較。

## 3. Short convolution 負責局部混合

Inkling 在兩個位置加入短卷積：

1. 每個 attention layer 的 key/value projection 之後。
2. attention 與 MLP residual branch 回到主 residual stream 之前。

短卷積先處理鄰近模式，attention 則繼續負責內容相關與較長距離的關係。這替 Transformer 補上一條具有局部 inductive bias 的資料路徑。

## 4. MoE：shared 與 routed experts 一起結算

Inkling 的 MoE 大致沿用 DeepSeek-V3：每層有 **256 routed experts + 2 shared experts**，每個 token 選出 6 個 routed experts，router 使用 sigmoid，並採 auxiliary-loss-free load-balancing bias。

Inkling 會將選中的 routed expert 分數與 shared expert 分數**聯合正規化**，再加權合併輸出。控制器因而能在同一尺度上分配兩類 expert 的貢獻。

Thinking Machines 的發布文與 model card 描述了這個機制，但沒有替它命名；LMSYS/SGLang 與 Thinking Machines 共同發布的 Day-0 系統文章則稱之為 **shared-expert sink**。因此「Sink MoE」可以作為工程簡稱，但它的具體意義仍是 shared/routed experts 共用同一組正規化權重預算。[^inkling-day0]

## 5. Muon 處理矩陣，Adam 處理其他參數

訓練採混合優化策略：大型 matrix weights 使用 **Muon**，其他參數交給 **Adam**。官方也將 weight decay strength 與 learning rate 的平方耦合，以控制不同訓練長度下的權重尺度。

Muon 對矩陣更新施加更符合矩陣幾何的處理，而 bias、norm 等向量或純量參數仍保留給成熟的 Adam 路線。這是一種依參數結構分工的工程選擇；官方並未宣稱它只靠優化器就消除了所有 loss spike，也沒有把節省 VRAM 作為這段設計的唯一結論。

### 部署仍要容納 975B 權重

41B active parameters 描述每個 token 參與計算的權重，不是載入 checkpoint 所需的容量。僅做理想化下界估算，975B 個 BF16 權重約需 $975\times 10^9\times2$ bytes，也就是約 **1.95 TB**；4-bit 權重本體約 **487.5 GB**，尚未包含 scale、metadata、embedding、KV cache 與 runtime workspace。這與官方給出的 BF16 至少 2 TB、NVFP4 至少 600 GB 聚合 VRAM 是一致量級。MoE 省的是 active compute，不會自動消除權重儲存與 expert routing 的頻寬成本。

## 6. Encoder-free 多模態仍保留輸入前處理

Inkling 的視覺與音訊元件從零開始共同訓練。音訊使用 dMel spectrogram，圖片切成 40×40 patches，經四層 hMLP 與輕量 embedding 後，和文字 tokens 一起交由主模型處理。

Inkling 採用 **encoder-free multimodal architecture**，沒有大型獨立 vision／audio encoder 再做晚期拼接。圖片與音訊仍會先經過 patching、spectrogram、hMLP 和 embedding，再送入主模型。

## Benchmark 的比較限制

Thinking Machines 明確表示 Inkling 不是當下最強的開放或封閉模型。官方主打能力分布、推理 token 效率、原生文字／圖片／音訊，以及透過 Tinker 客製化。部分評測使用內部 harness，外部模型則可能採 self-reported numbers；跨模型比較時必須連同 harness 與 checkpoint 差異一起看。

官方表示預訓練從零開始；post-training 的初始 SFT 仍使用了包括 Kimi K2.5 在內的開放權重模型生成之合成資料，之後才把主要算力投入大規模 RL。因此，本文不把它概括成「完全沒有蒸餾」。

## 部署時會碰到的成本

975B 只說明 checkpoint 的規模。實際計算與部署成本還取決於 5:1 attention、41B active parameters、expert routing、權重格式和 context 長度；這些資訊比單看總參數更接近系統真正要付的代價。

[^inkling-release]: [Thinking Machines Lab, “Introducing Inkling”](https://thinkingmachines.ai/news/introducing-inkling/). The release post is the source for the public architecture overview and training-token count.
[^inkling-card]: [Inkling Model Card](https://thinkingmachines.ai/model-card/inkling/). Used for parameter, context, modality, training, and deployment-memory details.
[^inkling-day0]: [LMSYS, “Inkling Day-0 Support”](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support). Used only for the serving-system terminology and implementation notes.
