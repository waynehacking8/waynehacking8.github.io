---
description: "Thinking Machines Lab Inkling 975B 架構解析：5:1 滑動視窗、相對位置編碼、短卷積、聯合正規化 MoE，以及 Muon/Adam 混合優化。"
date: "2026-07-20"
image: "https://thinkingmachines.ai/news/introducing-inkling/images/cover-social-inkling-post.png"
---

# Inkling 975B：不只把模型做大，而是重新分配每一分算力

*2026-07-20 · LLM Architecture / MoE / Multimodal*

<figure class="pb-article-hero">
  <img src="https://thinkingmachines.ai/news/introducing-inkling/images/cover-social-inkling-post.png" alt="Thinking Machines Lab Inkling 官方發布主視覺" loading="eager" decoding="async">
  <figcaption>官方發布主視覺 · Source: Thinking Machines Lab</figcaption>
</figure>

Thinking Machines Lab 在 2026 年 7 月 15 日發布首個開放權重模型 **Inkling**：總參數 975B、每個 token 啟用 41B，支援最高 1M tokens context。它以 45T 個文字、圖片、音訊與影片 tokens 預訓練，重點不是單一 benchmark 冠軍，而是成為可微調、原生多模態且能控制推理成本的基礎模型。

## TL;DR

- **975B total / 41B active** 的 MoE Transformer，1M context。
- 注意力層採 **5:1 sliding-window / global** 混合，而不是每層都做全域注意力。
- 捨棄 RoPE，採用 **relative positional embedding**。
- 在 attention 投影與 residual branch 加入 **short convolution**。
- 256 個 routed experts、2 個 shared experts，每個 token 啟用 6 個 routed experts；兩類 expert 的分數會一起正規化。
- 大型矩陣使用 **Muon**，其他參數使用 **Adam**。
- 視覺與音訊採 encoder-free 輸入，再與文字 tokens 共同處理。

## 1. 不是壓縮 KV，而是減少全域注意力出現的頻率

MLA 等方法的主要價值是壓縮 KV cache；這和「是否每層都執行全域注意力」是兩個不同維度。Inkling 選擇更直接的結構取捨：每六個 attention layers 中，五層只看局部滑動視窗，一層保留全域視野，並使用 8 個 KV heads。

這不代表模型完全放棄長距離資訊。全域層仍能週期性地交換跨區段訊息，而局部層把大部分計算集中在鄰近 token。對長序列而言，這種設計的價值在於降低昂貴全域 attention 的密度，而不只是縮小 cache 表示。

需要注意：官方目前公布的是架構比例與 benchmark，沒有提供足以把所有速度差異拆成單一機制的完整消融實驗。因此，「5:1 必然比所有壓縮注意力更快」仍不應當成普遍結論。

## 2. 相對位置編碼取代 RoPE

Inkling 使用相對位置 embedding，而不是目前 Llama、Qwen 等模型常見的 RoPE。官方表示，在這套混合滑動視窗架構中，相對位置表示對更長序列有更好的外推表現。

直覺上，RoPE 把位置資訊旋轉進 query/key；相對位置方法則直接表達 token 間的相對距離。後者和固定局部視窗很自然地搭配：局部 attention 最在意的本來就是「相隔幾格」，全域層再負責跨視窗整合。

這不是宣告 RoPE 已失效，而是提醒我們：位置編碼應該與 attention topology 一起設計，不能脫離模型其他結構單獨比較。

## 3. Short convolution 負責局部混合

Inkling 在兩個位置加入短卷積：

1. 每個 attention layer 的 key/value projection 之後。
2. attention 與 MLP residual branch 回到主 residual stream 之前。

短卷積擅長低成本的鄰近訊息混合，可先處理局部模式，再把 attention 留給內容相關、距離更長的關係。它不是「取代 attention」，而是替 Transformer 增加一條具有局部 inductive bias 的資料路徑。

## 4. MoE：shared 與 routed experts 一起結算

Inkling 的 MoE 大致沿用 DeepSeek-V3：每層有 **256 routed experts + 2 shared experts**，每個 token 選出 6 個 routed experts，router 使用 sigmoid，並採 auxiliary-loss-free load-balancing bias。

值得注意的差異是，選中的 routed expert 分數與 shared expert 分數會被**聯合正規化**，再加權合併輸出。這比把共享與路由兩條支線完全獨立相加更容易讓控制器在同一尺度上分配貢獻。

Thinking Machines 的發布文與 model card 描述了這個機制，但沒有替它命名；LMSYS/SGLang 與 Thinking Machines 共同發布的 Day-0 系統文章則稱之為 **shared-expert sink**。因此「Sink MoE」可以作為工程簡稱，但它的具體意義仍是 shared/routed experts 共用同一組正規化權重預算。

## 5. Muon + Adam，而不是所有參數都交給 AdamW

訓練採混合優化策略：大型 matrix weights 使用 **Muon**，其他參數交給 **Adam**。官方也將 weight decay strength 與 learning rate 的平方耦合，以控制不同訓練長度下的權重尺度。

Muon 對矩陣更新施加更符合矩陣幾何的處理，而 bias、norm 等向量或純量參數仍保留給成熟的 Adam 路線。這是一種依參數結構分工的工程選擇；官方並未宣稱它只靠優化器就消除了所有 loss spike，也沒有把節省 VRAM 作為這段設計的唯一結論。

## 6. 原生多模態，但不是「完全沒有任何轉換層」

Inkling 的視覺與音訊元件從零開始共同訓練。音訊使用 dMel spectrogram，圖片切成 40×40 patches，經四層 hMLP 與輕量 embedding 後，和文字 tokens 一起交由主模型處理。

因此更準確的說法是：它採 **encoder-free multimodal architecture**，沒有大型獨立 vision/audio encoder 再晚期拼接；但圖片和音訊仍需要 patching、spectrogram 與 embedding，不是原始訊號毫無轉換地直接進入 Transformer。

## Benchmark 應該怎麼讀

Thinking Machines 明確表示 Inkling 不是當下最強的開放或封閉模型。它的亮點是能力分布廣、推理 token 效率可控、原生文字／圖片／音訊，以及可透過 Tinker 客製化。部分評測使用官方內部 harness，外部模型則可能採 self-reported numbers；跨模型比較時應把 harness 與 checkpoint 差異一起看。

另外，「完全沒有蒸餾」也不是精確說法：官方表示預訓練從零開始，但 post-training 的初始 SFT 使用了包括 Kimi K2.5 在內的開放權重模型生成之合成資料，之後主要算力才投入大規模 RL。

## 結語

Inkling 最值得關注的不是 975B 這個數字，而是它把效率問題拆成多個結構層級：局部／全域 attention 的比例、相對位置、短卷積、expert 聯合正規化，以及按參數形狀選 optimizer。這些選擇共同指向同一件事：下一階段的 scaling 不只是在相同公式上堆更多 FLOPs，而是決定哪些資訊值得在哪一層被計算。

## References

- [Inkling 官方發布文](https://thinkingmachines.ai/news/introducing-inkling/)
- [Inkling Model Card](https://thinkingmachines.ai/model-card/inkling/)
- [LMSYS：Inkling Day-0 Support](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support)
