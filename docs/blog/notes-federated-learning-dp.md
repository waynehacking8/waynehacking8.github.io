---
description: "FedAvg under Non-IID data, DP-SGD, and secure aggregation, with paragraph-by-paragraph Traditional Chinese and measured negative results."
date: "2026-05-31"
updated: "2026-07-21"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/federated-dp.webp"
tags:
  - Privacy
  - Federated Learning
  - Differential Privacy
---

# FedAvg Under Non-IID Data: What DP-SGD and Secure Aggregation Protect｜Non-IID 資料下的 FedAvg：DP-SGD 與 Secure Aggregation 各自保護什麼

*2026-05-31 · updated 2026-07-21 · privacy-preserving ML*

<figure class="pb-article-hero">
  <img src="/assets/blog/federated-dp.webp" alt="Google Research 聯邦學習與差分隱私實驗圖" loading="eager" decoding="async">
  <figcaption>Federated learning with formal DP guarantees · 具有正式 DP 保證的聯邦學習 · <a href="https://research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/">Source: Google Research</a></figcaption>
</figure>

The [**federated-learning-lab**](https://github.com/waynehacking8/federated-learning-lab)[^fl-repo] implements FedAvg, FedProx, SCAFFOLD, DP-SGD, and secure aggregation from scratch. Its tests and experiments show where Non-IID data causes client drift, where extra communication rounds matter more than a different optimizer, and which privacy threat each defense covers.

[**federated-learning-lab**](https://github.com/waynehacking8/federated-learning-lab)[^fl-repo] 從頭實作 FedAvg、FedProx、SCAFFOLD、DP-SGD 與 secure aggregation。測試與實驗顯示 Non-IID 資料如何造成 client drift、哪些情況下增加 communication rounds 比換 optimizer 更有影響，以及每種隱私防禦各自處理哪個威脅。
{ .pb-translation lang=zh-Hant }

The repository includes 33/33 passing tests and cross-checks the implementations against the cited literature.

Repository 內含 33／33 通過的測試，並以引用文獻交叉檢查各項實作。
{ .pb-translation lang=zh-Hant }

## 1. The FedAvg training loop｜FedAvg 訓練迴圈

Raw training examples remain on each client, while model updates are sent to an aggregation server. The canonical FedAvg loop is:[^fedavg]

原始訓練樣本留在各個 client，model updates 則傳給 aggregation server。標準 FedAvg 迴圈如下：[^fedavg]
{ .pb-translation lang=zh-Hant }

1. The server broadcasts the global model.<br><span class="pb-inline-translation" lang="zh-Hant">Server 廣播 global model。</span>
2. Each client runs a few local SGD epochs on its own data.<br><span class="pb-inline-translation" lang="zh-Hant">每個 client 用自己的資料執行數個 local SGD epochs。</span>
3. Each client returns its updated weights.<br><span class="pb-inline-translation" lang="zh-Hant">每個 client 回傳更新後的 weights。</span>
4. The server computes a data-size-weighted average to produce the next global model.<br><span class="pb-inline-translation" lang="zh-Hant">Server 依 client data size 加權平均，產生下一版 global model。</span>

FedAvg keeps raw examples local, but heterogeneous client distributions make the local updates drift in different directions.

FedAvg 讓原始樣本留在本地，但異質的 client distributions 會使各地更新往不同方向偏移。
{ .pb-translation lang=zh-Hant }

## 2. The Non-IID problem｜Non-IID 問題

FedAvg works best when participating clients sample similar distributions. Real clients differ: one hospital sees a different case mix from another, and one phone’s keyboard sees different language. Under Non-IID data, the average of those local updates can land in a region that is poor for many clients. This is client drift.

參與的 clients 取樣自相近分布時，FedAvg 最容易運作。現實中的 clients 並不相同：不同醫院會遇到不同病例組合，不同手機鍵盤也會看到不同語言。資料是 Non-IID 時，local updates 的平均可能落在許多 clients 都表現不佳的區域，這就是 client drift。
{ .pb-translation lang=zh-Hant }

The lab implements and measures two common corrections:

Lab 實作並量測兩種常見修正方法：
{ .pb-translation lang=zh-Hant }

- **FedProx** adds a proximal term that penalizes movement too far from the global model. It can stabilize training when clients are heterogeneous.[^fedprox]<br><span class="pb-inline-translation" lang="zh-Hant">**FedProx** 加入 proximal term，懲罰離 global model 太遠的更新；clients 異質時可提高穩定性。[^fedprox]</span>
- **SCAFFOLD** tracks control variates that estimate and subtract the drift direction. It communicates more state, but corrects bias that FedProx only damps.[^scaffold]<br><span class="pb-inline-translation" lang="zh-Hant">**SCAFFOLD** 維護 control variates，估計並扣除 drift direction。它要傳輸更多狀態，但能修正 FedProx 只會抑制的 bias。[^scaffold]</span>

On the label-skewed MNIST split in this repository, FedProx and SCAFFOLD did not consistently outperform FedAvg by a large margin. Increasing communication rounds sometimes had more effect.

在這個 repository 的 label-skewed MNIST split 上，FedProx 與 SCAFFOLD 並未持續大幅勝過 FedAvg；增加 communication rounds 有時反而更有影響。
{ .pb-translation lang=zh-Hant }

## 3. Differential privacy: local data can still leak｜Differential privacy：本地資料仍可能外洩

Keeping raw data on-device does not by itself provide privacy. Model updates can expose information through membership inference or gradient inversion. DP-SGD adds a formal differential-privacy bound to training.[^dpsgd]

把原始資料留在裝置上，本身並不構成隱私保證。Model updates 仍可能透過 membership inference 或 gradient inversion 洩漏資訊。DP-SGD 會在訓練中加入正式的 differential-privacy bound。[^dpsgd]
{ .pb-translation lang=zh-Hant }

DP-SGD changes each training step in two ways:

DP-SGD 對每個 training step 做兩項修改：
{ .pb-translation lang=zh-Hant }

1. **Per-sample gradient clipping** bounds each example’s contribution to a maximum norm `C`.<br><span class="pb-inline-translation" lang="zh-Hant">**Per-sample gradient clipping** 把每筆樣本的貢獻限制在最大 norm `C`。</span>
2. **Gaussian noise** adds noise calibrated to `C` to the summed gradients.<br><span class="pb-inline-translation" lang="zh-Hant">**Gaussian noise** 依 `C` 校準後，加入 summed gradients。</span>

An $(\varepsilon,\delta)$-DP bound limits how much the output distribution can change when one example is added or removed. Lower $\varepsilon$ generally requires more noise, so the report must show privacy budget and utility together.

$(\varepsilon,\delta)$-DP bound 會限制加入或移除一筆樣本時，輸出分布最多能改變多少。較低的 $\varepsilon$ 通常需要更多 noise，因此報告必須同時呈現 privacy budget 與 utility。
{ .pb-translation lang=zh-Hant }

## 4. Secure aggregation hides individual updates｜Secure aggregation 隱藏個別更新

DP limits what the final model can reveal. Secure aggregation addresses a different threat: an aggregation server inspecting each client’s update. Clients mask their updates so the server can recover only the sum; the individual masks cancel when aggregated.[^secagg]

DP 限制 final model 能洩漏的資訊。Secure aggregation 處理另一個威脅：aggregation server 檢查每個 client 的 update。Clients 先遮罩更新，server 只能還原總和；個別 masks 會在聚合時互相抵銷。[^secagg]
{ .pb-translation lang=zh-Hant }

The two mechanisms are complementary. DP bounds model leakage, while secure aggregation limits what the server observes during training.

兩種機制互補：DP 約束 model leakage，secure aggregation 則限制 server 在訓練期間能看到的內容。
{ .pb-translation lang=zh-Hant }

## 5. Implementation checks｜實作檢查

A clipping bug or miscalibrated noise multiplier can invalidate the intended privacy bound. The lab therefore checks four things:

Clipping bug 或校準錯誤的 noise multiplier 都可能使預期的 privacy bound 失效，因此 lab 會檢查四件事：
{ .pb-translation lang=zh-Hant }

- Implementations cover FedAvg, FedProx, SCAFFOLD, FedPer, Byzantine-robust aggregation, FedAdam, and FedLoRA.<br><span class="pb-inline-translation" lang="zh-Hant">實作涵蓋 FedAvg、FedProx、SCAFFOLD、FedPer、Byzantine-robust aggregation、FedAdam 與 FedLoRA。</span>
- DP tests check clipping and noise behavior rather than only final accuracy.<br><span class="pb-inline-translation" lang="zh-Hant">DP 測試會檢查 clipping 與 noise 行為，不只看最終 accuracy。</span>
- Literature cross-checks verify that measured behavior is consistent with the cited methods.<br><span class="pb-inline-translation" lang="zh-Hant">文獻交叉檢查會確認實測行為是否和引用方法一致。</span>
- The 33/33 tests and negative results remain in the repository with reproduction commands.<br><span class="pb-inline-translation" lang="zh-Hant">33／33 項測試、負面結果與重現命令都保留在 repository。</span>

→ Implementations, tests, and negative results:<br><span class="pb-inline-translation" lang="zh-Hant">實作、測試與負面結果：</span>
[github.com/waynehacking8/federated-learning-lab](https://github.com/waynehacking8/federated-learning-lab)

[^fl-repo]: [Federated Learning Lab](https://github.com/waynehacking8/federated-learning-lab), the primary artifact for implementations, tests, and measured negative results discussed here.／本文實作、測試與負面結果的主要來源。
[^fedavg]: [McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data”](https://proceedings.mlr.press/v54/mcmahan17a.html).／FedAvg 原始論文。
[^fedprox]: [Li et al., “Federated Optimization in Heterogeneous Networks”](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html).／FedProx 論文。
[^scaffold]: [Karimireddy et al., “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning”](https://proceedings.mlr.press/v119/karimireddy20a.html).／SCAFFOLD 論文。
[^dpsgd]: [Abadi et al., “Deep Learning with Differential Privacy”](https://dl.acm.org/doi/10.1145/2976749.2978318).／DP-SGD 論文。
[^secagg]: [Bonawitz et al., “Practical Secure Aggregation for Privacy-Preserving Machine Learning”](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/).／Secure aggregation 論文。
