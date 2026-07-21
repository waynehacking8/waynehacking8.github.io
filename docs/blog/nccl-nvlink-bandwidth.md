---
description: "Measured NCCL collectives on an 8x H100 NVSwitch host, with paragraph-by-paragraph Traditional Chinese: why 366 GB/s does not help tiny TP-decode messages."
date: "2026-05-31"
updated: "2026-07-21"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/nccl-cube.webp"
tags:
  - NCCL
  - Distributed
  - NVLink
  - LLM Serving
---

# Where tensor-parallel inference hits the NVLink wall｜Tensor parallel inference 何時撞上 NVLink 牆

*2026-05-31 · updated 2026-07-21 · GPU / distributed systems*

<figure class="pb-article-hero">
  <img src="/assets/blog/nccl-cube.webp" alt="NVIDIA NCCL communication stack visualization" loading="eager" decoding="async">
  <figcaption>NCCL communication stack · NCCL 通訊堆疊 · <a href="https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/">Source: NVIDIA Developer</a></figcaption>
</figure>

The large-message number is **366 GB/s**. It is also the wrong number for tensor-parallel decode.

大訊息測到的數字是 **366 GB/s**，但這不是 tensor-parallel decode 該看的數字。
{ .pb-translation lang=zh-Hant }

I measured NCCL on an 8× H100 80 GB SXM5 NVSwitch host, using a four-GPU slice for the main sweep and 2/4/6 GPUs for scaling. All-reduce reaches 77% of the measured 478 GB/s per-GPU unidirectional NVLink budget. Decode lives below 64 KB, where the same collective sits on a roughly **23 μs launch/protocol floor**.[^repo] Multiply that floor by two collectives and 80 layers and bandwidth is no longer the bottleneck.

我在一台配有 8× H100 80 GB SXM5 與 NVSwitch 的主機上量測 NCCL。主要 sweep 使用四張 GPU，scaling 則測 2／4／6 張 GPU。All-reduce 達到每張 GPU 單向 NVLink 實測預算 478 GB/s 的 77%。Decode 的訊息小於 64 KB，此時同一個 collective 會停在約 **23 μs 的 launch／protocol 下限**。[^repo] 把這個固定成本乘上每層兩次 collective 和 80 層之後，瓶頸已經不是頻寬。
{ .pb-translation lang=zh-Hant }

## The measurement, with the denominator attached｜量測結果與比較基準

The harness drives NVIDIA’s `nccl-tests`, records the topology and NCCL tuning logs, then parses every sweep into committed CSV/JSON artifacts.[^nccl-tests] On the four-GPU NCCL 2.18.3 run:

Harness 會執行 NVIDIA 的 `nccl-tests`、記錄 topology 與 NCCL tuning logs，再把每次 sweep 解析成提交到 repo 的 CSV／JSON。[^nccl-tests] 四張 GPU、NCCL 2.18.3 的結果如下：
{ .pb-translation lang=zh-Hant }

| Collective／集合通訊 | Peak bus bandwidth／峰值 bus bandwidth | NVLink-budget share／NVLink 預算占比 | Small-message floor／小訊息下限 |
| --- | ---: | ---: | ---: |
| all-reduce | 366 GB/s | 77% | 22.7 μs |
| all-gather | 344 GB/s | 72% | 16.8 μs |
| reduce-scatter | 350 GB/s | 73% | 21.4 μs |

The host has 18 NVLinks per GPU at 26.562 GB/s each, so the comparison budget is $18\times26.562\approx478$ GB/s in one direction. That is a link budget, not an end-to-end application throughput claim.

每張 GPU 有 18 條 NVLink，每條 26.562 GB/s，因此單向比較預算是 $18\times26.562\approx478$ GB/s。這是 link budget，不是端到端應用程式的吞吐量。
{ .pb-translation lang=zh-Hant }

## A latency model that predicts the wall｜預測瓶頸的 latency model

For one collective, a useful first-order model is

單次 collective 可先用下列一階模型估算：
{ .pb-translation lang=zh-Hant }

$$
T_{\text{collective}}(M) \approx \alpha + \frac{M}{B_{\text{eff}}},
\tag{1}
$$

where $M$ is payload size, $\alpha$ is the launch/protocol floor, and $B_{\text{eff}}$ is measured bandwidth. If a decoder executes $k$ collectives in each of $L$ layers, its communication time per token is approximately

其中 $M$ 是 payload 大小，$\alpha$ 是 launch／protocol 下限，$B_{\text{eff}}$ 是實測頻寬。若 decoder 的 $L$ 層各執行 $k$ 次 collective，每個 token 的通訊時間約為：
{ .pb-translation lang=zh-Hant }

$$
T_{\text{comm/token}} \approx
Lk\left(\alpha + \frac{M_{\text{token}}}{B_{\text{eff}}}\right).
\tag{2}
$$

For an 80-layer model with two all-reduces per layer, a 23.1 μs quiet-box floor alone costs about

對一個 80 層、每層執行兩次 all-reduce 的模型而言，光是 23.1 μs 的 quiet-box 下限就需要：
{ .pb-translation lang=zh-Hant }

$$
80\times2\times23.1\ \mu\text{s}=3.70\ \text{ms/token}.
\tag{3}
$$

That is a hard ceiling of roughly 270 tokens/s before attention, GEMMs, sampling, or any other work. More large-message bandwidth barely moves it because $M_{\text{token}}$ is too small for the second term to dominate.

在還沒算 attention、GEMM、sampling 或其他工作前，這已經把上限壓在約 270 tokens/s。$M_{\text{token}}$ 太小，第二項無法主導結果，因此提高大訊息頻寬幾乎不會改變這個上限。
{ .pb-translation lang=zh-Hant }

## `busbw` is not always physical link traffic｜`busbw` 不一定等於實體鏈路流量

`nccl-tests` reports algorithm bandwidth (`algbw`) and a ring-normalized bus bandwidth. For all-reduce on $N$ ranks, the displayed normalization is

`nccl-tests` 會回報 algorithm bandwidth（`algbw`）與經 Ring 正規化的 bus bandwidth。對 $N$ 個 ranks 的 all-reduce，顯示值的換算方式是：
{ .pb-translation lang=zh-Hant }

$$
B_{\text{bus}} = B_{\text{alg}}\frac{2(N-1)}{N}.
\tag{4}
$$

That conversion describes Ring traffic well. It becomes misleading when the tuner selects NVLS, which reduces data inside NVSwitch. In the scaling run, displayed peak `busbw` rose from 365 GB/s at four GPUs to 443 GB/s at six. The tuning logs show Ring at 2/4 GPUs and NVLS at 6; the six-GPU algorithm bandwidth was 266 GB/s, and physical per-link traffic was closer to that value than to 443.[^scaling-report]

這個換算適合描述 Ring traffic。Tuner 選用在 NVSwitch 內完成 reduction 的 NVLS 時，數字就可能誤導。Scaling run 顯示的 peak `busbw` 從四張 GPU 的 365 GB/s 升到六張 GPU 的 443 GB/s；tuning logs 顯示 2／4 張 GPU 使用 Ring，六張則使用 NVLS。六張 GPU 的 algorithm bandwidth 是 266 GB/s，實體 per-link traffic 也更接近 266，而不是 443。[^scaling-report]
{ .pb-translation lang=zh-Hant }

NVLS still helped: pinned NVLS delivered **21% more algorithm bandwidth** than pinned Ring at six GPUs (266 vs 220 GB/s). The mechanism was less link traffic through in-switch reduction—not a fabric somehow exceeding its measured budget.

NVLS 仍有幫助：六張 GPU 時，固定使用 NVLS 的 algorithm bandwidth 比固定使用 Ring 高 **21%**（266 對 220 GB/s）。原因是 in-switch reduction 減少鏈路流量，不是 fabric 超過了實測預算。
{ .pb-translation lang=zh-Hant }

## CUDA Graphs reduce the fixed term｜CUDA Graphs 降低固定成本

On the controlled quiet-box rerun, eager all-reduce settled at 23.1 μs while CUDA-Graph capture settled at 13.7 μs. In Equation (3), that changes the communication floor from 3.70 ms/token to 2.19 ms/token, or from about 271 to 456 tokens/s before compute.[^latency-report]

在受控的 quiet-box 重跑中，eager all-reduce 穩定在 23.1 μs，CUDA Graph capture 則是 13.7 μs。代入式（3），通訊下限會從 3.70 ms/token 降到 2.19 ms/token；還沒加入 compute 前，上限約從 271 提高到 456 tokens/s。[^latency-report]
{ .pb-translation lang=zh-Hant }

Graphs also removed the long-tail spikes seen in eager mode. An earlier shared-box run had an 82 μs spike that I initially blamed on other NVSwitch tenants. The quiet-box rerun produced a similar spike at a different message size while the graph path remained flat. That rejected the original attribution: the tail came from host-side eager launch jitter, not fabric contention.

Graphs 也消除了 eager mode 的 long-tail spikes。先前 shared-box run 出現一次 82 μs spike，我起初把原因歸咎於其他 NVSwitch tenants。Quiet-box 重跑卻在另一個 message size 出現類似 spike，而 graph path 仍保持平坦。這推翻了原本的解釋：尾端來自 host-side eager launch jitter，不是 fabric contention。
{ .pb-translation lang=zh-Hant }

This correction matters more than a clean chart. A benchmark should make it possible to disprove its own explanation.

這次修正比一張乾淨的圖更重要。Benchmark 應該保留足以推翻自身解釋的證據。
{ .pb-translation lang=zh-Hant }

## Symmetric memory improved bandwidth, not the floor｜Symmetric memory 提高頻寬，沒有降低下限

NCCL 2.29.2 adds symmetric buffer registration. On the same four-GPU slice it did **not** produce the advertised class of small-message latency gain for this path:[^symmetric-report]

NCCL 2.29.2 加入 symmetric buffer registration。在同一組四張 GPU 上，這條路徑沒有測到所宣稱的那類小訊息 latency 改善：[^symmetric-report]
{ .pb-translation lang=zh-Hant }

| Configuration／設定 | Small-message floor／小訊息下限 | BusBW at 16 MB／16 MB BusBW |
| --- | ---: | ---: |
| NCCL 2.18.3 eager reference／eager 參考值 | 23.3 μs | 247 GB/s |
| NCCL 2.29.2 eager | 25.1 μs | 248 GB/s |
| NCCL 2.29.2 eager + symmetric | 23.6 μs | 329 GB/s |
| NCCL 2.29.2 graph | 16.3 μs | 238 GB/s |
| NCCL 2.29.2 graph + symmetric | 19.7 μs | 300 GB/s |

Registration raised large-message bandwidth by 33% (247→329 GB/s) but left the eager floor near 23–25 μs. On this intra-node small-message path, launches dominate; a zero-copy data path cannot remove a kernel launch.

Registration 將大訊息頻寬提高 33%（247→329 GB/s），eager 下限仍落在 23～25 μs。這條 intra-node 小訊息路徑由 launch 成本主導；zero-copy data path 無法消除 kernel launch。
{ .pb-translation lang=zh-Hant }

## What to carry into an inference design review｜Inference design review 要帶走的檢查項目

1. Quote **latency at the model’s actual message size**, not only peak bandwidth.<br><span class="pb-inline-translation" lang="zh-Hant">回報模型實際 message size 的 latency，不能只報 peak bandwidth。</span>
2. Keep `algbw`, normalized `busbw`, and physical traffic separate when the algorithm is not Ring.<br><span class="pb-inline-translation" lang="zh-Hant">演算法不是 Ring 時，分開記錄 `algbw`、正規化 `busbw` 與實體流量。</span>
3. Capture decode collectives in a CUDA Graph before concluding the fabric is slow.<br><span class="pb-inline-translation" lang="zh-Hant">判定 fabric 太慢前，先把 decode collectives capture 進 CUDA Graph。</span>
4. Record topology, versions, tuner decisions, and raw logs; NCCL version changes can move results even on identical hardware.<br><span class="pb-inline-translation" lang="zh-Hant">記錄 topology、版本、tuner 決策與 raw logs；即使硬體相同，NCCL 版本也可能改變結果。</span>
5. Rerun surprising tails under controlled host load, then update the explanation when the evidence rejects it.<br><span class="pb-inline-translation" lang="zh-Hant">在受控 host load 下重跑異常尾端；證據推翻原解釋時，就更新結論。</span>

Tensor-parallel decode issues hundreds of small, ordered collectives per token. Once their fixed launch cost dominates, reducing launch count and scheduling overhead matters more than raising peak large-message bandwidth.

Tensor-parallel decode 每個 token 會發出數百次小型且有順序的 collectives。固定 launch 成本主導後，減少 launch 次數與 scheduling overhead，比提高大訊息峰值頻寬更有效。
{ .pb-translation lang=zh-Hant }

→ Raw logs, CSVs, attribution scripts, and reproduction commands:<br><span class="pb-inline-translation" lang="zh-Hant">Raw logs、CSV、歸因腳本與重現命令：</span>
[github.com/waynehacking8/nccl-collectives-bench](https://github.com/waynehacking8/nccl-collectives-bench)

[^repo]: [NCCL collectives benchmark repository](https://github.com/waynehacking8/nccl-collectives-bench), the primary artifact for hardware state, raw results, analysis scripts, and current conclusions.／主要證據來源，包含硬體狀態、原始結果、分析腳本與目前結論。
[^nccl-tests]: [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests), the canonical benchmark driven by the harness.／Harness 執行的官方 benchmark。
[^scaling-report]: [Algorithm-attributed scaling report](https://github.com/waynehacking8/nccl-collectives-bench/blob/main/results/scaling_attributed/report.md), including pinned Ring/NVLS arms and captured tuner logs.／包含固定 Ring／NVLS 實驗與 tuner logs。
[^latency-report]: [TP latency report](https://github.com/waynehacking8/nccl-collectives-bench/blob/main/results/tp_latency_report.md), comparing eager, CUDA Graph, shared-box, and quiet-box measurements.／比較 eager、CUDA Graph、shared-box 與 quiet-box。
[^symmetric-report]: [NCCL symmetric-memory report](https://github.com/waynehacking8/nccl-collectives-bench/blob/main/results/symmetric/report.md), with NCCL 2.29.2 registered/unregistered raw runs and caveats.／包含 NCCL 2.29.2 註冊與未註冊的原始結果及限制。
