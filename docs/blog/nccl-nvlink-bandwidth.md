---
description: "Measured NCCL collectives on an 8x H100 NVSwitch host: why 366 GB/s does not help tiny TP-decode messages, and what NVLS, CUDA Graphs, and symmetric memory actually change."
date: "2026-05-31"
updated: "2026-07-21"
language: "en"
image: "https://wayne.is-a.dev/assets/blog/nccl-cube.webp"
tags:
  - NCCL
  - Distributed
  - NVLink
  - LLM Serving
---

# Where tensor-parallel inference hits the NVLink wall

*2026-05-31 · updated 2026-07-21 · GPU / distributed systems*

<figure class="pb-article-hero">
  <img src="/assets/blog/nccl-cube.webp" alt="NVIDIA NCCL communication stack visualization" loading="eager" decoding="async">
  <figcaption>NCCL communication stack · <a href="https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/">Source: NVIDIA Developer</a></figcaption>
</figure>

The large-message number is **366 GB/s**. It is also the wrong number for tensor-parallel
decode.

I measured NCCL on an 8× H100 80 GB SXM5 NVSwitch host, using a four-GPU slice for the main
sweep and 2/4/6 GPUs for scaling. All-reduce reaches 77% of the measured 478 GB/s per-GPU
unidirectional NVLink budget. Decode lives below 64 KB, where the same collective sits on a
roughly **23 μs launch/protocol floor**.[^repo] Multiply that floor by two collectives and 80
layers and bandwidth is no longer the bottleneck.

## The measurement, with the denominator attached

The harness drives NVIDIA’s `nccl-tests`, records the topology and NCCL tuning logs, then
parses every sweep into committed CSV/JSON artifacts.[^nccl-tests] On the four-GPU NCCL
2.18.3 run:

| Collective | Peak bus bandwidth | NVLink-budget share | Small-message floor |
| --- | ---: | ---: | ---: |
| all-reduce | 366 GB/s | 77% | 22.7 μs |
| all-gather | 344 GB/s | 72% | 16.8 μs |
| reduce-scatter | 350 GB/s | 73% | 21.4 μs |

The host has 18 NVLinks per GPU at 26.562 GB/s each, so the comparison budget is
$18\times26.562\approx478$ GB/s in one direction. That is a link budget, not an end-to-end
application throughput claim.

## A latency model that predicts the wall

For one collective, a useful first-order model is

$$
T_{\text{collective}}(M) \approx \alpha + \frac{M}{B_{\text{eff}}},
\tag{1}
$$

where $M$ is payload size, $\alpha$ is the launch/protocol floor, and $B_{\text{eff}}$ is
measured bandwidth. If a decoder executes $k$ collectives in each of $L$ layers, its
communication time per token is approximately

$$
T_{\text{comm/token}} \approx
Lk\left(\alpha + \frac{M_{\text{token}}}{B_{\text{eff}}}\right).
\tag{2}
$$

For an 80-layer model with two all-reduces per layer, a 23.1 μs quiet-box floor alone costs
about

$$
80\times2\times23.1\ \mu\text{s}=3.70\ \text{ms/token}.
\tag{3}
$$

That is a hard ceiling of roughly 270 tokens/s before attention, GEMMs, sampling, or any
other work. More large-message bandwidth barely moves it because $M_{\text{token}}$ is too
small for the second term to dominate.

## `busbw` is not always physical link traffic

`nccl-tests` reports algorithm bandwidth (`algbw`) and a ring-normalized bus bandwidth. For
all-reduce on $N$ ranks, the displayed normalization is

$$
B_{\text{bus}} = B_{\text{alg}}\frac{2(N-1)}{N}.
\tag{4}
$$

That conversion describes Ring traffic well. It becomes misleading when the tuner selects
NVLS, which reduces data inside NVSwitch. In the scaling run, displayed peak `busbw` rose
from 365 GB/s at four GPUs to 443 GB/s at six. The tuning logs show Ring at 2/4 GPUs and
NVLS at 6; the six-GPU algorithm bandwidth was 266 GB/s, and physical per-link traffic was
closer to that value than to 443.[^scaling-report]

NVLS still helped: pinned NVLS delivered **21% more algorithm bandwidth** than pinned Ring
at six GPUs (266 vs 220 GB/s). The mechanism was less link traffic through in-switch
reduction—not a fabric somehow exceeding its measured budget.

## CUDA Graphs attack the term that matters

On the controlled quiet-box rerun, eager all-reduce settled at 23.1 μs while CUDA-Graph
capture settled at 13.7 μs. In Equation (3), that changes the communication floor from
3.70 ms/token to 2.19 ms/token, or from about 271 to 456 tokens/s before compute.[^latency-report]

Graphs also removed the long-tail spikes seen in eager mode. An earlier shared-box run had an
82 μs spike that I initially blamed on other NVSwitch tenants. The quiet-box rerun produced a
similar spike at a different message size while the graph path remained flat. That rejected
the original attribution: the tail came from host-side eager launch jitter, not fabric
contention.

This correction matters more than a clean chart. A benchmark should make it possible to
disprove its own explanation.

## Symmetric memory improved bandwidth, not the floor

NCCL 2.29.2 adds symmetric buffer registration. On the same four-GPU slice it did **not**
produce the advertised class of small-message latency gain for this path:[^symmetric-report]

| Configuration | Small-message floor | BusBW at 16 MB |
| --- | ---: | ---: |
| NCCL 2.18.3 eager reference | 23.3 μs | 247 GB/s |
| NCCL 2.29.2 eager | 25.1 μs | 248 GB/s |
| NCCL 2.29.2 eager + symmetric | 23.6 μs | 329 GB/s |
| NCCL 2.29.2 graph | 16.3 μs | 238 GB/s |
| NCCL 2.29.2 graph + symmetric | 19.7 μs | 300 GB/s |

Registration raised large-message bandwidth by 33% (247→329 GB/s) but left the eager floor
near 23–25 μs. On this intra-node small-message path, launches dominate; a zero-copy data
path cannot remove a kernel launch.

## What to carry into an inference design review

1. Quote **latency at the model’s actual message size**, not only peak bandwidth.
2. Keep `algbw`, normalized `busbw`, and physical traffic separate when the algorithm is not
   Ring.
3. Capture decode collectives in a CUDA Graph before concluding the fabric is slow.
4. Record topology, versions, tuner decisions, and raw logs; NCCL version changes can move
   results even on identical hardware.
5. Rerun surprising tails under controlled host load, then update the explanation when the
   evidence rejects it.

The wall is not “NVLink is only 77% utilized.” The wall is that tensor-parallel decode asks a
high-bandwidth fabric to perform hundreds of tiny, ordered collectives per token. Once the
fixed cost dominates, the optimization target is launch count and scheduling—not another
bandwidth headline.

→ Raw logs, CSVs, attribution scripts, and reproduction commands:
[github.com/waynehacking8/nccl-collectives-bench](https://github.com/waynehacking8/nccl-collectives-bench)

[^repo]: [NCCL collectives benchmark repository](https://github.com/waynehacking8/nccl-collectives-bench), the primary artifact for hardware state, raw results, analysis scripts, and current conclusions.
[^nccl-tests]: [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests), the canonical benchmark driven by the harness.
[^scaling-report]: [Algorithm-attributed scaling report](https://github.com/waynehacking8/nccl-collectives-bench/blob/main/results/scaling_attributed/report.md), including pinned Ring/NVLS arms and captured tuner logs.
[^latency-report]: [TP latency report](https://github.com/waynehacking8/nccl-collectives-bench/blob/main/results/tp_latency_report.md), comparing eager, CUDA Graph, shared-box, and quiet-box measurements.
[^symmetric-report]: [NCCL symmetric-memory report](https://github.com/waynehacking8/nccl-collectives-bench/blob/main/results/symmetric/report.md), with NCCL 2.29.2 registered/unregistered raw runs and caveats.
