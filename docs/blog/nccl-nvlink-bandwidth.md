# Where tensor-parallel inference hits the NVLink wall

*2026-05-31 · GPU / distributed systems*

Tensor parallelism splits each layer across GPUs, so every forward pass pays for an
**all-reduce** over the network fabric. On a single node that fabric is NVLink/NVSwitch — and
how close you get to its theoretical budget decides whether TP helps or hurts. This post
measures it on **4× H100** and explains where the wall is.

Repo with the full harness and CSVs:
[**nccl-collectives-bench**](https://github.com/waynehacking8/nccl-collectives-bench).

## What was measured

A bandwidth sweep (message size 8 B → 8 GB) of the three collectives that bound distributed
LLM work — **all-reduce, all-gather, reduce-scatter** — driving the canonical
`nvidia/nccl-tests` and adding a parser + analysis layer on top. The headline number:

- **All-reduce bus bandwidth ≈ 366 GB/s**, about **77 % of the per-GPU NVLink uni-directional
  budget** on this box. That 77 % is the practical ceiling TP communication runs into; the
  remaining gap is protocol overhead and the algorithm's traffic multiplier.
- Algorithm ranking at large messages: **NVLS > Ring > Tree**. NVLink SHARP (NVLS) offloads
  the reduction into the switch, which is why it pulls ahead once messages are big enough to
  amortise setup.
- A **protocol study** (Simple / LL / LL128) showing the small-message latency floor — the
  regime that actually matters for **decode**, where each token's all-reduce is tiny.

## Why it matters for inference

Training all-reduces gradients on big tensors, so it lives in the bandwidth-bound regime
where 366 GB/s is good news. **Decode is the opposite**: one token at a time means small
messages, so you're pinned against the *latency* floor, not the bandwidth ceiling. That is the
real "TP wall" — past a certain TP degree, the per-token all-reduce latency dominates and
adding GPUs makes decode *slower*, not faster.

The repo also includes an **eager-vs-CUDA-Graph** comparison of that decode latency wall:
capturing the per-token step as a graph removes launch overhead that would otherwise be
indistinguishable from communication cost — a reminder to measure the right thing before
blaming the fabric.

## Takeaway

"Use tensor parallelism" is not free advice. Measure the all-reduce on *your* fabric, know
your 77 %, and know that the number that decides decode latency is the small-message floor —
not the big-message bandwidth everyone quotes.

→ Methodology, raw CSVs, and the roofline analysis:
[github.com/waynehacking8/nccl-collectives-bench](https://github.com/waynehacking8/nccl-collectives-bench)
