---
description: "150 vLLM servers across five builds on one Blackwell workstation GPU to find which patch clears the FlashInfer illegal memory access in vllm#61239."
date: "2026-11-16"
language: "en"
image: "https://wayne.is-a.dev/assets/blog/vllm-61239-faults.png"
tags:
  - vLLM
  - FlashInfer
  - CUDA
  - LLM Serving
---

# Which patch cleared the FlashInfer fault on my card

*2026-11-16 · vLLM / FlashInfer / debugging*

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/vllm-61239-faults.png" alt="Bar chart: 8 of 30 servers faulted on main before #64165, 0 of 30 on the four other builds" loading="eager" decoding="async">
  <figcaption>Illegal memory accesses per build, 30 servers each<span class="pb-cap-inline"><span class="pb-cap-dot"> · </span>Wayne</span></figcaption>
</figure>

vllm#61239 reports CUDA illegal memory accesses under async scheduling with FlashInfer, on models without speculative decoding.[^issue] Two patches were candidates. #61612 drops the `seq_lens_cpu_upper_bound` shortcut under async scheduling. #64165 is vadiklyutiy's fix for a pipeline parallel crash (#63505): `build()` now waits until the GPU has copied the previous plan out of the page-locked buffer before it calls `plan()` again. I wanted to know which one clears the fault on my card, so I ran 30 fresh servers on each of five builds. Main with #64165 never faulted; main before it faulted in 8 of 30.

## Setup

One Blackwell workstation GPU under WSL2, `openai/gpt-oss-20b` (sliding and full attention, no GDN or linear-attention layers), FlashInfer picked automatically, async scheduling on, no spec decode, no prefix caching, `--max-num-batched-tokens 8192`. Every server starts fresh and runs GSM8K: 1,319 chat requests at 64 concurrent. Five builds, 30 servers each, run interleaved and reshuffled every round, so a bad hour on the box lands on every build alike.

| vLLM | servers | illegal memory access | GSM8K wall time (median, no fault) |
|---|---:|---:|---:|
| main `c7dd34ccea`, before #64165 | 30 | 8 | 69 s |
| `c7dd34ccea` + #61612's `flashinfer.py` change | 30 | 0 | 70 s |
| `c7dd34ccea` + #64165's `flashinfer.py` change | 30 | 0 | 69 s |
| main `7833dce847`, with #64165 | 30 | 0 | 69 s |
| `7833dce847` + #61612's change | 30 | 0 | 70 s |

All 8 faults raised at `copy_event.synchronize()` in `async_utils.py`, 4 to 49 s into the load. 8 of 30 against 0 of 30 gives a one-sided Fisher p of 0.0023. GSM8K accuracy (flexible match) stayed between 0.924 and 0.939 on every build.

## What I take from it

For this case I read #64165 as the fix. Main with it never faulted, and neither did the older main with only #64165's `flashinfer.py` change. I can't say #61612 does nothing, though. Zero faults in 30 servers only puts each build's fault rate under about 10% at 95% confidence, so the run only shows that #61612 isn't needed to clear the fault I saw.

The faults came in a cluster. All 8 landed in unpatched servers 5 to 21, and the last 9 were clean. I still don't know why. A later unpatched run with core dumps on faulted again on its 4th server, so it hadn't gone away.

#64165's description says the new wait never blocked on one GPU (0 of 2,000 builds). I put a counter around its `_plan_copied.synchronize()` on main. Over 3 servers, each with at least 2,000 plan builds, it blocked twice, about 20 ms each, in the middle of the load. So the window does open in normal serving, just rarely.

A CUDA core dump of one unpatched fault puts it in `flashinfer::BatchPrefillWithPagedKVCacheKernel`. That kernel reads its offsets from the plan info `plan()` copies out of the page-locked buffer, so the dump fits the race. A dump can't prove it, though.

## What this doesn't cover

The issue's own shape (one 4,097-token prompt at a time, `--max-num-batched-tokens 2048`, prefix caching on, bf16 or fp8 KV) never faulted in 6 servers here, even unpatched. So this says nothing about that path, on sm80, or with a hybrid model.

This is also WSL2, where vLLM leaves pinned memory off for its own buffers unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set (I didn't set it). FlashInfer still allocates its plan buffer with `pin_memory=True`, so the copy in question is the same, but the timing around it may differ from bare Linux.

I posted the same numbers on the issue on 2026-11-14 and asked whether anyone who hit it on the original setup can retry on current main.[^comment] The scripts, the patches as applied, per-server results, a fault traceback and the core dump reading are in the gist.[^gist]

[^issue]: [vllm-project/vllm#61239](https://github.com/vllm-project/vllm/issues/61239), "FlashInfer metadata builder races under async scheduling for non-speculative models since #57214".
[^comment]: [My comment on vllm#61239](https://github.com/vllm-project/vllm/issues/61239#issuecomment-6877116453), 2026-11-14.
[^gist]: [Scripts, patches, per-server results and the core dump reading](https://gist.github.com/waynehacking8/1ff5e3fedd318ac5c216b1ca7a62255b).
