---
description: "150 vLLM servers across five builds on one RTX PRO 6000 to find which patch clears the FlashInfer illegal memory access in vllm#61239, with paragraph-by-paragraph Traditional Chinese."
date: "2026-11-16"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/vllm-61239-faults.png"
tags:
  - vLLM
  - FlashInfer
  - CUDA
  - LLM Serving
---

# Which patch cleared the FlashInfer fault on my card｜在我的卡上，是哪個 patch 修掉了 FlashInfer 的錯誤

*2026-11-16 · vLLM / FlashInfer / debugging*

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/vllm-61239-faults.png" alt="Bar chart: 8 of 30 servers faulted on main before #64165, 0 of 30 on the four other builds" loading="eager" decoding="async">
  <figcaption>Illegal memory accesses per build, 30 servers each · 每個版本 30 個 server 的 illegal memory access 次數 · Wayne</figcaption>
</figure>

vllm#61239 reports CUDA illegal memory accesses under async scheduling with FlashInfer, on models without speculative decoding.[^issue] Two patches were candidates. #61612 drops the `seq_lens_cpu_upper_bound` shortcut under async scheduling. #64165 is vadiklyutiy's fix for a pipeline parallel crash (#63505): `build()` now waits until the GPU has copied the previous plan out of the page-locked buffer before it calls `plan()` again. I wanted to know which one clears the fault on my card, so I ran 30 fresh servers on each of five builds. Main with #64165 never faulted; main before it faulted in 8 of 30.

vllm#61239 回報的是在 async scheduling 加 FlashInfer、而且沒開 speculative decoding 的模型上，會出現 CUDA illegal memory access。[^issue] 當時有兩個候選的 patch。#61612 是在 async scheduling 下拿掉 `seq_lens_cpu_upper_bound` 這個捷徑。#64165 是 vadiklyutiy 為了 pipeline parallel 的 crash (#63505) 做的修正：`build()` 會先等 GPU 把上一份 plan 從 page-locked buffer 複製出去，才再呼叫 `plan()`。我想知道在我的卡上到底是哪一個把錯誤修掉，所以五個版本各開了 30 個全新的 server 來測。結果有 #64165 的 main 一次都沒出錯，沒有它的 main 是 30 次裡錯 8 次。
{ .pb-translation lang=zh-Hant }

## Setup｜測試設定

One RTX PRO 6000 Blackwell Max-Q (sm_120) under WSL2, `openai/gpt-oss-20b` (sliding and full attention, no GDN or linear-attention layers), FlashInfer picked automatically, async scheduling on, no spec decode, no prefix caching, `--max-num-batched-tokens 8192`. Every server starts fresh and runs GSM8K: 1,319 chat requests at 64 concurrent. Five builds, 30 servers each, run interleaved and reshuffled every round, so a bad hour on the box lands on every build alike.

一張 RTX PRO 6000 Blackwell Max-Q (sm_120)，跑在 WSL2 上，模型是 `openai/gpt-oss-20b` (sliding 和 full attention，沒有 GDN 或 linear attention 層)，FlashInfer 讓 vLLM 自己選，開 async scheduling，不開 spec decode 也不開 prefix caching，`--max-num-batched-tokens 8192`。每個 server 都是重新啟動，跑一輪 GSM8K：1,319 個 chat request，64 個同時送。五個版本各跑 30 個 server，交錯執行、每輪重新洗牌，這樣機器狀況不好的那段時間會平均落在每個版本上。
{ .pb-translation lang=zh-Hant }

| vLLM | servers | illegal memory access | GSM8K wall time (median, no fault) |
|---|---:|---:|---:|
| main `c7dd34ccea`, before #64165 | 30 | 8 | 69 s |
| `c7dd34ccea` + #61612's `flashinfer.py` change | 30 | 0 | 70 s |
| `c7dd34ccea` + #64165's `flashinfer.py` change | 30 | 0 | 69 s |
| main `7833dce847`, with #64165 | 30 | 0 | 69 s |
| `7833dce847` + #61612's change | 30 | 0 | 70 s |

All 8 faults raised at `copy_event.synchronize()` in `async_utils.py`, 4 to 49 s into the load. 8 of 30 against 0 of 30 gives a one-sided Fisher p of 0.0023. GSM8K accuracy (flexible match) stayed between 0.924 and 0.939 on every build.

8 次錯誤都是在 `async_utils.py` 的 `copy_event.synchronize()` 丟出來的，時間點在開始送 request 後 4 到 49 秒。拿 8/30 跟 0/30 做單尾 Fisher 檢定，p 值是 0.0023。每個版本的 GSM8K 準確率 (flexible match) 都在 0.924 到 0.939 之間。
{ .pb-translation lang=zh-Hant }

## What I take from it｜我的解讀

For this case I read #64165 as the fix. Main with it never faulted, and neither did the older main with only #64165's `flashinfer.py` change. I can't say #61612 does nothing, though. Zero faults in 30 servers only puts each build's fault rate under about 10% at 95% confidence, so the run only shows that #61612 isn't needed to clear the fault I saw.

在這個情境下，我認為真正修掉問題的是 #64165。有它的 main 一次都沒出錯，舊的 main 只加上 #64165 對 `flashinfer.py` 的改動也一樣。不過我沒辦法說 #61612 完全沒用。30 個 server 零錯誤，只能說每個版本的錯誤率在 95% 信心水準下大概在 10% 以下，所以頂多只能說，要修掉我遇到的這個錯誤，不一定要 #61612。
{ .pb-translation lang=zh-Hant }

The faults came in a cluster. All 8 landed in unpatched servers 5 to 21, and the last 9 were clean. I still don't know why. A later unpatched run with core dumps on faulted again on its 4th server, so it hadn't gone away.

錯誤是集中出現的。8 次全部落在未修正版本的第 5 到第 21 個 server，最後 9 個都沒事。原因我到現在還不知道。但問題沒有自己消失：後來我開著 core dump 再跑一次未修正的版本，第 4 個 server 又出錯了。
{ .pb-translation lang=zh-Hant }

#64165's description says the new wait never blocked on one GPU (0 of 2,000 builds). I put a counter around its `_plan_copied.synchronize()` on main. Over 3 servers, each with at least 2,000 plan builds, it blocked twice, about 20 ms each, in the middle of the load. So the window does open in normal serving, just rarely.

#64165 的描述寫說，在單張 GPU 上這個新的等待從來沒有卡住過 (2,000 次 build 裡 0 次)。我在 main 上的 `_plan_copied.synchronize()` 外面加了計數器。3 個 server、每個至少 2,000 次 plan build，它卡住了 2 次，每次大約 20 ms，都在負載中段。所以正常跑服務的時候，這個 race window 真的會出現，只是很少。
{ .pb-translation lang=zh-Hant }

A CUDA core dump of one unpatched fault puts it in `flashinfer::BatchPrefillWithPagedKVCacheKernel`. That kernel reads its offsets from the plan info `plan()` copies out of the page-locked buffer, so the dump fits the race. A dump can't prove it, though.

其中一次未修正版本出錯時的 CUDA core dump，指到 `flashinfer::BatchPrefillWithPagedKVCacheKernel`。這個 kernel 的 offset 是從 plan info 讀的，而 plan info 就是 `plan()` 從 page-locked buffer 複製出去的那份，所以 dump 的結果跟這個 race 對得上。不過光靠 dump 還不能證明就是它。
{ .pb-translation lang=zh-Hant }

## What this doesn't cover｜這次測試沒涵蓋的部分

The issue's own shape (one 4,097-token prompt at a time, `--max-num-batched-tokens 2048`, prefix caching on, bf16 or fp8 KV) never faulted in 6 servers here, even unpatched. So this says nothing about that path, on sm80, or with a hybrid model.

issue 原本的情境 (一次一個 4,097 token 的 prompt、`--max-num-batched-tokens 2048`、開 prefix caching、bf16 或 fp8 KV) 我在這裡跑了 6 個 server，就算沒修正也一次都沒出錯。所以這次的結果不能拿來說明那條路徑，也不能套到 sm80 或 hybrid 模型上。
{ .pb-translation lang=zh-Hant }

This is also WSL2, where vLLM leaves pinned memory off for its own buffers unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set (I didn't set it). FlashInfer still allocates its plan buffer with `pin_memory=True`, so the copy in question is the same, but the timing around it may differ from bare Linux.

另外這是在 WSL2 上跑的。WSL2 下 vLLM 自己的 buffer 預設不開 pinned memory，除非設 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (我沒設)。FlashInfer 的 plan buffer 還是用 `pin_memory=True` 配置，所以這裡講的那段複製本身沒變，只是前後的時序可能跟原生 Linux 不同。
{ .pb-translation lang=zh-Hant }

I posted the same numbers on the issue on 2026-11-14 and asked whether anyone who hit it on the original setup can retry on current main.[^comment] The scripts, the patches as applied, per-server results, a fault traceback and the core dump reading are in the gist.[^gist]

我在 2026-11-14 把同樣的數字貼到 issue 上，也問了原本遇到問題的人能不能用現在的 main 再試一次。[^comment] 測試 script、套用的 patch、每個 server 的結果、錯誤的 traceback 和 core dump 的判讀都放在 gist 裡。[^gist]
{ .pb-translation lang=zh-Hant }

[^issue]: [vllm-project/vllm#61239](https://github.com/vllm-project/vllm/issues/61239), "FlashInfer metadata builder races under async scheduling for non-speculative models since #57214".
[^comment]: [My comment on vllm#61239](https://github.com/vllm-project/vllm/issues/61239#issuecomment-6877116453), 2026-11-14.
[^gist]: [Scripts, patches, per-server results and the core dump reading](https://gist.github.com/waynehacking8/1ff5e3fedd318ac5c216b1ca7a62255b).
