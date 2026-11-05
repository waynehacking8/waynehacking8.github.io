---
seo_title: Patches | Wayne Chiu
description: Upstream fixes in FlashInfer, vLLM, PyTorch, Dynamo and more.
---

# Patches

My upstream patches to the LLM inference stack: <!-- patches:summary -->.

<p class="pb-pr-wall">
<strong>Every pull request and review, in full, is on the <a class="pb-pr-wall-main" href="https://prs.wayne.is-a.dev">PR wall</a>.</strong>
It rebuilds from GitHub every 10 minutes and is also an <a class="pb-pr-wall-rss" href="https://prs.wayne.is-a.dev/feed.xml">RSS feed</a>.
</p>

This page picks out the ones worth reading, by project. Two themes keep coming
up. One is getting early consumer Blackwell (SM120 / SM121) to work across the
stack: CUTLASS and FlashInfer kernels, the vLLM and SGLang engines, and Dynamo and
TensorRT-LLM disaggregated serving. The other is correctness bugs, where the code
passes its tests and still computes the wrong answer.

## By project

<!-- patches:projects -->

!!! note "Why I look for these"
    The bugs I care about most do not crash. A cuDNN path that computed only the
    first expert group, a `moe_permute` write past the end of its buffer, an attention
    layout mismatch that corrupted logits on new hardware: each one passed its tests.
    I found them by reading the kernels, because there was no stack trace to follow.
