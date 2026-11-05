---
seo_title: Patches | Wayne Chiu
description: Upstream fixes in FlashInfer, vLLM, PyTorch, Dynamo and more.
---

# Patches

My upstream patches to the LLM inference stack: **24 merged and 26 in review**
as of 2026-11-05.

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

<div class="pb-timeline pb-patches" markdown>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/flashinfer.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/flashinfer-ai/flashinfer">FlashInfer</a></p>
<p class="pb-tl-date">10 merged, 3 in review</p>

<p class="pb-patch-group">Merged</p>

- [SM120/121 multi-CTA radix top-k stream hang](https://github.com/flashinfer-ai/flashinfer/pull/3615)
- [SM120 NVFP4 attention qk-correction layout, row-sum and lse](https://github.com/flashinfer-ai/flashinfer/pull/3838)
- [b12x MoE workspace bucketing to stop kernel-cache thrash](https://github.com/flashinfer-ai/flashinfer/pull/3836)
- [MXFP8-aware cutlass MoE gemm profiler](https://github.com/flashinfer-ai/flashinfer/pull/3614)

<p class="pb-patch-group">In review</p>

- [JIT ninja parallelism capped by available memory](https://github.com/flashinfer-ai/flashinfer/pull/3636)
- [Combined paged-KV tensor in the trace path](https://github.com/flashinfer-ai/flashinfer/pull/3690)

</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/vllm.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/vllm-project/vllm">vLLM</a></p>
<p class="pb-tl-date">7 merged, 7 in review, including llm-compressor and compressed-tensors</p>

<p class="pb-patch-group">Merged</p>

- [Skip Qwen3 deepstack buffers without vision](https://github.com/vllm-project/vllm/pull/49397)
- [Tokenizer survives pickling](https://github.com/vllm-project/vllm/pull/45460)
- [Skip device-map entries with no local module in dispatch](https://github.com/vllm-project/compressed-tensors/pull/737) (compressed-tensors)
- [Research Hub entry: FP8 KV cache on Phi-3's fused QKV](https://github.com/vllm-project/llm-compressor/pull/3344) (llm-compressor)

<p class="pb-patch-group">In review</p>

- [Reject NVFP4 MoE checkpoints with missing per-expert scales](https://github.com/vllm-project/vllm/pull/45320)
- [Upcast E8M0 scales for CUTLASS block FP8 on SM12x](https://github.com/vllm-project/vllm/pull/47988)
- [Stop re-queuing waiting requests that `max_loras` blocks](https://github.com/vllm-project/vllm/pull/65795)

</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/internlm.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/InternLM/lmdeploy">LMDeploy</a></p>
<p class="pb-tl-date">3 merged</p>

<p class="pb-patch-group">Merged</p>

- [InternVL LoRA loading TypeError](https://github.com/InternLM/lmdeploy/pull/4684)
- [Double-counted `max_q_seqlen` in decode delta](https://github.com/InternLM/lmdeploy/pull/4685)
- [Builtin chat-template ImportError](https://github.com/InternLM/lmdeploy/pull/4690)

</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/dynamo.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/ai-dynamo/dynamo">Dynamo</a></p>
<p class="pb-tl-date">2 merged, 4 in review</p>

<p class="pb-patch-group">Merged</p>

- [KV-router cancels in-flight recovery on worker removal](https://github.com/ai-dynamo/dynamo/pull/10616)
- [Resolve aggregate planner workers by DGD component type](https://github.com/ai-dynamo/dynamo/pull/11578)

<p class="pb-patch-group">In review</p>

- [Blackwell workstation GPU SKU in the operator](https://github.com/ai-dynamo/dynamo/pull/10579)
- [OTLP traces without logs](https://github.com/ai-dynamo/dynamo/pull/10806)
- [NIXL push and pull connectors in PD](https://github.com/ai-dynamo/dynamo/pull/11219)

</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/pytorch.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/pytorch/pytorch">PyTorch</a></p>
<p class="pb-tl-date">2 landed, 1 in review, including torchao</p>

<p class="pb-patch-group">Merged</p>

- [`nccl.broadcast` was dropping its `root` argument](https://github.com/pytorch/pytorch/pull/187216), landed in `main` as `c3c33fd`
- [PT2E/X86 plain-linear annotation fallback for reused `nn.Linear`](https://github.com/pytorch/ao/pull/4480) (torchao)

<p class="pb-patch-group">In review</p>

- [Extend the reused-module fallback to `nn.Conv2d`](https://github.com/pytorch/ao/pull/4482) (torchao)

</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/nvidia.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/NVIDIA">NVIDIA</a></p>
<p class="pb-tl-date">6 in review across CUTLASS, TensorRT-LLM and Triton</p>

<p class="pb-patch-group">In review</p>

- [SM120 grouped NVFP4 block-scaled GEMM in `cutlass_library`](https://github.com/NVIDIA/cutlass/pull/3346) (CUTLASS)
- [CuTeDSL `make_ptr` on sub-byte dtypes](https://github.com/NVIDIA/cutlass/pull/3334) (CUTLASS)
- [Gate MoE BF16 to FP8 conversion on compiler support](https://github.com/NVIDIA/TensorRT-LLM/pull/15934) (TensorRT-LLM)
- [Detect the MTP routed-expert layout separately](https://github.com/NVIDIA/TensorRT-LLM/pull/16276) (TensorRT-LLM)
- [Weight autoscaling queue time by request count](https://github.com/triton-inference-server/server/pull/8980) (Triton)

</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/sglang.png" alt="" width="64" height="64" /></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title"><a href="https://github.com/sgl-project/sglang">SGLang</a></p>
<p class="pb-tl-date">5 in review</p>

<p class="pb-patch-group">In review</p>

- [`trtllm_mla` rejects prefill on SM12x](https://github.com/sgl-project/sglang/pull/28837)
- [Supported MoE runner for ModelOpt NVFP4 on SM110/SM120](https://github.com/sgl-project/sglang/pull/45520)
- [Fuse ordinary FP8 KV stores](https://github.com/sgl-project/sglang/pull/31652)

</div>
</div>

</div>

!!! note "Why I look for these"
    The bugs I care about most do not crash. A cuDNN path that computed only the
    first expert group, a `moe_permute` write past the end of its buffer, an attention
    layout mismatch that corrupted logits on new hardware: each one passed its tests.
    I found them by reading the kernels, because there was no stack trace to follow.
