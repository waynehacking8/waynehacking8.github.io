---
seo_title: Patches | Wayne Chiu
description: Upstream fixes in FlashInfer, vLLM, PyTorch, Dynamo and more.
---

# Patches

Upstream open-source contributions to the LLM-inference stack: **24 merged and
26 in review** as of 2026-11-05.

<p class="pb-pr-wall">
<strong>Every pull request and review, in full, is on the <a class="pb-pr-wall-main" href="https://prs.wayne.is-a.dev">PR wall</a>.</strong>
It rebuilds from GitHub every 10 minutes and is also an <a class="pb-pr-wall-rss" href="https://prs.wayne.is-a.dev/feed.xml">RSS feed</a>.
</p>

The tables below are a curated snapshot of representative work.

My focus is **early consumer-Blackwell (SM120 / SM121) enablement** across the stack,
from CUTLASS / FlashInfer kernels through vLLM / SGLang engines to Dynamo /
TensorRT-LLM disaggregated serving, plus silent-correctness bugs: code that passes
tests but computes the wrong answer.

## Merged / landed in main

| Project | # | Representative merged work |
| --- | :-: | --- |
| **FlashInfer** | 10 | [SM120/121 multi-CTA radix top-k stream hang](https://github.com/flashinfer-ai/flashinfer/pull/3615); [SM120 NVFP4 attention qk-correction layout, row-sum and lse](https://github.com/flashinfer-ai/flashinfer/pull/3838); [b12x MoE workspace bucketing to stop kernel-cache thrash](https://github.com/flashinfer-ai/flashinfer/pull/3836) |
| **vLLM** | 5 | [Skip Qwen3 deepstack buffers without vision](https://github.com/vllm-project/vllm/pull/49397); [Tokenizer survives pickling](https://github.com/vllm-project/vllm/pull/45460); [streaming `message_start` sets `type`/`role`](https://github.com/vllm-project/vllm/pull/45376); [clear error for structured outputs on diffusion decoders](https://github.com/vllm-project/vllm/pull/45468) |
| **LMDeploy** | 3 | [InternVL LoRA loading TypeError](https://github.com/InternLM/lmdeploy/pull/4684); [double-counted `max_q_seqlen` in decode delta](https://github.com/InternLM/lmdeploy/pull/4685); [builtin chat-template ImportError](https://github.com/InternLM/lmdeploy/pull/4690) |
| **PyTorch** | 1 | [`nccl.broadcast` was dropping its `root` argument](https://github.com/pytorch/pytorch/pull/187216), landed in `main` (`c3c33fd`) |
| **Dynamo** | 2 | [KV-router cancels in-flight recovery on worker removal](https://github.com/ai-dynamo/dynamo/pull/10616); [resolve aggregate planner workers by DGD component type](https://github.com/ai-dynamo/dynamo/pull/11578) |
| **llm-compressor** | 1 | [Research Hub entry: FP8 KV cache on Phi-3's fused QKV](https://github.com/vllm-project/llm-compressor/pull/3344) |
| **compressed-tensors** | 1 | [Skip device-map entries with no local module in dispatch](https://github.com/vllm-project/compressed-tensors/pull/737) |
| **torchao** | 1 | [PT2E/X86 plain-linear annotation fallback for reused `nn.Linear`](https://github.com/pytorch/ao/pull/4480) |

## In review

| Project | # | Representative work |
| --- | :-: | --- |
| **vLLM** | 7 | NVFP4 MoE per-expert scale validation; CUTLASS block FP8 E8M0 upcast on SM12x; stop re-queuing waiting requests that `max_loras` blocks |
| **SGLang** | 5 | `trtllm_mla` rejects prefill on SM12x; supported MoE runner for ModelOpt NVFP4 on SM110/SM120; fused FP8 KV stores |
| **Dynamo** | 4 | Blackwell workstation GPU SKU in the operator; OTLP traces without logs; NIXL push and pull connectors in PD |
| **FlashInfer** | 3 | JIT ninja parallelism capped by available memory; quoted `-ccbin` host compiler path; combined paged-KV trace fix |
| **NVIDIA CUTLASS** | 3 | SM120 grouped NVFP4 block-scaled GEMM in `cutlass_library`; CuTeDSL sub-byte `make_ptr` and `is_major` fixes |
| **NVIDIA TensorRT-LLM** | 2 | MoE BF16 to FP8 conversion gated on compiler support; MTP routed-expert layout detection |
| **torchao** | 1 | Reused-module fallback extended to `nn.Conv2d` (follow-up to merged #4480) |
| **Triton Inference Server** | 1 | Weight autoscaling queue time by request count |

!!! note "The pattern behind these"
    The most valuable class of bug I hunt: **tests green, answers wrong**: a cuDNN path
    computing only the first expert group, an out-of-bounds `moe_permute` write, an
    attention layout mismatch that silently corrupts logits on new hardware. These come
    from reading kernels, not from stack traces.
