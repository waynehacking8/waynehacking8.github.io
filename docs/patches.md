# Patches

Upstream open-source contributions to the LLM-inference stack — **17 merged/landed ·
60 in review** as of 2026-07-18.

<div class="pb-pr-wall">
<a class="pb-pr-wall-main" href="https://prs.wayne.is-a.dev">
<span class="pb-pr-wall-status"><span class="pb-live-dot" aria-hidden="true"></span>Live PR wall</span>
<strong class="pb-pr-wall-domain">prs.wayne.is-a.dev</strong>
<span class="pb-pr-wall-counts">17 merged / landed · 60 in review</span>
<span class="pb-pr-wall-arrow" aria-hidden="true">↗</span>
</a>
<p class="pb-pr-wall-note">Synced every few minutes <span aria-hidden="true">·</span> <a class="pb-pr-wall-rss" href="https://prs.wayne.is-a.dev/feed.xml">RSS feed</a></p>
</div>

The tables below are a curated snapshot of representative work; the live wall is the
always-current source of truth.

My focus is **early consumer-Blackwell (SM120 / SM121) enablement** across the stack —
from CUTLASS / FlashInfer kernels through vLLM / SGLang engines to Dynamo /
TensorRT-LLM disaggregated serving — plus silent-correctness bugs: code that passes
tests but computes the wrong answer.

## Merged / landed in main

| Project | # | Representative merged work |
| --- | :-: | --- |
| **FlashInfer** | 7 | [SM120/121 multi-CTA radix top-k stream hang](https://github.com/flashinfer-ai/flashinfer/pull/3615) · [SM120 NVFP4 attention qk-correction layout / row-sum / lse](https://github.com/flashinfer-ai/flashinfer/pull/3838) · [MXFP8-aware MoE gemm profiler](https://github.com/flashinfer-ai/flashinfer/pull/3614) |
| **vLLM** | 3 | [Tokenizer survives pickling](https://github.com/vllm-project/vllm/pull/45460) · [streaming `message_start` sets `type`/`role`](https://github.com/vllm-project/vllm/pull/45376) · [clear error for structured outputs on diffusion decoders](https://github.com/vllm-project/vllm/pull/45468) |
| **LMDeploy** | 3 | [InternVL LoRA loading TypeError](https://github.com/InternLM/lmdeploy/pull/4684) · [double-counted `max_q_seqlen` in decode delta](https://github.com/InternLM/lmdeploy/pull/4685) · [builtin chat-template ImportError](https://github.com/InternLM/lmdeploy/pull/4690) |
| **PyTorch** | 1 | [`nccl.broadcast` was dropping its `root` argument](https://github.com/pytorch/pytorch/pull/187216) — landed in `main` (`c3c33fd`) |
| **Dynamo** | 1 | [KV-router cancels in-flight recovery on worker removal](https://github.com/ai-dynamo/dynamo/pull/10616) |
| **compressed-tensors** | 1 | [Skip device-map entries with no local module in dispatch](https://github.com/vllm-project/compressed-tensors/pull/737) |
| **torchao** | 1 | [PT2E/X86 plain-linear annotation fallback for reused `nn.Linear`](https://github.com/pytorch/ao/pull/4480) |

## In review

| Project | # | Representative work |
| --- | :-: | --- |
| **SGLang** | 17 | SM120/SM121 dispatch for `int8_scaled_mm` & `fp8_blockwise_scaled_grouped_mm` · SM120 shared-mem-safe attention block size |
| **vLLM** | 15 | NVFP4 MoE per-expert scale validation · FP8 MoE+LoRA routed to Marlin · async-KV-load scheduling fix |
| **FlashInfer** | 13 | NVFP4 global-scale threading through the unified MoE API · cuDNN full-sequence Q batch stride in batch prefill |
| **Dynamo** | 5 | Blackwell workstation GPU SKU support · KV-router hardening follow-ups |
| **NVIDIA TensorRT-LLM** | 4 | CuteDSL MoE ghost-token & global-index fixes · DeepSeek-V2-Lite `e_score_correction_bias` guard |
| **NVIDIA CUTLASS** | 4 | SM120 grouped NVFP4 block-scaled GEMM in `cutlass_library` · CuTeDSL sub-byte `make_ptr` / `is_major` / uint-lowering fixes |
| **torchao** | 1 | Reused-module fallback extended to `nn.Conv2d` (follow-up to merged #4480) |
| **LMCache** | 1 | Clear `reqs_status` on async-lookup timeout to prevent recall KeyError |

!!! note "The pattern behind these"
    The most valuable class of bug I hunt: **tests green, answers wrong** — a cuDNN path
    computing only the first expert group, an out-of-bounds `moe_permute` write, an
    attention layout mismatch that silently corrupts logits on new hardware. These come
    from reading kernels, not from stack traces.
