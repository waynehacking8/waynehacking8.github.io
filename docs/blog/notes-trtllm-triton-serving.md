---
description: "Taking an LLM from a Hugging Face checkpoint to a production TensorRT-LLM + Triton endpoint on 4x H100, benchmarked against vLLM — including the measured low/mid vs high concurrency crossover."
---

# Notes on Serving LLMs with TensorRT-LLM and Triton

*2026-05-31 · LLM serving / NVIDIA stack*

<figure class="pb-article-hero">
  <img src="/assets/blog/trtllm-triton.webp" alt="NVIDIA TensorRT-LLM 與 Triton 推論服務架構視覺" loading="eager" decoding="async">
  <figcaption>TensorRT-LLM production serving · <a href="https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/">Source: NVIDIA Developer</a></figcaption>
</figure>

These are working notes on taking an open-weights LLM from a Hugging Face checkpoint to a
production-style serving endpoint on the NVIDIA stack — **TensorRT-LLM** for the engine,
**Triton Inference Server** for the deployment surface — and benchmarking it honestly against
**vLLM** on multi-GPU hardware. They follow the harness in
[**trtllm-triton-serving**](https://github.com/waynehacking8/trtllm-triton-serving)
(4× H100, NVLink).

The goal is to move from "I use vLLM" to "I can stand up the NVIDIA inference stack on real
multi-GPU hardware and reason about the trade-offs."

## 1. The serving pipeline

The path from checkpoint to endpoint has four stages. Each one is a place where a decision
affects latency, throughput, or accuracy:

1. **Checkpoint** — a Hugging Face model.
2. **Engine build** — compile to a TensorRT-LLM engine for a *fixed* tensor-parallel degree,
   precision, and batching policy.
3. **Model repository** — wrap the engine in a Triton `tensorrt_llm`-backend model repo.
4. **Serving + load test** — `trtllm-serve` (or Triton) exposes an OpenAI-compatible endpoint;
   a load generator drives it under controlled concurrency.

The key mental shift from vLLM: TensorRT-LLM does **ahead-of-time compilation**. vLLM is a
runtime that takes the model and serves it; TensorRT-LLM *builds an engine* specialized to your
GPU, TP degree, and precision first. That build is where the performance comes from, and also
where the rigidity comes from.

## 2. Tensor parallelism (TP)

For a model that doesn't fit on one GPU — or to cut latency — TensorRT-LLM shards each layer
across GPUs. On a 4× H100 NVLink box, `TP=4` means every forward pass does an **all-reduce**
across the four GPUs over NVLink.

> The all-reduce is not free. On this fabric it tops out around 77 % of the NVLink budget
> (see the separate [NVLink-wall notes](nccl-nvlink-bandwidth.md)). For **prefill** (large
> tensors) you're bandwidth-bound and TP helps. For **decode** (one token at a time) you're
> pinned against the small-message latency floor, and past a point more TP makes decode
> *slower*. Pick TP for the regime you actually serve.

## 3. Precision: FP16 vs FP8

The engine is built for a specific precision. The two that matter most on Hopper:

| Precision | Memory | Throughput | Accuracy risk |
| --- | --- | --- | --- |
| **FP16** | baseline | baseline | none (reference) |
| **FP8** | ~½ weights + KV-cache | higher | small, model-dependent |

FP8 uses the Hopper Transformer Engine and shrinks both weights and the KV-cache, which is
often the real bottleneck for long contexts. The honest move is to **measure** the accuracy
delta on your task rather than assume FP8 is free — a quantization study belongs in the same
harness as the throughput numbers.

## 4. The batching policy that actually matters

Two features dominate real serving throughput:

- **In-flight (continuous) batching** — new requests join the running batch at the next
  iteration instead of waiting for the current batch to drain. This is what keeps GPUs busy
  under bursty traffic; vLLM and TensorRT-LLM both do it.
- **Paged KV-cache** — the KV-cache is allocated in pages, so memory isn't reserved for the
  worst-case sequence length per request. This is what lets you fit more concurrent sequences.

If a "benchmark" doesn't enable these, it isn't measuring production serving — it's measuring a
toy.

## 5. The benchmark trap: comparing the same work

The single most common mistake in "X vs Y" LLM benchmarks is **not decoding the same number of
tokens**. If stack A happens to emit shorter completions, it looks faster while doing less work.

The fix used in the harness is a **controlled methodology**: every request decodes *exactly*
256 tokens by setting `ignore_eos=True` and `min_tokens=max_tokens`. Now throughput and
latency compare identical work across TensorRT-LLM, Triton, and vLLM. Without this, the numbers
are noise.

Metrics worth reporting, all under *matched concurrency*:

- **Throughput** (tokens/s, total) — the headline.
- **TTFT** (time to first token) — dominated by prefill; what the user feels first.
- **Inter-token latency** — dominated by decode; what the user feels while reading.

## 6. Triton as the production surface

The measured runs can use TensorRT-LLM's own OpenAI server (`trtllm-serve`), but the
**production path** is the Triton `tensorrt_llm`-backend model repository (`triton_model_repo/`):

- It exposes the engine over a hardened, observable server (metrics, health, dynamic batching
  config) instead of a script.
- It's the same control plane you'd use for an ensemble (tokenizer → engine → de-tokenizer) and
  for multi-model hosting.

Treat `trtllm-serve` as the fast path for benchmarking and Triton as the path you'd actually
ship behind a gateway.

## 7. When does TensorRT-LLM win? (the measured answer)

Not always — and the measurement says *which* regime, not a vibe. Across a matched-work sweep on
4× H100, the result lands on a **concurrency crossover**:

- **TensorRT-LLM (with CUDA graphs) wins at low-to-mid concurrency** — the latency-sensitive
  regime. The ahead-of-time engine plus CUDA-graph capture removes per-iteration launch overhead
  that dominates when the batch is small, so TTFT and inter-token latency are lower.
- **vLLM wins at high concurrency** — the throughput-saturated regime, where its scheduler keeps
  the GPU packed and the launch-overhead advantage no longer matters.

> One caveat that cost a real bug: **CUDA graphs only help if the config actually enables them.**
> A run that looks like "TensorRT-LLM is barely faster" can be a mis-set graph config; fixing it
> moved the low-concurrency number substantially. Always confirm the optimisation you're
> crediting is switched on before drawing the curve.

So the decision rule is about your **load**, not brand loyalty: latency-bound, low/mid
concurrency → TensorRT-LLM + CUDA graphs; throughput-bound, high concurrency → vLLM. The honest
deliverable is a reproducible **serve → benchmark** loop with documented methodology that draws
that crossover for *your* hardware.

## Takeaway

Serving an LLM well is mostly about three things: putting tensor parallelism in the regime that
helps, enabling continuous batching + paged KV-cache, and **measuring the same work** across
stacks. The measured crossover: TensorRT-LLM + CUDA graphs win **low/mid concurrency** (latency),
vLLM wins **high concurrency** (throughput) — and a Triton control plane is what you'd actually
put in production.

→ Full pipeline, Triton model repo, and the matched-work harness:
[github.com/waynehacking8/trtllm-triton-serving](https://github.com/waynehacking8/trtllm-triton-serving)
