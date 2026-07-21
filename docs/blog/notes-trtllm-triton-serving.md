---
description: "TensorRT-LLM vs vLLM on 4x H100, with paragraph-by-paragraph Traditional Chinese and the measured low/mid- versus high-concurrency crossover."
date: "2026-05-31"
updated: "2026-07-21"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/trtllm-triton.webp"
tags:
  - Serving
  - TensorRT-LLM
  - Triton
---

# TensorRT-LLM vs vLLM on 4×H100: The Concurrency Crossover｜4×H100 上的 TensorRT-LLM 與 vLLM：Concurrency crossover

*2026-05-31 · updated 2026-07-21 · LLM serving / NVIDIA stack*

<figure class="pb-article-hero">
  <img src="/assets/blog/trtllm-triton.webp" alt="NVIDIA TensorRT-LLM 與 Triton 推論服務架構視覺" loading="eager" decoding="async">
  <figcaption>TensorRT-LLM production serving · TensorRT-LLM production serving 架構 · <a href="https://developer.nvidia.com/blog/scaling-llms-with-nvidia-triton-and-nvidia-tensorrt-llm-using-kubernetes/">Source: NVIDIA Developer</a></figcaption>
</figure>

The [**trtllm-triton-serving**](https://github.com/waynehacking8/trtllm-triton-serving) harness takes an open-weights Hugging Face checkpoint through a TensorRT-LLM engine build and exposes it with `trtllm-serve` or Triton. The matched-work benchmark compares that stack with vLLM on 4× H100 connected by NVLink.[^serving-repo]

[**trtllm-triton-serving**](https://github.com/waynehacking8/trtllm-triton-serving) harness 會把 Hugging Face 的開放權重 checkpoint 建成 TensorRT-LLM engine，再透過 `trtllm-serve` 或 Triton 對外服務。Matched-work benchmark 則在 NVLink 連接的 4× H100 上，拿這套 stack 和 vLLM 比較。[^serving-repo]
{ .pb-translation lang=zh-Hant }

## 1. The serving pipeline｜Serving pipeline

The path from checkpoint to endpoint has four stages. A choice at any stage can change latency, throughput, or accuracy:

Checkpoint 到 endpoint 分成四個階段；任何一階段的設定都可能改變 latency、throughput 或 accuracy：
{ .pb-translation lang=zh-Hant }

1. **Checkpoint**: a Hugging Face model.<br><span class="pb-inline-translation" lang="zh-Hant">**Checkpoint**：Hugging Face model。</span>
2. **Engine build**: compile a TensorRT-LLM engine for a selected tensor-parallel degree, precision, and batching policy.<br><span class="pb-inline-translation" lang="zh-Hant">**Engine build**：依指定的 tensor-parallel degree、precision 與 batching policy 編譯 TensorRT-LLM engine。</span>
3. **Model repository**: wrap the engine in a Triton `tensorrt_llm` backend repository.<br><span class="pb-inline-translation" lang="zh-Hant">**Model repository**：把 engine 包進 Triton `tensorrt_llm` backend repository。</span>
4. **Serving and load test**: expose an OpenAI-compatible endpoint and drive it with controlled concurrency.<br><span class="pb-inline-translation" lang="zh-Hant">**Serving 與 load test**：提供 OpenAI-compatible endpoint，再用受控 concurrency 施加負載。</span>

TensorRT-LLM builds an engine for a selected GPU architecture, TP degree, precision, and batching configuration. vLLM makes more of those decisions at runtime, so the two stacks trade build-time specialization for runtime flexibility.[^trt-docs]

TensorRT-LLM 會針對指定的 GPU architecture、TP degree、precision 與 batching configuration 建立 engine。vLLM 則在 runtime 做更多決策；兩套 stack 的差別，是 build-time specialization 與 runtime flexibility 的取捨。[^trt-docs]
{ .pb-translation lang=zh-Hant }

## 2. Tensor parallelism (TP)｜Tensor parallelism（TP）

When a model does not fit on one GPU—or when the target is lower latency—TensorRT-LLM can shard each layer across GPUs. On a 4× H100 NVLink host, `TP=4` introduces all-reduce communication across all four GPUs during each forward pass.

模型放不進單張 GPU，或目標是降低 latency 時，TensorRT-LLM 可以把每一層切到多張 GPU。4× H100 NVLink 主機使用 `TP=4` 時，每次 forward pass 都會在四張 GPU 間執行 all-reduce。
{ .pb-translation lang=zh-Hant }

> On this fabric, all-reduce reaches about 77% of the NVLink budget (see the separate [NVLink-wall notes](nccl-nvlink-bandwidth.md)). Prefill uses large tensors and benefits from bandwidth. Decode communicates one token at a time and can hit the small-message latency floor; adding TP beyond that point can make decode slower.
>
> <span class="pb-inline-translation" lang="zh-Hant">在這組 fabric 上，all-reduce 約達 NVLink budget 的 77%（見另一篇 [NVLink-wall 筆記](nccl-nvlink-bandwidth.md)）。Prefill 使用大 tensor，能受益於頻寬；decode 一次只處理一個 token，可能撞上小訊息 latency 下限。超過該點後再增加 TP，反而可能讓 decode 變慢。</span>

Choose TP for the message sizes and latency regime of the deployed workload.

TP 應依部署 workload 的 message sizes 與 latency regime 選擇。
{ .pb-translation lang=zh-Hant }

## 3. Precision: FP16 vs FP8｜Precision：FP16 與 FP8

This benchmark compares FP16 and FP8 on Hopper:

這份 benchmark 比較 Hopper 上的 FP16 與 FP8：
{ .pb-translation lang=zh-Hant }

| Precision | Memory／記憶體 | Throughput／吞吐量 | Accuracy risk／準確度風險 |
| --- | --- | --- | --- |
| **FP16** | baseline／基準 | baseline／基準 | reference／參考值 |
| **FP8** | ~½ weights + KV cache／約一半的 weights 與 KV cache | higher／較高 | small, model-dependent／幅度小但依模型而異 |

FP8 uses Hopper Transformer Engine and reduces weight and KV-cache storage. The accuracy delta must be measured on the target workload, under the same harness as the throughput result.

FP8 使用 Hopper Transformer Engine，並減少 weight 與 KV-cache storage。Accuracy delta 必須在目標 workload 上量測，而且要和 throughput 使用同一套 harness。
{ .pb-translation lang=zh-Hant }

## 4. Continuous batching and paged KV cache｜Continuous batching 與 paged KV cache

Two features shape the production configurations compared here:

本文比較的 production configurations 都包含兩項功能：
{ .pb-translation lang=zh-Hant }

- **In-flight (continuous) batching** lets new requests join a running batch at the next iteration instead of waiting for it to drain. Both vLLM and TensorRT-LLM support it.<br><span class="pb-inline-translation" lang="zh-Hant">**In-flight（continuous）batching** 允許新 request 在下一個 iteration 加入目前 batch，不必等整批清空；vLLM 與 TensorRT-LLM 都支援。</span>
- **Paged KV cache** allocates cache in pages rather than reserving the worst-case sequence length for each request, allowing more concurrent sequences to fit.<br><span class="pb-inline-translation" lang="zh-Hant">**Paged KV cache** 以 pages 分配 cache，不會為每個 request 預留 worst-case sequence length，因此能容納更多 concurrent sequences。</span>

Results without these features describe static batching rather than the production configurations measured here.

未啟用這兩項功能的結果描述的是 static batching，不能和本文的 production configurations 直接比較。
{ .pb-translation lang=zh-Hant }

## 5. Match the generated work｜讓兩套 stack 生成相同工作量

Completion length is a confounder: a stack that emits fewer tokens can appear faster while doing less work. The harness therefore forces every request to decode exactly 256 tokens with `ignore_eos=True` and matched minimum and maximum token limits.

Completion length 是一個 confounder：某套 stack 產生較少 tokens 時，即使只是少做工作，看起來也會比較快。因此，harness 使用 `ignore_eos=True` 和一致的最小／最大 token limits，強制每個 request 都 decode 256 tokens。
{ .pb-translation lang=zh-Hant }

Report all metrics at matched concurrency:

所有指標都必須在相同 concurrency 下回報：
{ .pb-translation lang=zh-Hant }

- **Throughput** in total tokens/s.<br><span class="pb-inline-translation" lang="zh-Hant">**Throughput**：總 tokens/s。</span>
- **TTFT (time to first token)**, dominated by prefill.<br><span class="pb-inline-translation" lang="zh-Hant">**TTFT（time to first token）**：主要受 prefill 影響。</span>
- **Inter-token latency**, dominated by decode.<br><span class="pb-inline-translation" lang="zh-Hant">**Inter-token latency**：主要受 decode 影響。</span>

## 6. Triton as the deployment surface｜以 Triton 作為部署介面

The benchmark uses TensorRT-LLM’s OpenAI server, `trtllm-serve`. The repository also includes a Triton `tensorrt_llm` backend model repository for deployments that need Triton metrics, health checks, ensembles, or multi-model hosting.[^triton-docs]

Benchmark 使用 TensorRT-LLM 的 OpenAI server，也就是 `trtllm-serve`。Repository 另附 Triton `tensorrt_llm` backend model repository，供需要 Triton metrics、health checks、ensembles 或 multi-model hosting 的部署使用。[^triton-docs]
{ .pb-translation lang=zh-Hant }

The Triton path can place tokenizer, engine, and detokenizer stages in one ensemble and expose them through a shared control plane.

Triton path 可以把 tokenizer、engine 與 detokenizer 放進同一個 ensemble，再透過共用 control plane 對外服務。
{ .pb-translation lang=zh-Hant }

## 7. Measured crossover on 4×H100｜4×H100 上的實測 crossover

On the matched-work sweep, TensorRT-LLM led at low-to-mid concurrency and vLLM led at high concurrency:

Matched-work sweep 顯示 TensorRT-LLM 在 low-to-mid concurrency 領先，vLLM 則在 high concurrency 領先：
{ .pb-translation lang=zh-Hant }

- **TensorRT-LLM with CUDA Graphs leads at low-to-mid concurrency.** Ahead-of-time specialization and graph capture reduce per-iteration launch overhead when batches are small, lowering TTFT and inter-token latency.<br><span class="pb-inline-translation" lang="zh-Hant">**TensorRT-LLM 加 CUDA Graphs 在 low-to-mid concurrency 領先。** Batch 較小時，ahead-of-time specialization 與 graph capture 會降低每個 iteration 的 launch overhead，進而降低 TTFT 與 inter-token latency。</span>
- **vLLM leads at high concurrency.** In the throughput-saturated regime, its scheduler keeps the GPU packed and the launch-overhead advantage no longer determines the result.<br><span class="pb-inline-translation" lang="zh-Hant">**vLLM 在 high concurrency 領先。** 進入 throughput-saturated regime 後，它的 scheduler 能維持 GPU 飽和，launch-overhead 優勢也不再主導結果。</span>

> One configuration bug changed this curve: CUDA Graphs only help when the graph setting is actually enabled. A misconfigured run made TensorRT-LLM look only slightly faster; fixing the setting moved the low-concurrency result substantially.
>
> <span class="pb-inline-translation" lang="zh-Hant">一個 configuration bug 曾經改變整條曲線：只有真正啟用 graph setting 時，CUDA Graphs 才有效。設定錯誤的 run 讓 TensorRT-LLM 看起來只快一點；修正後，low-concurrency 結果明顯移動。</span>

Choose from the measured load regime: TensorRT-LLM with CUDA Graphs for the latency-sensitive low/mid-concurrency range, and vLLM for the tested high-concurrency throughput range. Rerun the sweep on the target hardware before carrying over the crossover.

選擇要依實測 load regime：latency-sensitive 的 low／mid-concurrency 範圍使用 TensorRT-LLM 加 CUDA Graphs；本文測試的 high-concurrency throughput 範圍則由 vLLM 領先。把 crossover 套到其他系統前，應先在目標硬體上重跑 sweep。
{ .pb-translation lang=zh-Hant }

→ Full pipeline, Triton model repository, and matched-work harness:<br><span class="pb-inline-translation" lang="zh-Hant">完整 pipeline、Triton model repository 與 matched-work harness：</span>
[github.com/waynehacking8/trtllm-triton-serving](https://github.com/waynehacking8/trtllm-triton-serving)

[^serving-repo]: [TensorRT-LLM + Triton serving benchmark](https://github.com/waynehacking8/trtllm-triton-serving), the primary artifact for engine configs, the matched-work harness, and the measured crossover.／Engine configs、matched-work harness 與實測 crossover 的主要來源。
[^trt-docs]: [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/), NVIDIA’s engine-build and runtime reference.／NVIDIA 的 engine-build 與 runtime 文件。
[^triton-docs]: [Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/), NVIDIA’s model-repository, scheduling, metrics, and protocol reference.／NVIDIA 的 model repository、scheduling、metrics 與 protocol 文件。
