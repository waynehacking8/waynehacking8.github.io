# Projects

## Applied AI Systems

### LocalMind-Core-GX10
Enterprise multi-agent AI platform on DGX-class hardware. Sole developer.
A polyglot architecture — a Rust performance core for dispatch and caching, a NestJS +
TypeScript API gateway, and Python ML services — with a unified provider router and an
agent layer over LightRAG hybrid retrieval (vector + keyword + knowledge graph). One
platform replaces ad-hoc per-team integrations, with built-in tracing, regression
testing, and safety validation.
*Python · TypeScript · Rust · vLLM · LightRAG · NestJS · DGX*
*(SYNCROBOTIC, Sep 2025 – Present)*

## Research / Privacy-Preserving ML

### Federated Learning Lab
From-scratch federated learning — FedAvg / FedProx / SCAFFOLD, DP-SGD and secure
aggregation, plus FedPer / Byzantine-robust / FedAdam / FedLoRA. 33/33 tests,
literature-cross-validated, with honest negative results on Non-IID MNIST.
*Python · PyTorch · Federated Learning · Differential Privacy*
:material-github: [waynehacking8/federated-learning-lab](https://github.com/waynehacking8/federated-learning-lab)
:material-notebook-outline: [Notes: Federated Learning & Differential Privacy](blog/notes-federated-learning-dp.md)

## NVIDIA Inference & Multi-GPU Stack

> A portfolio of reproducible harnesses on the NVIDIA-native stack, run on 4× H100
> (NVLink) and an RTX Pro 6000 (Blackwell). Each repo states what it is, what it is NOT,
> credits the upstream tools it drives, and ships measured results rather than claims.

### GPU Inference Benchmarks & Kernel Studies
LLM serving internals, hands-on: KV-cache behaviour, quantization trade-offs, Flash
Attention; hand-written kernels (tiled GEMM, WMMA Tensor Core matmul) profiled with
Nsight Compute to connect kernel-level decisions to end-to-end serving latency.
*CUDA · Triton · PyTorch · Nsight* *(in progress — code release planned)*

### TensorRT-LLM + Triton Multi-GPU Serving
Reproducible build → serve → benchmark harness: TensorRT-LLM engines on Triton (TP=4),
measured head-to-head against vLLM under matched concurrency.
*TensorRT-LLM · Triton · vLLM · 4× H100 NVLink* *(release planned once results land)*
:material-notebook-outline: [Notes: serving on TensorRT-LLM + Triton](blog/notes-trtllm-triton-serving.md)

### NCCL Collectives Benchmark
Bus-bandwidth micro-benchmarks for all-reduce / all-gather / reduce-scatter on 4× H100
NVLink, analysed against the theoretical link budget. Measured: all-reduce **366 GB/s
(77% of NVLink)**; NVLS (NVLink SHARP) > Ring > Tree; protocol study (Simple/LL128/LL).
*NCCL · nccl-tests · CUDA · 4× H100 NVSwitch*
:material-github: [waynehacking8/nccl-collectives-bench](https://github.com/waynehacking8/nccl-collectives-bench)
:material-notebook-outline: [Notes: where TP inference hits the NVLink wall](blog/nccl-nvlink-bandwidth.md)

### NIM Agent Blueprint
Agentic RAG reference architecture on NVIDIA NIM microservices (LLM + embedding +
reranker) with a plan → retrieve → generate → validate loop and a built-in eval harness.
Measured: retrieval **recall@3 100%**, and **0% hallucination** on out-of-corpus questions
with a guarded prompt vs **~40%** without it (ablation).
*NVIDIA NIM · RAG · agents · OpenTelemetry · FastAPI*
:material-github: [waynehacking8/nim-agent-blueprint](https://github.com/waynehacking8/nim-agent-blueprint)
:material-notebook-outline: [Notes: 0% vs 40% hallucination](blog/rag-groundedness-guardrail.md)

### Blackwell Tensor Core Kernels
Hand-written CUDA GEMM kernels (naive → tiled → WMMA Tensor Core), benchmarked across
Hopper (sm_90) and Blackwell (sm_120) as a fraction of the cuBLAS ceiling.
*CUDA · Tensor Cores · WMMA · Nsight · H100 / RTX Pro 6000* *(release planned once results land)*
:material-notebook-outline: [Notes: CUDA Tensor Core GEMM (WMMA)](blog/notes-cuda-tensor-core-gemm.md)
