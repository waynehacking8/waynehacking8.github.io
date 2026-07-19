# Projects

Reproducible, measured work — each repo states what it is, what it is *not*, credits
the upstream tools it drives, and ships measured results rather than claims. GPU work
runs on 4× H100 (NVLink) and Blackwell workstation hardware.

<div class="pb-filters">
<span class="pb-filters-label">Tags:</span>
<button data-tag="cuda">CUDA</button>
<button data-tag="serving">Serving</button>
<button data-tag="benchmarks">Benchmarks</button>
<button data-tag="agents">Agents</button>
<button data-tag="rag">RAG</button>
<button data-tag="privacy">Privacy</button>
</div>

<div class="pb-pub" data-tags="agents,rag,serving">
<div>
<p class="pb-pub-title">LocalMind-Core-GX10 — enterprise multi-agent AI platform</p>
<p class="pb-pub-meta"><b>Sole developer</b> · SYNCROBOTIC · Sep 2025 – Jun 2026 · Python / TypeScript / Rust / vLLM / LightRAG / NestJS / DGX</p>
<p>Polyglot architecture — a Rust performance core for dispatch and caching, a NestJS +
TypeScript API gateway, and Python ML services — with a unified provider router and an
agent layer over LightRAG hybrid retrieval (vector + keyword + knowledge graph). One
platform replaces ad-hoc per-team integrations, with built-in tracing, regression
testing, and safety validation. Shipped at two enterprise customers.</p>
<span class="pb-tag pb-tag-blue">Agents</span><span class="pb-tag pb-tag-green">RAG</span><span class="pb-tag pb-tag-purple">Serving</span>
</div>
</div>

<div class="pb-pub" data-tags="cuda,benchmarks,serving">
<div>
<p class="pb-pub-title">GPU inference benchmarks &amp; kernel studies</p>
<p class="pb-pub-meta">CUDA / Triton / PyTorch / Nsight · <em>in progress — code release planned</em></p>
<p>LLM serving internals, hands-on: KV-cache behaviour, quantization trade-offs, Flash
Attention; hand-written kernels (tiled GEMM, WMMA Tensor Core matmul) profiled with
Nsight Compute to connect kernel-level decisions to end-to-end serving latency.</p>
<span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-purple">Serving</span><span class="pb-tag pb-tag-blue">Benchmarks</span>
</div>
</div>

<div class="pb-pub" data-tags="serving,benchmarks">
<div>
<p class="pb-pub-title">TensorRT-LLM + Triton multi-GPU serving</p>
<p class="pb-pub-meta">TensorRT-LLM / Triton / vLLM / 4× H100 NVLink · <em>release planned once results land</em> · <a href="../blog/notes-trtllm-triton-serving/">notes</a></p>
<p>Reproducible build → serve → benchmark harness: TensorRT-LLM engines on Triton
(TP=4), measured head-to-head against vLLM under matched concurrency.</p>
<span class="pb-tag pb-tag-purple">Serving</span><span class="pb-tag pb-tag-blue">Benchmarks</span>
</div>
</div>

<div class="pb-pub" data-tags="cuda,benchmarks">
<div>
<p class="pb-pub-title">NCCL collectives benchmark</p>
<p class="pb-pub-meta">NCCL / nccl-tests / CUDA / 4× H100 NVSwitch · <a href="https://github.com/waynehacking8/nccl-collectives-bench">github</a> · <a href="../blog/nccl-nvlink-bandwidth/">notes</a></p>
<p>Bus-bandwidth micro-benchmarks for all-reduce / all-gather / reduce-scatter on
4× H100 NVLink, analysed against the theoretical link budget. Measured: all-reduce
<b>366 GB/s (77% of NVLink)</b>; NVLS &gt; Ring &gt; Tree; protocol study (Simple/LL128/LL).</p>
<span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Benchmarks</span>
</div>
</div>

<div class="pb-pub" data-tags="agents,rag,benchmarks">
<div>
<p class="pb-pub-title">NIM agent blueprint</p>
<p class="pb-pub-meta">NVIDIA NIM / RAG / agents / OpenTelemetry / FastAPI · <a href="https://github.com/waynehacking8/nim-agent-blueprint">github</a> · <a href="../blog/rag-groundedness-guardrail/">notes</a></p>
<p>Agentic RAG reference architecture on NVIDIA NIM microservices (LLM + embedding +
reranker) with a plan → retrieve → generate → validate loop and a built-in eval
harness. Measured: retrieval <b>recall@3 100%</b>, and <b>0% hallucination</b> on
out-of-corpus questions with a guarded prompt vs <b>~40%</b> without it (ablation).</p>
<span class="pb-tag pb-tag-blue">Agents</span><span class="pb-tag pb-tag-green">RAG</span>
</div>
</div>

<div class="pb-pub" data-tags="cuda,benchmarks">
<div>
<p class="pb-pub-title">Blackwell Tensor Core kernels</p>
<p class="pb-pub-meta">CUDA / Tensor Cores / WMMA / Nsight / H100 + Blackwell workstation · <em>release planned once results land</em> · <a href="../blog/notes-cuda-tensor-core-gemm/">notes</a></p>
<p>Hand-written CUDA GEMM kernels (naive → tiled → WMMA Tensor Core), benchmarked
across Hopper (sm_90) and Blackwell (sm_120) as a fraction of the cuBLAS ceiling.</p>
<span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Benchmarks</span>
</div>
</div>

<div class="pb-pub" data-tags="privacy">
<div>
<p class="pb-pub-title">Federated learning lab</p>
<p class="pb-pub-meta">Python / PyTorch / FL / DP · <a href="https://github.com/waynehacking8/federated-learning-lab">github</a> · <a href="../blog/notes-federated-learning-dp/">notes</a></p>
<p>From-scratch federated learning — FedAvg / FedProx / SCAFFOLD, DP-SGD and secure
aggregation, plus FedPer / Byzantine-robust / FedAdam / FedLoRA. 33/33 tests,
literature-cross-validated, with honest negative results on Non-IID MNIST.</p>
<span class="pb-tag pb-tag-red">Privacy</span><span class="pb-tag pb-tag-purple">Federated Learning</span>
</div>
</div>
