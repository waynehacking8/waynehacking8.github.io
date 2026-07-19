# Blog

Technical notes on machine learning, GPU programming, and security.

<div class="pb-posts">
<a class="pb-post-card" href="notes-trtllm-triton-serving/">
<p class="pb-post-title">Notes on Serving LLMs with TensorRT-LLM and Triton</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-green">Serving</span><span class="pb-tag pb-tag-purple">NVIDIA stack</span></p>
<p class="pb-post-excerpt">From a Hugging Face checkpoint to a production TensorRT-LLM + Triton endpoint, benchmarked against vLLM.</p>
<p class="pb-post-date">2026-05-31</p>
</a>
<a class="pb-post-card" href="nccl-nvlink-bandwidth/">
<p class="pb-post-title">Where tensor-parallel inference hits the NVLink wall</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-blue">GPU</span><span class="pb-tag pb-tag-red">Distributed</span></p>
<p class="pb-post-excerpt">Measuring NCCL all-reduce on NVLink and why tensor-parallel decode hits a small-message latency wall, not a bandwidth one.</p>
<p class="pb-post-date">2026-05-31</p>
</a>
<a class="pb-post-card" href="rag-groundedness-guardrail/">
<p class="pb-post-title">0&#8202;% vs 40&#8202;%: making a RAG agent refuse to hallucinate</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-purple">LLM</span><span class="pb-tag pb-tag-green">RAG</span></p>
<p class="pb-post-excerpt">One guardrail that drops out-of-corpus hallucination from ~40% to 0% — and why an eval harness is the only way to see it.</p>
<p class="pb-post-date">2026-05-31</p>
</a>
<a class="pb-post-card" href="notes-federated-learning-dp/">
<p class="pb-post-title">Notes on Federated Learning and Differential Privacy</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-red">Privacy</span><span class="pb-tag pb-tag-blue">FL</span></p>
<p class="pb-post-excerpt">FedAvg / FedProx / SCAFFOLD from scratch, what breaks under Non-IID data, and how DP-SGD adds real guarantees.</p>
<p class="pb-post-date">2026-05-31</p>
</a>
<a class="pb-post-card" href="notes-cuda-tensor-core-gemm/">
<p class="pb-post-title">Notes on CUDA Tensor Core GEMM (WMMA)</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Kernels</span></p>
<p class="pb-post-excerpt">Matrix multiply from naive to tiled to WMMA — shared-memory tiling, fragments, and the cuBLAS ceiling.</p>
<p class="pb-post-date">2026-05-31</p>
</a>
<a class="pb-post-card" href="meta-rl/">
<p class="pb-post-title">Meta-Reinforcement Learning of Structured Exploration Strategies</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-purple">ML</span><span class="pb-tag pb-tag-red">RL</span></p>
<p class="pb-post-excerpt">「學習如何學習」— how meta-RL lets an agent adapt quickly to new environments and tasks.</p>
<p class="pb-post-date">2025-01-11</p>
</a>
<a class="pb-post-card" href="pp-intro/">
<p class="pb-post-title">CUDA Programming 入門</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Parallel</span></p>
<p class="pb-post-excerpt">NVIDIA 平行運算平台與程式設計模型入門:threads、blocks、記憶體階層。</p>
<p class="pb-post-date">2025-01-11</p>
</a>
<a class="pb-post-card" href="pentest-intro/">
<p class="pb-post-title">滲透測試入門技術</p>
<p class="pb-post-tags"><span class="pb-tag pb-tag-red">Security</span></p>
<p class="pb-post-excerpt">滲透測試的定義、流程、類型與工具 — 模擬駭客攻擊找出可修補的漏洞。</p>
<p class="pb-post-date">2025-01-10</p>
</a>
</div>
