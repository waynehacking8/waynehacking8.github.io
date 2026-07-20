# Blog

Technical notes on model architecture, LLM systems, GPU programming, privacy, and security.

<div class="pb-filters" aria-label="Filter blog posts by tag">
<span class="pb-filters-label">Tags:</span>
<button type="button" data-tag="Data">Data</button>
<button type="button" data-tag="Attribution">Attribution</button>
<button type="button" data-tag="Algorithm">Algorithm</button>
<button type="button" data-tag="Theory">Theory</button>
<button type="button" data-tag="Survey">Survey</button>
<button type="button" data-tag="Security">Security</button>
<button type="button" data-tag="Library">Library</button>
</div>

<div class="pb-posts">
<a class="pb-post-card" href="inkling-975b-architecture/" data-tags="Algorithm,Theory,Data">
<span class="pb-post-image pb-post-cover pb-cover-inkling"><span class="pb-cover-mark">5:1</span><span class="pb-cover-label">Inkling 975B<br>Local ↔ Global</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Inkling 975B：不只把模型做大，而是重新分配每一分算力</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-green">MoE</span><span class="pb-tag pb-tag-blue">Multimodal</span></span>
<span class="pb-post-excerpt">5:1 滑動視窗、相對位置編碼、短卷積與 Muon/Adam：查核官方資料後拆解 Thinky 的第一個開放權重模型。</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="kimi-k3-architecture/" data-tags="Algorithm,Theory">
<span class="pb-post-image pb-post-cover pb-cover-kimi"><span class="pb-cover-mark">2.8T</span><span class="pb-cover-label">Kimi K3<br>KDA × AttnRes</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Kimi K3 2.8T：用 KDA 與 AttnRes 推進 3T 級開放模型</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-blue">Linear Attention</span><span class="pb-tag pb-tag-green">Agents</span></span>
<span class="pb-post-excerpt">KDA、Attention Residuals 與 896 選 16 的 Stable LatentMoE；並釐清 API 已上線、權重尚待釋出的差別。</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="notes-trtllm-triton-serving/" data-tags="Algorithm,Library">
<span class="pb-post-image pb-post-cover pb-cover-serving"><span class="pb-cover-mark">TRT</span><span class="pb-cover-label">TensorRT-LLM<br>× Triton</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on Serving LLMs with TensorRT-LLM and Triton</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">Serving</span><span class="pb-tag pb-tag-purple">NVIDIA stack</span></span>
<span class="pb-post-excerpt">From a Hugging Face checkpoint to a production TensorRT-LLM + Triton endpoint, benchmarked against vLLM.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="nccl-nvlink-bandwidth/" data-tags="Algorithm,Theory">
<span class="pb-post-image pb-post-cover pb-cover-nvlink"><span class="pb-cover-mark">↔</span><span class="pb-cover-label">NCCL / NVLink<br>Bandwidth Wall</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Where tensor-parallel inference hits the NVLink wall</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-blue">GPU</span><span class="pb-tag pb-tag-red">Distributed</span></span>
<span class="pb-post-excerpt">Measuring NCCL all-reduce and why tensor-parallel decode hits a small-message latency wall, not a bandwidth one.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="rag-groundedness-guardrail/" data-tags="Data,Attribution,Security">
<span class="pb-post-image pb-post-cover pb-cover-rag"><span class="pb-cover-mark">0%</span><span class="pb-cover-label">Grounded RAG<br>or Refuse</span></span>
<span class="pb-post-body">
<span class="pb-post-title">0% vs 40%: making a RAG agent refuse to hallucinate</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">LLM</span><span class="pb-tag pb-tag-green">RAG</span></span>
<span class="pb-post-excerpt">One guardrail that drops out-of-corpus hallucination from ~40% to 0%—and why an eval harness is the only way to see it.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-federated-learning-dp/" data-tags="Data,Theory,Security">
<span class="pb-post-image pb-post-cover pb-cover-fl"><span class="pb-cover-mark">ε</span><span class="pb-cover-label">Federated Learning<br>+ Differential Privacy</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on Federated Learning and Differential Privacy</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Privacy</span><span class="pb-tag pb-tag-blue">FL</span></span>
<span class="pb-post-excerpt">FedAvg, FedProx, and SCAFFOLD from scratch; what breaks under Non-IID data, and how DP-SGD adds real guarantees.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-cuda-tensor-core-gemm/" data-tags="Algorithm,Library">
<span class="pb-post-image pb-post-cover pb-cover-cuda"><span class="pb-cover-mark">∑</span><span class="pb-cover-label">CUDA Tensor Core<br>GEMM / WMMA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on CUDA Tensor Core GEMM (WMMA)</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Kernels</span></span>
<span class="pb-post-excerpt">Matrix multiply from naive to tiled to WMMA—shared-memory tiling, fragments, and the cuBLAS ceiling.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="meta-rl/" data-tags="Algorithm,Theory,Survey">
<span class="pb-post-image pb-post-cover pb-cover-rl"><span class="pb-cover-mark">π</span><span class="pb-cover-label">Meta-RL<br>Explore → Adapt</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Meta-Reinforcement Learning of Structured Exploration Strategies</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">ML</span><span class="pb-tag pb-tag-red">RL</span></span>
<span class="pb-post-excerpt">從「學會一個任務」走向「學會如何學習」：meta-RL 如何讓 agent 快速適應新環境。</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pp-intro/" data-tags="Algorithm,Library,Survey">
<span class="pb-post-image pb-post-cover pb-cover-parallel"><span class="pb-cover-mark">&lt;&lt;<span class="pb-cover-slash">/</span>&gt;&gt;</span><span class="pb-cover-label">CUDA C++<br>Parallel Programming</span></span>
<span class="pb-post-body">
<span class="pb-post-title">CUDA Programming 入門</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Parallel</span></span>
<span class="pb-post-excerpt">從執行緒層級、記憶體模型到核心函數，建立 CUDA 平行程式設計的完整基礎。</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pentest-intro/" data-tags="Security,Survey">
<span class="pb-post-image pb-post-cover pb-cover-security"><span class="pb-cover-mark">⌁</span><span class="pb-cover-label">Penetration Testing<br>Attack Surface</span></span>
<span class="pb-post-body">
<span class="pb-post-title">滲透測試入門技術</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Security</span></span>
<span class="pb-post-excerpt">從資訊蒐集、弱點掃描到驗證與報告，理解滲透測試的目標、流程與常用工具。</span>
<span class="pb-post-date">2025-01-10</span>
</span>
</a>
</div>
