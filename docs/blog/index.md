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
<span class="pb-post-image"><img src="/assets/blog/inkling-cover.png" alt="Inkling 官方發布視覺" width="1200" height="630" loading="eager" fetchpriority="high" decoding="async"><span class="pb-image-credit">Thinking Machines</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Inkling 975B：不只把模型做大，而是重新分配每一分算力</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-green">MoE</span><span class="pb-tag pb-tag-blue">Multimodal</span></span>
<span class="pb-post-excerpt">5:1 滑動視窗、相對位置編碼、短卷積與 Muon/Adam：查核官方資料後拆解 Thinky 的第一個開放權重模型。</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="kimi-k3-architecture/" data-tags="Algorithm,Theory">
<span class="pb-post-image"><img src="https://kimi-file.moonshot.cn/prod-chat-kimi/kfs/4/2/2026-07-17/d9cs7176rtp4tqfofnsg?x-tos-process=image%2Fauto-orient%2C1%2Fstrip%2Fignore-error%2C1" alt="Kimi K3 官方發布視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Moonshot AI</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Kimi K3 2.8T：用 KDA 與 AttnRes 推進 3T 級開放模型</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-blue">Linear Attention</span><span class="pb-tag pb-tag-green">Agents</span></span>
<span class="pb-post-excerpt">KDA、Attention Residuals 與 896 選 16 的 Stable LatentMoE；並釐清 API 已上線、權重尚待釋出的差別。</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="notes-trtllm-triton-serving/" data-tags="Algorithm,Library">
<span class="pb-post-image"><img src="https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/llm-graphic-1.png" alt="NVIDIA TensorRT-LLM 與 Triton 部署架構" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on Serving LLMs with TensorRT-LLM and Triton</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">Serving</span><span class="pb-tag pb-tag-purple">NVIDIA stack</span></span>
<span class="pb-post-excerpt">From a Hugging Face checkpoint to a production TensorRT-LLM + Triton endpoint, benchmarked against vLLM.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="nccl-nvlink-bandwidth/" data-tags="Algorithm,Theory">
<span class="pb-post-image"><img src="https://developer-blogs.nvidia.com/wp-content/uploads/2025/07/neon-green-cube.png" alt="NVIDIA NCCL 官方主視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Where tensor-parallel inference hits the NVLink wall</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-blue">GPU</span><span class="pb-tag pb-tag-red">Distributed</span></span>
<span class="pb-post-excerpt">Measuring NCCL all-reduce and why tensor-parallel decode hits a small-message latency wall, not a bandwidth one.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="rag-groundedness-guardrail/" data-tags="Data,Attribution,Security">
<span class="pb-post-image pb-image-contain"><img src="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/multitenant-rag-single-tenant-architecture.svg" alt="Microsoft 單租戶 RAG 架構圖" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Microsoft Learn</span></span>
<span class="pb-post-body">
<span class="pb-post-title">0% vs 40%: making a RAG agent refuse to hallucinate</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">LLM</span><span class="pb-tag pb-tag-green">RAG</span></span>
<span class="pb-post-excerpt">One guardrail that drops out-of-corpus hallucination from ~40% to 0%—and why an eval harness is the only way to see it.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-federated-learning-dp/" data-tags="Data,Theory,Security">
<span class="pb-post-image"><img src="https://storage.googleapis.com/gweb-research2023-media/images/1d76da3d272f64e79d71b9da25000d2b-A.width-800.format-jpeg.jpg" alt="Google Research 聯邦學習與差分隱私示意圖" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Google Research</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on Federated Learning and Differential Privacy</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Privacy</span><span class="pb-tag pb-tag-blue">FL</span></span>
<span class="pb-post-excerpt">FedAvg, FedProx, and SCAFFOLD from scratch; what breaks under Non-IID data, and how DP-SGD adds real guarantees.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-cuda-tensor-core-gemm/" data-tags="Algorithm,Library">
<span class="pb-post-image pb-image-contain"><img src="https://developer-blogs.nvidia.com/wp-content/uploads/2017/12/wmma-warp-tile-structure.png" alt="NVIDIA WMMA warp tile 結構圖" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on CUDA Tensor Core GEMM (WMMA)</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Kernels</span></span>
<span class="pb-post-excerpt">Matrix multiply from naive to tiled to WMMA—shared-memory tiling, fragments, and the cuBLAS ceiling.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="meta-rl/" data-tags="Algorithm,Theory,Survey">
<span class="pb-post-image"><img src="https://lh3.googleusercontent.com/cL8md6gY7oJsiO_FtioQQtgxyeYisTwW7DsYGPSAZJ8aGADjz3TKoBHCwNATwEHldTzUmy7vdjXkoXZ-G0KL-6TI1SbeMU8fzXvlG970hHJoD5NpN7I=w640-h360-n-nu-rw-lo" alt="DeepMind meta-reinforcement learning 實驗主視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Google DeepMind</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Meta-Reinforcement Learning of Structured Exploration Strategies</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">ML</span><span class="pb-tag pb-tag-red">RL</span></span>
<span class="pb-post-excerpt">從「學會一個任務」走向「學會如何學習」：meta-RL 如何讓 agent 快速適應新環境。</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pp-intro/" data-tags="Algorithm,Library,Survey">
<span class="pb-post-image"><img src="https://developer.download.nvidia.com/images/cuda-platform-for-accelerated-computing-1200x630.jpg" alt="NVIDIA CUDA accelerated computing 官方視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">CUDA Programming 入門</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Parallel</span></span>
<span class="pb-post-excerpt">從執行緒層級、記憶體模型到核心函數，建立 CUDA 平行程式設計的完整基礎。</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pentest-intro/" data-tags="Security,Survey">
<span class="pb-post-image"><img src="https://www.kali.org/blog/kali-linux-2026-2-release/images/banner-2026.2-release.jpg" alt="Kali Linux 2026.2 官方桌面視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Kali Linux</span></span>
<span class="pb-post-body">
<span class="pb-post-title">滲透測試入門技術</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Security</span></span>
<span class="pb-post-excerpt">從資訊蒐集、弱點掃描到驗證與報告，理解滲透測試的目標、流程與常用工具。</span>
<span class="pb-post-date">2025-01-10</span>
</span>
</a>
</div>

<details class="pb-image-sources">
<summary>Image sources</summary>

Artwork and diagrams are credited to their original publishers: [Thinking Machines](https://thinkingmachines.ai/news/introducing-inkling/), [Moonshot AI](https://platform.kimi.ai/), [NVIDIA Developer](https://developer.nvidia.com/blog/), [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag), [Google Research](https://research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/), [Google DeepMind](https://deepmind.google/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/), and [Kali Linux](https://www.kali.org/blog/kali-linux-2026-2-release/).

</details>
