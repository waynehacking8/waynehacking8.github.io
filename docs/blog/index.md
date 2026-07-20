# Blog

I use these notes to answer the questions that survive the launch post: where the
latency comes from, which claim the evidence actually supports, and what breaks when
an architecture meets a real runtime. Equations, measurements, and source notes stay
next to the claim they qualify.

<div class="pb-filters" aria-label="Filter blog posts by tag">
<span class="pb-filters-label">Topics:</span>
<button type="button" data-tag="Architecture">Architecture</button>
<button type="button" data-tag="Serving">Serving</button>
<button type="button" data-tag="CUDA">CUDA</button>
<button type="button" data-tag="Distributed">Distributed</button>
<button type="button" data-tag="RAG">RAG</button>
<button type="button" data-tag="Privacy">Privacy</button>
<button type="button" data-tag="ML">ML</button>
<button type="button" data-tag="Security">Security</button>
</div>

<div class="pb-posts">
<a class="pb-post-card" href="inkling-975b-architecture/" data-tags="Architecture">
<span class="pb-post-image"><img src="/assets/blog/inkling-cover.png" alt="Inkling 官方發布視覺" width="1200" height="630" loading="eager" fetchpriority="high" decoding="async"><span class="pb-image-credit">Thinking Machines</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Inkling 975B：不只把模型做大，而是重新分配每一分算力</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-green">MoE</span><span class="pb-tag pb-tag-blue">Multimodal</span></span>
<span class="pb-post-excerpt">5:1 滑動視窗、相對位置編碼、短卷積與 Muon/Adam：查核官方資料後拆解 Thinky 的第一個開放權重模型。</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="kimi-k3-architecture/" data-tags="Architecture">
<span class="pb-post-image"><img src="/assets/blog/kimi-k3-cover.webp" alt="Kimi K3 官方發布視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Moonshot AI</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Kimi K3 2.8T：用 KDA 與 AttnRes 推進 3T 級開放模型</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-blue">Linear Attention</span><span class="pb-tag pb-tag-green">Agents</span></span>
<span class="pb-post-excerpt">KDA、Attention Residuals 與 896 選 16 的 Stable LatentMoE；並釐清 API 已上線、權重尚待釋出的差別。</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="notes-trtllm-triton-serving/" data-tags="Serving">
<span class="pb-post-image"><img src="/assets/blog/trtllm-triton.webp" alt="NVIDIA TensorRT-LLM 與 Triton 部署架構" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on Serving LLMs with TensorRT-LLM and Triton</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">Serving</span><span class="pb-tag pb-tag-purple">NVIDIA stack</span></span>
<span class="pb-post-excerpt">From a Hugging Face checkpoint to a production TensorRT-LLM + Triton endpoint, benchmarked against vLLM.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="nccl-nvlink-bandwidth/" data-tags="Distributed">
<span class="pb-post-image"><img src="/assets/blog/nccl-cube.webp" alt="NVIDIA NCCL 官方主視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Where tensor-parallel inference hits the NVLink wall</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-blue">GPU</span><span class="pb-tag pb-tag-red">Distributed</span></span>
<span class="pb-post-excerpt">366 GB/s, a 23 μs floor, and the measurements that separate NVLS, CUDA Graphs, and symmetric memory from their headlines.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="rag-groundedness-guardrail/" data-tags="RAG,Security">
<span class="pb-post-image pb-image-contain"><img src="/assets/blog/rag-architecture.svg" alt="Microsoft 單租戶 RAG 架構圖" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Microsoft Learn</span></span>
<span class="pb-post-body">
<span class="pb-post-title">The 0% RAG result failed the harder test</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">LLM</span><span class="pb-tag pb-tag-green">RAG</span></span>
<span class="pb-post-excerpt">The easy set said 0%. Adversarial near-misses said 48%. Nine gates later, grounded verification—not a larger judge—won.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-federated-learning-dp/" data-tags="Privacy,ML,Security">
<span class="pb-post-image"><img src="/assets/blog/federated-dp.webp" alt="Google Research 聯邦學習與差分隱私示意圖" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Google Research</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on Federated Learning and Differential Privacy</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Privacy</span><span class="pb-tag pb-tag-blue">FL</span></span>
<span class="pb-post-excerpt">FedAvg, FedProx, and SCAFFOLD from scratch; what breaks under Non-IID data, and how DP-SGD adds real guarantees.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-cuda-tensor-core-gemm/" data-tags="CUDA">
<span class="pb-post-image pb-image-contain"><img src="/assets/blog/wmma-tile.webp" alt="NVIDIA WMMA warp tile 結構圖" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Notes on CUDA Tensor Core GEMM (WMMA)</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Kernels</span></span>
<span class="pb-post-excerpt">Matrix multiply from naive to tiled to WMMA—shared-memory tiling, fragments, and the cuBLAS ceiling.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="meta-rl/" data-tags="ML">
<span class="pb-post-image"><img src="/assets/blog/meta-rl.webp" alt="DeepMind meta-reinforcement learning 實驗主視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Google DeepMind</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Meta-RL：讓 policy 在 episode 內學會更新自己</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">ML</span><span class="pb-tag pb-tag-red">RL</span></span>
<span class="pb-post-excerpt">把 task inference 放進 gradient、recurrent state 或 latent context；以及怎麼測出 adaptation 而不是 task memorization。</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pp-intro/" data-tags="CUDA">
<span class="pb-post-image"><img src="/assets/blog/cuda-platform.webp" alt="NVIDIA CUDA accelerated computing 官方視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">CUDA Programming 入門</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Parallel</span></span>
<span class="pb-post-excerpt">從執行緒層級、記憶體模型到核心函數，建立 CUDA 平行程式設計的完整基礎。</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pentest-intro/" data-tags="Security">
<span class="pb-post-image"><img src="/assets/blog/kali-2026-2.webp" alt="Kali Linux 2026.2 官方桌面視覺" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Kali Linux</span></span>
<span class="pb-post-body">
<span class="pb-post-title">滲透測試不是掃描器：從授權邊界到可複測證據</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Security</span></span>
<span class="pb-post-excerpt">先寫 Rules of Engagement，再談工具；用最小化驗證、證據鏈與 acceptance test 把 finding 交給修復者。</span>
<span class="pb-post-date">2025-01-10</span>
</span>
</a>
</div>

<details class="pb-image-sources">
<summary>Image sources</summary>

Artwork and diagrams are credited to their original publishers: [Thinking Machines](https://thinkingmachines.ai/news/introducing-inkling/), [Moonshot AI](https://platform.kimi.ai/), [NVIDIA Developer](https://developer.nvidia.com/blog/), [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag), [Google Research](https://research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/), [Google DeepMind](https://deepmind.google/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/), and [Kali Linux](https://www.kali.org/blog/kali-linux-2026-2-release/).

</details>
