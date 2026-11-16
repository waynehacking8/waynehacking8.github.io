# Blog

Notes I write while reading model releases and running benchmarks: architecture,
serving, CUDA, and some privacy and security. When a post makes a claim, the
measurement or source that backs it sits right beside it.

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
<a class="pb-post-card" href="vllm-flashinfer-async-race/" data-tags="Serving,CUDA">
<span class="pb-post-image pb-image-contain"><img src="/assets/blog/vllm-61239-faults.png" alt="vllm#61239 illegal memory accesses per build" width="1200" height="630" loading="eager" fetchpriority="high" decoding="async"><span class="pb-image-credit">Wayne</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Which patch cleared the FlashInfer fault on my card</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-blue">Serving</span><span class="pb-tag pb-tag-red">CUDA</span></span>
<span class="pb-post-excerpt">Five vLLM builds, 30 servers each, on one Blackwell workstation GPU: main faulted 8 times without #64165 and never with it.</span>
<span class="pb-post-date">2026-11-16</span>
</span>
</a>
<a class="pb-post-card" href="agent-harness-boundaries/" data-tags="Architecture,Agents,Security">
<span class="pb-post-image pb-image-contain"><img src="/assets/blog/agent-runtime/overview.png" alt="Agent runtime architectures of Prime, Hermes and OpenClaw compared" width="1300" height="680" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Wayne</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Prime, Hermes, OpenClaw: Architecture Choices in Three Agent Runtimes</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-green">Agents</span><span class="pb-tag pb-tag-red">Security</span></span>
<span class="pb-post-excerpt">From persistent Python to a personal agent service to a Gateway control plane: how three runtimes take over the model's work.</span>
<span class="pb-post-date">2026-09-15</span>
</span>
</a>
<a class="pb-post-card" href="inkling-975b-architecture/" data-tags="Architecture">
<span class="pb-post-image"><img src="/assets/blog/inkling-cover.png" alt="Inkling launch artwork" width="1200" height="630" loading="eager" fetchpriority="high" decoding="async"><span class="pb-image-credit">Thinking Machines</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Inkling 975B: How 5:1 Hybrid Attention Reallocates Compute</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-green">MoE</span><span class="pb-tag pb-tag-blue">Multimodal</span></span>
<span class="pb-post-excerpt">5:1 sliding windows, relative position encoding, short convolutions and Muon/Adam in Thinking Machines' first open-weight model, checked against the official material.</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="kimi-k3-architecture/" data-tags="Architecture">
<span class="pb-post-image"><img src="/assets/blog/kimi-k3-cover.webp" alt="Kimi K3 launch artwork" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Moonshot AI</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Kimi K3 2.8T: KDA, AttnRes and a 16-of-896 MoE</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">Architecture</span><span class="pb-tag pb-tag-blue">Linear Attention</span><span class="pb-tag pb-tag-green">Agents</span></span>
<span class="pb-post-excerpt">KDA, Attention Residuals and a 16-of-896 Stable LatentMoE, and the difference between the API being live and the weights still unreleased.</span>
<span class="pb-post-date">2026-07-20</span>
</span>
</a>
<a class="pb-post-card" href="notes-trtllm-triton-serving/" data-tags="Serving">
<span class="pb-post-image"><img src="/assets/blog/trtllm-triton.webp" alt="NVIDIA TensorRT-LLM and Triton deployment architecture" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">TensorRT-LLM vs vLLM on 4×H100: The Concurrency Crossover</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">Serving</span><span class="pb-tag pb-tag-purple">NVIDIA stack</span></span>
<span class="pb-post-excerpt">A matched benchmark: TensorRT-LLM leads at low and mid concurrency, vLLM leads at high concurrency.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="nccl-nvlink-bandwidth/" data-tags="Distributed">
<span class="pb-post-image"><img src="/assets/blog/nccl-cube.webp" alt="NVIDIA NCCL artwork" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Where tensor-parallel inference hits the NVLink wall</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-blue">GPU</span><span class="pb-tag pb-tag-red">Distributed</span></span>
<span class="pb-post-excerpt">366 GB/s against a 23 μs fixed cost, and what NVLS, CUDA Graphs and symmetric memory each change when measured separately.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="rag-groundedness-guardrail/" data-tags="RAG,Security">
<span class="pb-post-image pb-image-contain"><img src="/assets/blog/rag-architecture.svg" alt="Microsoft single-tenant RAG architecture" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Microsoft Learn</span></span>
<span class="pb-post-body">
<span class="pb-post-title">The 0% RAG result failed the harder test</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">LLM</span><span class="pb-tag pb-tag-green">RAG</span></span>
<span class="pb-post-excerpt">0% on the easy set, 48% on adversarial near-misses. Across nine gates, a grounded verifier caught errors the generative judges missed.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-federated-learning-dp/" data-tags="Privacy,ML,Security">
<span class="pb-post-image"><img src="/assets/blog/federated-dp.webp" alt="Google Research figure on federated learning with differential privacy" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Google Research</span></span>
<span class="pb-post-body">
<span class="pb-post-title">FedAvg Under Non-IID Data: What DP-SGD and Secure Aggregation Protect</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Privacy</span><span class="pb-tag pb-tag-blue">FL</span></span>
<span class="pb-post-excerpt">From client drift to privacy bounds: measured limits of FedAvg, FedProx, SCAFFOLD, DP-SGD and secure aggregation.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="notes-cuda-tensor-core-gemm/" data-tags="CUDA">
<span class="pb-post-image pb-image-contain"><img src="/assets/blog/wmma-tile.webp" alt="NVIDIA WMMA warp tile structure" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">From Naive GEMM to WMMA: Where Each CUDA Kernel Stalls</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Kernels</span></span>
<span class="pb-post-excerpt">Naive, shared-memory tiling and WMMA kernels compared, with cuBLAS on the same shape as the production baseline.</span>
<span class="pb-post-date">2026-05-31</span>
</span>
</a>
<a class="pb-post-card" href="meta-rl/" data-tags="ML">
<span class="pb-post-image"><img src="/assets/blog/meta-rl.webp" alt="DeepMind meta-reinforcement learning artwork" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Google DeepMind</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Meta-RL: Teaching a Policy to Update Itself Within an Episode</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-purple">ML</span><span class="pb-tag pb-tag-red">RL</span></span>
<span class="pb-post-excerpt">Task inference placed in gradients, recurrent state or latent context, and how to test for adaptation rather than task memorization.</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pp-intro/" data-tags="CUDA">
<span class="pb-post-image"><img src="/assets/blog/cuda-platform.webp" alt="NVIDIA CUDA accelerated computing artwork" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">NVIDIA</span></span>
<span class="pb-post-body">
<span class="pb-post-title">Intro to CUDA Programming</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Parallel</span></span>
<span class="pb-post-excerpt">Using the execution model, memory hierarchy and synchronization to find where a CUDA kernel is actually bound.</span>
<span class="pb-post-date">2025-01-11</span>
</span>
</a>
<a class="pb-post-card" href="pentest-intro/" data-tags="Security">
<span class="pb-post-image"><img src="/assets/blog/kali-2026-2.webp" alt="Kali Linux 2026.2 desktop artwork" loading="lazy" fetchpriority="low" decoding="async"><span class="pb-image-credit">Kali Linux</span></span>
<span class="pb-post-body">
<span class="pb-post-title">A Pentest Is Not a Scanner: From Authorization Boundaries to Retestable Evidence</span>
<span class="pb-post-tags"><span class="pb-tag pb-tag-red">Security</span></span>
<span class="pb-post-excerpt">Rules of Engagement before tools, then minimal verification, an evidence chain and acceptance tests to hand each finding to the people fixing it.</span>
<span class="pb-post-date">2025-01-10</span>
</span>
</a>
</div>

<details class="pb-image-sources">
<summary>Image sources</summary>

Artwork and diagrams are credited to their original publishers: [Thinking Machines](https://thinkingmachines.ai/news/introducing-inkling/), [Moonshot AI](https://platform.kimi.ai/), [NVIDIA Developer](https://developer.nvidia.com/blog/), [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag), [Google Research](https://research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/), [Google DeepMind](https://deepmind.google/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/), and [Kali Linux](https://www.kali.org/blog/kali-linux-2026-2-release/).

</details>
