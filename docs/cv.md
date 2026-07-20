---
hide:
  - navigation
  - toc
description: >-
  Curriculum vitae of Wei Cheng (Wayne) Chiu — LLM inference and GPU
  performance, on-premises LLM deployment, and upstream open-source work.
---

# CV

<div class="pb-cv-head" markdown>
<p class="pb-cv-contact">
<a href="mailto:waynehacking8@gmail.com">waynehacking8@gmail.com</a> ·
<a href="https://www.linkedin.com/in/wei-cheng-chiu/">LinkedIn</a> ·
<a href="https://github.com/waynehacking8">GitHub</a> ·
<a href="https://prs.wayne.is-a.dev">PR wall</a> ·
Taipei, Taiwan
</p>
<p class="pb-cv-updated">Last updated Jul 2026</p>
</div>

## Summary

Machine learning engineer working the whole depth of the LLM inference stack —
hand-written CUDA / Tensor-Core kernels underneath, TensorRT-LLM / vLLM / SGLang
serving above, and customer deployments on top. Currently a Field Application
Engineer at **Taiwan AILabs**, putting on-premises LLM systems (FedGPT) into
customer environments. Previously **sole developer** of an enterprise multi-agent
AI platform on DGX-class hardware, taken from PoC to production at **two enterprise
customers**. Active upstream — **17 merged/landed patches · 60 in review** across
FlashInfer, vLLM, SGLang, PyTorch, Dynamo, CUTLASS, TensorRT-LLM and LMDeploy,
concentrated on early consumer-Blackwell (**SM120**) and **NVFP4** enablement.
Research background in AI/LLM security and privacy-preserving ML.

## Technical Skills

<div class="pb-cv-skills" markdown>
<dl>
  <dt>LLM Serving &amp; GPU</dt>
  <dd>vLLM, TensorRT-LLM, SGLang, Triton Inference Server, NVIDIA NIM, Dynamo,
      FlashInfer, NCCL, Flash Attention, KV-cache, quantization (FP8 / INT4 /
      NVFP4); CUDA C/C++, WMMA &amp; <code>mma.sync</code> Tensor Cores, CUTLASS,
      Nsight Systems &amp; Compute</dd>
  <dt>LLM Apps &amp; Agents</dt>
  <dd>Multi-agent orchestration (LangChain, LangGraph), hybrid RAG (vector +
      keyword + knowledge graph), LLM evaluation &amp; guardrails, function
      calling / tool use, MCP, coding agents</dd>
  <dt>Infrastructure &amp; Ops</dt>
  <dd>Linux, Kubernetes, Docker, on-prem GPU serving, distributed tracing;
      H100, DGX Spark (GB10) and Blackwell workstation hardware</dd>
  <dt>Privacy &amp; Security ML</dt>
  <dd>Federated learning, differential privacy, secure multi-party computation
      (MP-SPDZ), safety alignment</dd>
  <dt>Languages</dt>
  <dd>Python, TypeScript, Rust, C/C++; PyTorch, FastAPI, NestJS;
      PostgreSQL, Milvus, Redis</dd>
</dl>
</div>

## Experience

<div class="pb-timeline" markdown>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/tail.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">Field Application Engineer</p>
<p class="pb-tl-org"><a href="https://ailabs.tw/zh/home/">Taiwan AILabs</a></p>
<p class="pb-tl-date">Jul 2026 – Present · Taipei City, Taiwan</p>

- Field deployment of on-premises LLM systems (**FedGPT**) into customer environments — Linux / Kubernetes operations, GPU serving, and customer-facing troubleshooting.
</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/syncrobotic.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">Machine Learning Engineer — Multi-Agent AI, RAG, LLM Serving</p>
<p class="pb-tl-org">SYNCROBOTIC</p>
<p class="pb-tl-date">Sep 2025 – Jun 2026 · New Taipei City, Taiwan</p>

- **Sole developer** of an enterprise multi-agent AI platform on DGX-class hardware — owned end-to-end from PoC through production and rolled out at **two enterprise customers**, partnering with each through deployment and troubleshooting.
- Scoped ambiguous enterprise requirements into a working multi-agent system, and unified local and commercial LLM backends under one routing layer, removing per-team model wrappers.
- Designed a planning–execution–validation orchestration with topological scheduling and parallel agent dispatch for multi-step workflows.
- Built a hybrid retrieval layer on LightRAG (vector + keyword + knowledge graph) with embedding-based semantic caching, cutting redundant LLM calls and keeping answers consistent on repeat queries.
- Owned production reliability — prompt/pipeline regression tests, output validation across quality and safety, distributed tracing; added CUDA-accelerated hot paths on critical inference.

*Stack: Python · TypeScript · Rust · vLLM · LightRAG · NestJS · CUDA*
</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/advantech.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">AI Agent Engineer (Intern)</p>
<p class="pb-tl-org">Advantech</p>
<p class="pb-tl-date">Jun 2025 – Aug 2025 · Taipei City, Taiwan</p>

- Built coding agents for internal developer tooling using function calling / tool use — automating code review, test generation, and static analysis across the SDLC (Claude Code-style assistants).
- Benchmarked tool-use reliability across commercial and local LLM backends; designed reusable prompt templates and skill patterns adopted by the team.
</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/tsmc.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">Facility Engineer — R&amp;D Alternative Military Service</p>
<p class="pb-tl-org">TSMC (Taiwan Semiconductor Manufacturing Co.)</p>
<p class="pb-tl-date">May 2021 – Oct 2022 · Tainan City, Taiwan</p>

- Supported R&D fab facility operations at the world's leading semiconductor foundry under the national R&D alternative military service program.
</div>
</div>

</div>

## Open Source

<div class="pb-timeline" markdown>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/nvidia.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">SM120 / NVFP4 enablement across the LLM-inference stack</p>
<p class="pb-tl-org">FlashInfer · vLLM · SGLang · PyTorch · Dynamo · CUTLASS · TensorRT-LLM · LMDeploy</p>
<p class="pb-tl-date">2026 – Present · 17 merged/landed · 60 in review</p>

- Kernel-level support for consumer Blackwell (**SM120**) and **NVFP4** carried upward through kernels → engines → disaggregated serving.
- Focus on *silent-correctness* bugs — the ones where tests pass and the answers are still wrong.
- Every patch tracked on the auto-updating [PR wall](https://prs.wayne.is-a.dev) ([RSS](https://prs.wayne.is-a.dev/feed.xml)); see also [Patches](patches.md).
</div>
</div>

</div>

## Research &amp; Publications

<div class="pb-timeline" markdown>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/ntust.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">Graduate Researcher — Prof. Shao-Jui Wang's Lab</p>
<p class="pb-tl-org">NTUST</p>
<p class="pb-tl-date">Aug 2024 – Present</p>

- **SelGrad: Selective Gradient Projection for Efficient Safety Alignment Against Harmful Fine-Tuning** — first author, with Shao-Jui Wang. Under review at *IEEE Transactions on Dependable and Secure Computing (TDSC)*, 2026. Preserves LLM safety alignment under harmful/adversarial fine-tuning at lower compute than full re-alignment.
- Research in **AI/LLM security and safety alignment** and **privacy-enhancing ML** — federated learning, differential privacy, and secure multi-party computation.
- Differentially-private synthetic-data generation with **MP-SPDZ** for the NICS privacy-enhancing-technologies competition.
</div>
</div>

</div>

## Education

<div class="pb-timeline" markdown>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/ntust.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">M.S. in Computer Science and Information Engineering</p>
<p class="pb-tl-org">National Taiwan University of Science and Technology (NTUST)</p>
<p class="pb-tl-date">Aug 2024 – Apr 2026 · GPA 4.09 · Taipei, Taiwan</p>

Research focus: LLMOps (fine-tuning, safety alignment), CUDA parallel programming,
inference acceleration.
</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/ntu.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">M.S. in Structural Engineering</p>
<p class="pb-tl-org">National Taiwan University (NTU)</p>
<p class="pb-tl-date">Sep 2018 – Jun 2020 · Taipei, Taiwan</p>
</div>
</div>

<div class="pb-tl-item" markdown>
<div class="pb-tl-icon"><img src="/assets/logos/ntu.png" alt=""></div>
<div class="pb-tl-body" markdown>
<p class="pb-tl-title">B.S. in Civil Engineering</p>
<p class="pb-tl-org">National Taiwan University (NTU)</p>
<p class="pb-tl-date">Sep 2014 – Jun 2018 · Taipei, Taiwan</p>
</div>
</div>

</div>

## Certifications &amp; Awards

- **NVIDIA Deep Learning Institute** — Fundamentals of Accelerated Computing with CUDA C/C++ and CUDA Python (5 certificates), 2024.
- **NICS Privacy-Enhancing Technologies (PETs) Competition** — Rank 3/11, differentially-private synthetic data via MP-SPDZ.
- **International ICT Innovative Services Awards**, 2024.
