---
hide:
  - navigation
  - toc
---

# About

Hi, I'm **Wayne** 👋 — a GPU performance engineer, happiest when an upstream kernel
fix lands. Currently, I'm a Field Application Engineer at
<a href="https://ailabs.tw/zh/home/"><span class="chip chip-teal"><img src="assets/logos/tail.png" alt="" width="64" height="64" />Taiwan AILabs</span></a>,
deploying on-premises LLM systems (FedGPT) into customer environments — Linux/Kubernetes
operations, GPU serving, and customer-facing troubleshooting.

Previously, I was the sole developer of an enterprise multi-agent AI platform at
<a href="https://syncrobotic.ai/"><span class="chip chip-navy"><img src="assets/logos/syncrobotic.png" alt="" width="64" height="64" />SYNCROBOTIC</span></a>, built internal coding agents at
<a href="https://www.advantech.com/en"><span class="chip chip-blue"><img src="assets/logos/advantech.png" alt="" width="64" height="64" />Advantech</span></a>,
and served at <a href="https://www.tsmc.com/english"><span class="chip chip-red"><img src="assets/logos/tsmc.png" alt="" width="64" height="64" />TSMC</span></a>
under the national R&D program. I hold an M.S. in Computer Science from
<a href="https://www.ntust.edu.tw/"><span class="chip chip-red"><img src="assets/logos/ntust.png" alt="" width="64" height="64" />NTUST</span></a>
(GPA 4.09), where I researched **LLM safety alignment** and **privacy-preserving ML**
in Prof. Shao-Jui Wang's lab.

Outside work I live in the upstream LLM-inference stack — **17 merged/landed patches ·
60 in review** across
<a href="https://github.com/flashinfer-ai/flashinfer"><span class="chip chip-purple"><img src="assets/logos/flashinfer.png" alt="" width="64" height="64" />FlashInfer</span></a>,
<a href="https://github.com/vllm-project/vllm"><span class="chip chip-blue"><img src="assets/logos/vllm.png" alt="" width="64" height="64" />vLLM</span></a>,
<a href="https://github.com/sgl-project/sglang"><span class="chip chip-orange"><img src="assets/logos/sglang.png" alt="" width="64" height="64" />SGLang</span></a>,
<a href="https://github.com/pytorch/pytorch"><span class="chip chip-orange"><img src="assets/logos/pytorch.png" alt="" width="64" height="64" />PyTorch</span></a>,
<a href="https://github.com/ai-dynamo/dynamo"><span class="chip chip-green"><img src="assets/logos/dynamo.png" alt="" width="64" height="64" />Dynamo</span></a>,
<a href="https://github.com/NVIDIA"><span class="chip chip-green"><img src="assets/logos/nvidia.png" alt="" width="64" height="64" />NVIDIA</span></a>
<a href="https://github.com/NVIDIA/cutlass">CUTLASS</a> / <a href="https://github.com/NVIDIA/TensorRT-LLM">TensorRT-LLM</a>, and
<a href="https://github.com/InternLM/lmdeploy"><span class="chip chip-navy"><img src="assets/logos/internlm.png" alt="" width="64" height="64" />LMDeploy</span></a> —
see the auto-updating [PR wall](https://prs.wayne.is-a.dev) and [Patches](patches.md).

## Focus

Broadly, I care about **inference performance you can trust** — fast kernels that also
compute the right answer. Three threads:

1. **Serving internals**: KV-cache, quantization trade-offs, attention kernels —
   enough depth to reason about cost and latency *at design time*.
2. **Upstream enablement**: early consumer-Blackwell (**SM120**) and **NVFP4** support
   across kernels → engines → disaggregated serving; my favorite prey is the
   *silent-correctness* bug — tests green, answers wrong.
3. **Trustworthy ML**: safety alignment against harmful fine-tuning; federated
   learning, differential privacy, secure multi-party computation.

## News

<div class="pb-news">
<dl>
  <dt>Jul 2026</dt>
  <dd>💼 Joined <a href="https://ailabs.tw/zh/home/"><strong>Taiwan AILabs</strong></a> as a Field Application Engineer —
      on-prem LLM deployments (FedGPT).</dd>
  <dt>Jul 2026</dt>
  <dd>🧱 The <a href="https://prs.wayne.is-a.dev">live PR wall</a> went up — every
      upstream patch, auto-updating.</dd>
  <dt>Jul 2026</dt>
  <dd>📈 Upstream tally: <strong>17 merged/landed · 60 in review</strong> across the
      LLM-inference stack.</dd>
  <dt>Apr 2026</dt>
  <dd>🎓 M.S. in Computer Science from <a href="https://www.ntust.edu.tw/"><strong>NTUST</strong></a>, GPA 4.09.</dd>
  <dt>Jan 2026</dt>
  <dd>📝 <strong>SelGrad</strong> (first-author) submitted — under review at
      <em>IEEE TDSC</em>.</dd>
  <dt>Sep 2025</dt>
  <dd>💼 Joined <a href="https://syncrobotic.ai/"><strong>SYNCROBOTIC</strong></a> — sole developer of an enterprise
      multi-agent platform, shipped at two customers.</dd>
  <dt>Jun 2025</dt>
  <dd>💼 Summer at <a href="https://www.advantech.com/en"><strong>Advantech</strong></a> building internal coding agents.</dd>
  <dt>Dec 2024</dt>
  <dd>📜 <a href="https://learn.nvidia.com/">NVIDIA DLI</a> certificates — Accelerated Computing with CUDA (Python &amp; C/C++).</dd>
  <dt>Aug 2024</dt>
  <dd>🔬 Started graduate research on LLM security &amp; privacy-preserving ML at
      <a href="https://www.ntust.edu.tw/">NTUST</a>.</dd>
</dl>
</div>

## Selected Work

<div class="pb-pub" markdown>
<div markdown>
<p class="pb-pub-title"><a href="patches/">SM120 / NVFP4 enablement across the LLM-inference stack</a></p>
<p class="pb-pub-meta"><a href="https://github.com/flashinfer-ai/flashinfer">FlashInfer</a> · <a href="https://github.com/NVIDIA/cutlass">CUTLASS</a> · <a href="https://github.com/vllm-project/vllm">vLLM</a> · <a href="https://github.com/sgl-project/sglang">SGLang</a> · <a href="https://github.com/NVIDIA/TensorRT-LLM">TensorRT-LLM</a> · <a href="https://github.com/ai-dynamo/dynamo">Dynamo</a> — kernels to engines to disaggregated serving</p>
<span class="pb-tag pb-tag-green">CUDA</span><span class="pb-tag pb-tag-blue">Blackwell</span><span class="pb-tag pb-tag-purple">NVFP4</span>
</div>
</div>

<div class="pb-pub" markdown>
<div markdown>
<p class="pb-pub-title"><a href="https://prs.wayne.is-a.dev">Live PR wall — prs.wayne.is-a.dev</a></p>
<p class="pb-pub-meta">Auto-updating feed of every upstream contribution, with <a href="https://prs.wayne.is-a.dev/feed.xml">RSS</a></p>
<span class="pb-tag pb-tag-blue">Open Source</span><span class="pb-tag pb-tag-green">17 merged</span><span class="pb-tag pb-tag-red">60 in review</span>
</div>
<img src="assets/previews/pr-wall-preview.png" alt="PR wall preview" width="1378" height="874" loading="lazy" decoding="async" />
</div>

<div class="pb-pub" markdown>
<div markdown>
<p class="pb-pub-title">SelGrad: Selective Gradient Projection for Efficient Safety Alignment Against Harmful Fine-Tuning</p>
<p class="pb-pub-meta"><b>Wei-Cheng Chiu</b>, et al. — under review at <em>IEEE TDSC</em></p>
<span class="pb-tag pb-tag-red">Safety</span><span class="pb-tag pb-tag-purple">Alignment</span>
</div>
</div>

<div class="pb-pub" markdown>
<div markdown>
<p class="pb-pub-title"><a href="projects/">Enterprise multi-agent AI platform</a></p>
<p class="pb-pub-meta">PoC → production at two enterprise customers — routing layer, planning/execution/validation orchestration, hybrid RAG</p>
<span class="pb-tag pb-tag-blue">Agents</span><span class="pb-tag pb-tag-green">RAG</span><span class="pb-tag pb-tag-purple">vLLM</span>
</div>
</div>

## Misc

I'm from Taiwan :flag_tw: and based in Taipei. Away from a profiler you'll find me tending an
over-engineered Obsidian vault. The views on this site are my own and do not represent
those of my employer or affiliated institutions.
