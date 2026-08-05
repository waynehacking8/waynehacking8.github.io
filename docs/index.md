---
hide:
  - navigation
  - toc
---

# About

Upstream contributions across the LLM-inference stack span
<a href="https://github.com/flashinfer-ai/flashinfer"><span class="chip chip-purple"><img src="assets/logos/flashinfer.png" alt="" width="64" height="64" />FlashInfer</span></a>,
<a href="https://github.com/vllm-project/vllm"><span class="chip chip-blue"><img src="assets/logos/vllm.png" alt="" width="64" height="64" />vLLM</span></a>,
<a href="https://github.com/sgl-project/sglang"><span class="chip chip-orange"><img src="assets/logos/sglang.png" alt="" width="64" height="64" />SGLang</span></a>,
<a href="https://github.com/pytorch/pytorch"><span class="chip chip-orange"><img src="assets/logos/pytorch.png" alt="" width="64" height="64" />PyTorch</span></a>,
<a href="https://github.com/ai-dynamo/dynamo"><span class="chip chip-green"><img src="assets/logos/dynamo.png" alt="" width="64" height="64" />Dynamo</span></a>,
<a href="https://github.com/NVIDIA"><span class="chip chip-green"><img src="assets/logos/nvidia.png" alt="" width="64" height="64" />NVIDIA</span></a>
<a href="https://github.com/NVIDIA/cutlass">CUTLASS</a> / <a href="https://github.com/NVIDIA/TensorRT-LLM">TensorRT-LLM</a>, and
<a href="https://github.com/InternLM/lmdeploy"><span class="chip chip-navy"><img src="assets/logos/internlm.png" alt="" width="64" height="64" />LMDeploy</span></a> —
with work on correctness, distributed inference, and serving performance. See the
auto-updating [PR wall](https://prs.wayne.is-a.dev) and [Patches](patches.md).

Production LLM infrastructure for real customer environments, from site constraints
and preflight validation to acceptance and handover. Work spans Kubernetes, GPU runtime,
networking, storage, certificates, and model-serving endpoints.

Experience taking an enterprise multi-agent platform from PoC to production, with a
focus on model serving, performance evaluation, and solution architecture.

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
  <dd>💼 Joined <a href="https://ailabs.tw/"><strong>Taiwan AILabs</strong></a> —
      on-prem LLM deployments (FedGPT).</dd>
  <dt>Jul 2026</dt>
  <dd>🧱 The <a href="https://prs.wayne.is-a.dev">live PR wall</a> went up — every
      upstream patch, auto-updating.</dd>
  <dt>Apr 2026</dt>
  <dd>🎓 M.S. in Computer Science from <a href="https://www.ntust.edu.tw/"><strong>NTUST</strong></a>, GPA 4.09.</dd>
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
<span class="pb-tag pb-tag-blue">Open Source</span><span class="pb-tag pb-tag-green">Live status</span>
</div>
<img src="assets/previews/pr-wall-preview.png" alt="PR wall preview" width="1378" height="874" loading="lazy" decoding="async" />
</div>

## Misc

I'm from Taiwan :flag_tw: and based in Taipei. Away from a profiler you'll find me tending an
over-engineered Obsidian vault. The views on this site are my own and do not represent
those of my employer or affiliated institutions.
