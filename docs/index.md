---
hide:
  - navigation
  - toc
---

# About

Hi, I'm **Wayne** 👋, a software engineer on LLM infrastructure at
<a href="https://ailabs.tw/"><span class="chip chip-teal"><img src="assets/logos/tail.png" alt="" width="64" height="64" />Taiwan AILabs</span></a>,
building and running on-prem LLM systems across Kubernetes, GPU runtime,
networking, storage, and model-serving endpoints.

Previously, I was the sole developer of an enterprise multi-agent AI platform at
<a href="https://syncrobotic.ai/"><span class="chip chip-navy"><img src="assets/logos/syncrobotic.png" alt="" width="64" height="64" />SYNCROBOTIC</span></a>, built internal coding agents at
<a href="https://www.advantech.com/en"><span class="chip chip-blue"><img src="assets/logos/advantech.png" alt="" width="64" height="64" />Advantech</span></a>,
and served at <a href="https://www.tsmc.com/english"><span class="chip chip-red"><img src="assets/logos/tsmc.png" alt="" width="64" height="64" />TSMC</span></a>
under the national R&D program. I hold an M.S. in Computer Science from
<a href="https://www.ntust.edu.tw/"><span class="chip chip-red"><img src="assets/logos/ntust.png" alt="" width="64" height="64" />NTUST</span></a>
(GPA 4.09), where I researched **LLM safety alignment** and **privacy-preserving ML**
in Prof. Shao-Jui Wang's lab.

Outside work I contribute upstream across the LLM-inference stack, with **20+
merged/landed changes** spanning
<a href="https://github.com/flashinfer-ai/flashinfer"><span class="chip chip-purple"><img src="assets/logos/flashinfer.png" alt="" width="64" height="64" />FlashInfer</span></a>,
<a href="https://github.com/vllm-project/vllm"><span class="chip chip-blue"><img src="assets/logos/vllm.png" alt="" width="64" height="64" />vLLM</span></a>,
<a href="https://github.com/sgl-project/sglang"><span class="chip chip-orange"><img src="assets/logos/sglang.png" alt="" width="64" height="64" />SGLang</span></a>,
<a href="https://github.com/pytorch/pytorch"><span class="chip chip-orange"><img src="assets/logos/pytorch.png" alt="" width="64" height="64" />PyTorch</span></a>,
<a href="https://github.com/ai-dynamo/dynamo"><span class="chip chip-green"><img src="assets/logos/dynamo.png" alt="" width="64" height="64" />Dynamo</span></a>,
<a href="https://github.com/NVIDIA"><span class="chip chip-green"><img src="assets/logos/nvidia.png" alt="" width="64" height="64" />NVIDIA</span></a>
<a href="https://github.com/NVIDIA/cutlass">CUTLASS</a> / <a href="https://github.com/NVIDIA/TensorRT-LLM">TensorRT-LLM</a>, and
<a href="https://github.com/InternLM/lmdeploy"><span class="chip chip-navy"><img src="assets/logos/internlm.png" alt="" width="64" height="64" />LMDeploy</span></a>,
including SM120/NVFP4 correctness fixes in FlashInfer and an NCCL broadcast fix
landed in PyTorch. See the auto-updating [PR wall](https://prs.wayne.is-a.dev)
for the complete record.

## Focus

Broadly, I care about **inference performance you can trust**: fast kernels that also
compute the right answer. Three threads:

1. **Serving internals**: KV-cache, quantization trade-offs, attention kernels,
   enough depth to reason about cost and latency *at design time*.
2. **Upstream enablement**: early consumer-Blackwell (**SM120**) and **NVFP4** support
   across kernels → engines → disaggregated serving; my favorite prey is the
   *silent-correctness* bug: tests green, answers wrong.
3. **Trustworthy ML**: safety alignment against harmful fine-tuning; federated
   learning, differential privacy, secure multi-party computation.

## News

<div class="pb-news">
<dl>
  <dt>Jul 2026</dt>
  <dd>💼 Joined <a href="https://ailabs.tw/"><strong>Taiwan AILabs</strong></a>, on-prem LLM deployments (FedGPT).</dd>
  <dt>Jul 2026</dt>
  <dd>🧱 The <a href="https://prs.wayne.is-a.dev">live PR wall</a> went up, every
      upstream patch, auto-updating.</dd>
  <dt>Apr 2026</dt>
  <dd>🎓 M.S. in Computer Science from <a href="https://www.ntust.edu.tw/"><strong>NTUST</strong></a>, GPA 4.09.</dd>
  <dt>Sep 2025</dt>
  <dd>💼 Joined <a href="https://syncrobotic.ai/"><strong>SYNCROBOTIC</strong></a>, sole developer of an enterprise
      multi-agent platform, shipped at two customers.</dd>
  <dt>Jun 2025</dt>
  <dd>💼 Summer at <a href="https://www.advantech.com/en"><strong>Advantech</strong></a> building internal coding agents.</dd>
  <dt>Dec 2024</dt>
  <dd>📜 <a href="https://learn.nvidia.com/">NVIDIA DLI</a> certificates: Accelerated Computing with CUDA (Python &amp; C/C++).</dd>
  <dt>Aug 2024</dt>
  <dd>🔬 Started graduate research on LLM security &amp; privacy-preserving ML at
      <a href="https://www.ntust.edu.tw/">NTUST</a>.</dd>
</dl>
</div>

## Misc

I'm from Taiwan :flag_tw: and based in Taipei. Away from a profiler you'll find me tending an
over-engineered Obsidian vault. The views on this site are my own and do not represent
those of my employer or affiliated institutions.
