---
title: Wayne Chiu
template: home.html
tagline: Field application engineer on LLM infrastructure at Taiwan AILabs. I also send fixes upstream to vLLM and FlashInfer.
place: Taipei, Taiwan
school: CSIE NTUST

# Content of the one-page home (overrides/home.html). OPEN SOURCE is not listed
# here: its rows are built from the PR wall feed by hooks/patches.py.
experience:
  - org: Taiwan AILabs
    url: https://ailabs.tw/
    logo: /assets/logos/tail.png
    role: Field Application Engineer
    date: Jul 2026 · Present
    note: Deploy and operate on-prem LLM systems (FedGPT) on Kubernetes and NVIDIA GPUs, from bare metal to handover.
  - org: SYNCROBOTIC
    url: https://syncrobotic.ai/
    logo: /assets/logos/syncrobotic.png
    role: Machine Learning Engineer
    date: Sep 2025 · Jun 2026
    note: Sole developer of an enterprise multi-agent RAG platform on DGX hardware, shipped to two customers.
  - org: Advantech
    url: https://www.advantech.com/en
    logo: /assets/logos/advantech.png
    role: AI Agent Engineer (Intern)
    date: Jun 2025 · Aug 2025
    note: Built tool-using coding agents for code review, test generation, and static analysis.
  - org: TSMC
    url: https://www.tsmc.com/english
    logo: /assets/logos/tsmc.png
    role: Facility Engineer (R&D Alternative Military Service)
    date: May 2021 · Oct 2022

education:
  - org: National Taiwan University of Science and Technology
    url: https://www.ntust.edu.tw/
    logo: /assets/logos/ntust.png
    role: M.S. in Computer Science and Information Engineering
    date: Aug 2024 · Apr 2026
    bullets:
      - GPA 4.09.
      - Research on LLM safety alignment and privacy-preserving ML in Prof. Shao-Jui Wang's lab.
      - Coursework in Machine Learning, Deep Learning, Parallel Programming and Cybersecurity.
  - org: National Taiwan University
    logo: /assets/logos/ntu.png
    role: M.S. in Structural Engineering
    date: Sep 2018 · Jun 2020
  - org: National Taiwan University
    logo: /assets/logos/ntu.png
    role: B.S. in Civil Engineering
    date: Sep 2014 · Jun 2018

publications:
  - title: "SelGrad: Selective Gradient Projection for Efficient Safety Alignment Against Harmful Fine-Tuning"
    role: First author. IEEE TDSC, under review.
    date: "2026"

projects:
  - name: nccl-collectives-bench
    url: https://github.com/waynehacking8/nccl-collectives-bench
    post: /blog/nccl-nvlink-bandwidth/
    note: NCCL collective benchmarks (all-reduce, all-gather, reduce-scatter) on 4x H100 NVLink, with an analytical roofline checked against the measurements.
  - name: trtllm-triton-serving
    url: https://github.com/waynehacking8/trtllm-triton-serving
    post: /blog/notes-trtllm-triton-serving/
    note: LLM serving on TensorRT-LLM with tensor parallelism across 4x H100, benchmarked against vLLM at the same model, hardware and request load, plus a Triton deployment template.
  - name: nim-agent-blueprint
    url: https://github.com/waynehacking8/nim-agent-blueprint
    post: /blog/rag-groundedness-guardrail/
    note: Agentic RAG reference architecture on NVIDIA NIM microservices. The agent plans, retrieves and validates its answer, and an eval harness with OpenTelemetry traces checks it.
  - name: federated-learning-lab
    url: https://github.com/waynehacking8/federated-learning-lab
    post: /blog/notes-federated-learning-dp/
    note: Federated learning written from scratch with DP and secure aggregation, 33 passing tests, results checked against published numbers.

certifications:
  - org: NVIDIA Deep Learning Institute
    url: https://learn.nvidia.com/
    logo: /assets/logos/nvidia.png
    role: Fundamentals of Accelerated Computing with CUDA Python
    date: Dec 2024
  - org: NVIDIA Deep Learning Institute
    url: https://learn.nvidia.com/
    logo: /assets/logos/nvidia.png
    role: Fundamentals of Accelerated Computing with CUDA C/C++
    date: Nov 2024
---
