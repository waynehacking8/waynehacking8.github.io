---
description: "A walkthrough of the Thinking Machines Lab Inkling 975B architecture: 5:1 sliding-window attention, relative position embeddings, short convolutions, jointly normalized MoE, and hybrid Muon/Adam optimization."
date: "2026-07-20"
language: "en"
image: "https://thinkingmachines.ai/news/introducing-inkling/images/cover-social-inkling-post.png"
tags:
  - Architecture
  - MoE
  - Multimodal
---

# Inkling 975B: How 5:1 Hybrid Attention Reallocates Compute

*2026-07-20 · LLM Architecture / MoE / Multimodal*

<figure class="pb-article-hero">
  <img src="/assets/blog/inkling-cover.png" width="1200" height="630" alt="Official launch key visual for Thinking Machines Lab Inkling" loading="eager" decoding="async">
  <figcaption>Official launch key visual<span class="pb-cap-inline"><span class="pb-cap-dot"> · </span><a href="https://thinkingmachines.ai/news/introducing-inkling/">Source: Thinking Machines Lab</a></span></figcaption>
</figure>

On July 15, 2026, Thinking Machines Lab released its first open-weight model, **Inkling**. It has 975B total parameters, activates 41B per token, supports up to 1M tokens of context, and was pretrained on 45T text, image, audio and video tokens. The lab positions it as a natively multimodal base model meant for fine-tuning, and uses sparse activation and hybrid attention to keep inference cost under control.[^inkling-release][^inkling-card]

## TL;DR

- A **975B total / 41B active** MoE Transformer with 1M context.
- The attention layers use a **5:1 sliding-window / global** mix instead of global attention in every layer.
- RoPE is dropped in favor of **relative positional embedding**.
- **Short convolution** is added to the attention projections and the residual branches.
- 256 routed experts and 2 shared experts, with 6 routed experts active per token. The scores of both kinds of expert are normalized together.
- **Muon** for the large matrices, **Adam** for everything else.
- Vision and audio use encoder-free input, then are processed together with text tokens.

## 1. 5:1 hybrid attention: global compute happens less often

Methods like MLA mainly compress the KV cache. That is a different axis from whether every layer runs global attention. Inkling makes a more direct structural trade: in every six attention layers, five look only at a local sliding window and one keeps a global view, with 8 KV heads.

One layer in six still exchanges information across segments, and the other five confine most of their compute to nearby tokens. For long sequences this lowers how dense global attention is. It deals with how often global compute happens, which is a different problem from shrinking the cache representation.

The lab has published only the architecture ratio and benchmarks so far, with no full ablations. So I can't attribute every speed difference to the 5:1 attention, and I can't generalize it to "always faster than every compressed attention."

### Writing the 5:1 saving as a cost formula

Let the sequence length be $n$ and the sliding-window width $w$. Ignoring head dimension and constants, the score computation of one global attention layer is about $n^2$, and a local layer is about $nw$. For a group of six layers:

$$
C_{\text{Inkling}} \propto 5nw+n^2,
\qquad
\frac{C_{\text{Inkling}}}{C_{\text{all-global}}}
=\frac{5nw+n^2}{6n^2}
=\frac{1}{6}+\frac{5w}{6n}.
$$

When $n \gg w$, the attention-score term approaches one sixth of six global layers, and it never drops to zero. The global layer still costs $O(n^2)$. End-to-end latency also includes the QKV projections, MoE, communication and memory movement, so the formula does not mean the whole model gets six times faster.

## 2. Relative position embeddings replace RoPE

Inkling uses relative position embeddings instead of the RoPE found in Llama, Qwen and most current models. The lab says that in this hybrid sliding-window architecture, relative position representations extrapolate better to longer sequences.

My intuition: RoPE rotates position information into the query and key, while relative position methods express the distance between tokens directly. The latter fits a fixed local window naturally. Local attention mostly cares about "how many positions apart", and the global layers handle integration across windows.

Inkling's choice applies only to this hybrid sliding-window architecture. Position encoding still has to be evaluated together with the attention topology, and can't be compared on its own, apart from the rest of the model.

## 3. Short convolution handles local mixing

Inkling adds short convolutions in two places:

1. After the key/value projection in every attention layer.
2. Before the attention and MLP residual branches return to the main residual stream.

The short convolution handles nearby patterns first, and attention keeps handling content-dependent and longer-range relations. This gives the Transformer an extra data path with a local inductive bias.

## 4. MoE: shared and routed experts settled together

Inkling's MoE largely follows DeepSeek-V3: each layer has **256 routed experts + 2 shared experts**, each token selects 6 routed experts, the router uses a sigmoid, and load balancing uses an auxiliary-loss-free bias.

Inkling **jointly normalizes** the scores of the selected routed experts and the shared experts, then combines the outputs with those weights. The router can therefore allocate the contribution of both kinds of expert on the same scale.

The Thinking Machines release post and model card describe this mechanism but do not name it. The Day-0 systems article published jointly by LMSYS/SGLang and Thinking Machines calls it a **shared-expert sink**. So "Sink MoE" works as an engineering shorthand, but what it means concretely is that shared and routed experts share one normalized weight budget.[^inkling-day0]

## 5. Muon for matrices, Adam for everything else

Training uses a hybrid optimizer strategy: **Muon** for large matrix weights and **Adam** for the other parameters. The lab also couples weight decay strength to the square of the learning rate, to control the weight scale across different training lengths.

Muon applies an update that respects matrix geometry, while vectors and scalars such as biases and norms stay on the mature Adam path. It is an engineering split by parameter structure. The lab does not claim the optimizer alone removes all loss spikes, and it does not present VRAM savings as the sole conclusion of this design.

### Deployment still has to hold 975B weights

41B active parameters describes the weights that take part in computing each token. It is not the capacity needed to load the checkpoint. As an idealized lower bound, 975B BF16 weights take about $975\times 10^9\times2$ bytes, which is about **1.95 TB**. The 4-bit weights alone are about **487.5 GB**, before scales, metadata, embeddings, KV cache and runtime workspace. That is the same order of magnitude as the lab's stated minimum aggregate VRAM of 2 TB for BF16 and 600 GB for NVFP4. MoE saves active compute. It does not remove the cost of weight storage or the bandwidth cost of expert routing.

## 6. Encoder-free multimodal still has input preprocessing

Inkling's vision and audio components are trained jointly from scratch. Audio uses dMel spectrograms, and images are cut into 40×40 patches, passed through a four-layer hMLP and a lightweight embedding, then handed to the main model together with the text tokens.

Inkling uses an **encoder-free multimodal architecture**: there is no large separate vision or audio encoder with late fusion. Images and audio still go through patching, spectrograms, the hMLP and embeddings before they reach the main model.

## Limits of the benchmark comparisons

Thinking Machines says plainly that Inkling is not the strongest open or closed model today. The lab emphasizes the capability profile, reasoning token efficiency, native text/image/audio, and customization through Tinker. Some evaluations use an internal harness, and external models may use self-reported numbers, so any cross-model comparison has to account for harness and checkpoint differences.

The lab says pretraining started from scratch. For post-training, the initial SFT still used synthetic data generated by open-weight models, Kimi K2.5 among them, and only afterwards was most of the compute put into large-scale RL. So I don't summarize it as "no distillation at all."

## What deployment will cost

975B only describes the size of the checkpoint. Actual compute and deployment cost also depend on the 5:1 attention, the 41B active parameters, expert routing, weight format and context length. Those say more about what the system really pays than total parameters alone.

[^inkling-release]: [Thinking Machines Lab, “Introducing Inkling”](https://thinkingmachines.ai/news/introducing-inkling/). The release post is the source for the public architecture overview and training-token count.
[^inkling-card]: [Inkling Model Card](https://thinkingmachines.ai/model-card/inkling/). Used for parameter, context, modality, training, and deployment-memory details.
[^inkling-day0]: [LMSYS, “Inkling Day-0 Support”](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support). Used only for the serving-system terminology and implementation notes.
