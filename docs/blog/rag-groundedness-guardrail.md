---
description: "Why a 0% RAG hallucination result failed on adversarial near-misses, what nine gates actually caught, and why grounding beats a larger judge."
date: "2026-05-31"
updated: "2026-07-21"
language: "en"
image: "https://wayne.is-a.dev/assets/blog/rag-architecture.svg"
tags:
  - RAG
  - Evaluation
  - Grounding
  - LLM
---

# The 0% RAG result failed the harder test

*2026-05-31 · updated 2026-07-21 · RAG / evaluation / grounding*

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/rag-architecture.svg" alt="Microsoft single-tenant RAG reference architecture" loading="eager" decoding="async">
  <figcaption>Grounded RAG reference architecture · <a href="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag">Source: Microsoft Learn</a></figcaption>
</figure>

My first evaluation made a tidy claim: a guarded prompt cut hallucination from **4/10 to
0/10** on unanswerable questions. The run was real, reproducible, and almost useless as a
headline. Ten questions were too few, and “out of corpus” was the easy failure mode—the
model had no relevant passage tempting it toward a plausible answer.[^repo]

The scaled rerun used 100 answerable and 100 adversarially unanswerable SQuAD 2.0 questions.
Each near-miss question retrieves an on-topic passage that does *not* contain the answer.
Here the guarded generator hallucinated on **48/100** questions; without the guard it failed
on **78/100**.[^squad-report] The 0% result did not generalize. That reversal is the useful
result.

## The two evaluations were testing different problems

| Evaluation | Unanswerable set | Guarded | Unguarded | What it measures |
| --- | ---: | ---: | ---: | --- |
| Initial harness | 10 out-of-corpus questions | 0/10 | 4/10 | Can the model abstain when evidence is plainly absent? |
| Scaled rerun | 100 adversarial near-misses | 48/100 | 78/100 | Can it resist a relevant-looking passage that lacks the answer? |

The first result still describes those ten examples. It just cannot support “this guardrail
solves hallucination.” The harder set changes both the point estimate and the mechanism: the
context is no longer empty or irrelevant, so “answer only from context” can steer the model
toward a wrong span instead of toward abstention.

## Define the failure before reporting the percentage

For the unanswerable slice, let $h_i=1$ when answer $i$ contains a claim unsupported by the
retrieved passage. The reported hallucination rate is

$$
\widehat{H}_{\text{unanswerable}} =
\frac{1}{N_{\text{unanswerable}}}
\sum_{i=1}^{N_{\text{unanswerable}}} h_i.
\tag{1}
$$

That denominator must travel with the percentage. A zero from $N=10$ and a zero from
$N=1{,}000$ are not the same amount of evidence. The scaled report therefore includes Wilson
95% confidence intervals and paired exact McNemar tests where two gates score the same
answers.[^squad-report]

Retrieval recall and hallucination rate also stay separate. An answer may retrieve the gold
passage and still invent a claim; an unanswerable question may retrieve the intended
near-miss perfectly and *still* require abstention.

## Same-model validation shares the generator’s blind spots

The original `validate()` step asked the same 8B model family to judge its own unguarded
answers. It caught only one of four hallucinations in the small run. At scale, the plain
self-judge reached **27% recall**. Switching to an independent 8B model family raised recall
to **41%**, but 53 of 96 hallucinations escaped both judges.[^judge-report]

| Gate on the same 200 answers | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Same-family judge | 79% | 27% | 0.40 |
| Cross-family judge | 74% | 41% | 0.52 |
| Same-family judge + chain of thought | 81% | 40% | 0.53 |
| Cross-family judge + chain of thought | 62% | 49% | 0.55 |
| MiniCheck-FT5 770M | 66% | 36% | 0.47 |
| Cross-family judge **OR** MiniCheck | 63% | 55% | 0.59 |

The smallest model in that table gives the most interesting signal. MiniCheck is a 770M
grounded NLI verifier: it compares the answer against evidence instead of deciding from its
parametric memory. It recovered 12 of the 53 errors missed by both 8B judges—more blind spots
than any generative-judge variant.[^minicheck][^judge-report]

This is why “use a bigger judge” is not the default answer. A separate 70B experiment found
no meaningful recall advantage over the cross-family 8B judge on its paired set, while the
grounded verifier continued to catch errors the large parametric judge missed.[^repo]

## Better retrieval did not make abstention safer

Adding a cross-encoder reranker improved retrieval recall@3 from **85% to 96%** and substring
answer accuracy from **70% to 81%**. Hallucination did not improve: guarded answers moved
from 48% to 52%, and unguarded answers from 78% to 86%; neither change was statistically
significant.[^rerank-report]

That result is easy to misread. The reranker worked—it surfaced more relevant passages. On an
adversarially unanswerable question, however, a more relevant *answerless* passage gives the
generator a better near-miss. Retrieval quality and abstention safety are different axes.

## What I would deploy from these results

The measured gate with the best coverage is the union of two different failure detectors:
block when either the cross-family judge or MiniCheck flags the answer. It reaches 55% recall
on the common Phase-5 run, with 63% precision. That is not a solved system; 29 of 96
hallucinations still escaped every tested gate.

So the production contract is explicit:

1. **Generate with an abstention instruction**, because it still cuts the hard-set failure
   rate by 30 points.
2. **Verify against evidence**, not only with another generative model.
3. **Log the rejected and escaped cases** with retrieved spans so the next evaluation set is
   built from real failures.
4. **Report precision and recall**, not just the residual hallucination rate; an aggressive
   gate can look safe by blocking correct answers.
5. **Keep a deterministic adversarial set** and rerun paired comparisons whenever the model,
   retriever, prompt, or judge changes.

The lesson is no longer “one prompt gets to zero.” It is that easy abstention tests can make a
weak guardrail look finished. The useful evaluation is the one designed to break the story
you most want to tell.

→ Reproduce the runs, inspect row-level verdicts, and read the statistical caveats:
[github.com/waynehacking8/nim-agent-blueprint](https://github.com/waynehacking8/nim-agent-blueprint)

[^repo]: [NIM Agent Blueprint](https://github.com/waynehacking8/nim-agent-blueprint), the primary repository containing the agent graph, datasets, committed row-level verdicts, reports, and reproduction commands.
[^squad-report]: [SQuAD 2.0 evaluation report](https://github.com/waynehacking8/nim-agent-blueprint/blob/main/eval/report_squad.md), including the N=200 design, Wilson intervals, and paired comparisons.
[^judge-report]: [Judge-variant report](https://github.com/waynehacking8/nim-agent-blueprint/blob/main/eval/report_judge_variants.md), comparing same-family, cross-family, chain-of-thought, MiniCheck, and union gates on shared answers.
[^rerank-report]: [Reranker ablation](https://github.com/waynehacking8/nim-agent-blueprint/blob/main/eval/report_rerank.md), with paired retrieval, accuracy, and hallucination results.
[^minicheck]: [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/abs/2404.10774), the grounded verifier used in the evaluation.
