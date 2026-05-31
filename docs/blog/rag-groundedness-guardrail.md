---
description: "One guardrail that drops a RAG agent's out-of-corpus hallucination rate from ~50% to 0% — and why an eval harness is the only way to see the difference."
---

# 0 % vs 50 %: making a RAG agent refuse to hallucinate

*2026-05-31 · LLM / RAG*

A retrieval-augmented agent is only as trustworthy as its behaviour on questions whose answer
**isn't in the corpus**. The failure mode is quiet: instead of saying "I don't know," the model
invents a confident, well-formed, wrong answer. This post shows a single guardrail that takes
that from common to never — and, crucially, *measures* it.

Reference architecture:
[**nim-agent-blueprint**](https://github.com/waynehacking8/nim-agent-blueprint) — agentic RAG on
the NVIDIA NIM stack with a built-in eval harness.

## The ablation

The agent loop is **plan → retrieve → generate → validate**. The interesting variable is the
generation prompt's contract with the retrieved context:

| Configuration | Out-of-corpus hallucination rate |
| --- | --- |
| Generate freely from context | **~50 %** |
| Guarded prompt (answer *only* from context; otherwise abstain) | **0 %** |

Same model, same retriever, same questions. The only change is a prompt that makes "I can't
answer that from the provided sources" a first-class, rewarded output — plus a **validate**
step that checks the answer is grounded in retrieved spans before returning it. On in-corpus
questions, retrieval **recall@3 stayed at 94–100 %**, so the guardrail buys safety without
costing coverage.

## Why "just prompt better" isn't the lesson

The lesson isn't the prompt — it's that the difference between 50 % and 0 % is **invisible
without an eval harness**. A demo that only asks in-corpus questions looks perfect in both
configurations. You only see the 50 % when you deliberately ask things the corpus can't
answer and *score groundedness*. So the blueprint ships with:

- **retrieval hit-rate** (is the answer even retrievable?),
- **answer groundedness** via LLM-as-judge (is the answer supported by what was retrieved?),
- **latency**, and OpenTelemetry traces per agent step.

That's the difference between "it works on my five questions" and "here is the number a
partner can hold me to."

## Takeaway

For enterprise RAG, abstention is a feature, not a failure. Make "I don't know" a rewarded
output, validate groundedness before returning, and **measure the out-of-corpus rate** — it's
the number that separates a demo from something you'd put in front of a customer.

→ Runnable blueprint + eval harness:
[github.com/waynehacking8/nim-agent-blueprint](https://github.com/waynehacking8/nim-agent-blueprint)
