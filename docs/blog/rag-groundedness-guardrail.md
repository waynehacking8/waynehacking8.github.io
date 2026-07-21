---
description: "Why a 0% RAG hallucination result failed on adversarial near-misses, with paragraph-by-paragraph Traditional Chinese and results from nine gates."
date: "2026-05-31"
updated: "2026-07-21"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/rag-architecture.svg"
tags:
  - RAG
  - Evaluation
  - Grounding
  - LLM
---

# The 0% RAG result failed the harder test｜RAG 的 0% 結果沒有通過更難的測試

*2026-05-31 · updated 2026-07-21 · RAG / evaluation / grounding*

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/rag-architecture.svg" alt="Microsoft single-tenant RAG reference architecture" loading="eager" decoding="async">
  <figcaption>Grounded RAG reference architecture · Grounded RAG 參考架構 · <a href="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag">Source: Microsoft Learn</a></figcaption>
</figure>

My first evaluation made a tidy claim: a guarded prompt cut hallucination from **4/10 to 0/10** on unanswerable questions. The run was real, reproducible, and almost useless as a headline. Ten questions were too few, and “out of corpus” was the easy failure mode—the model had no relevant passage tempting it toward a plausible answer.[^repo]

第一次評估得到一個很整齊的結論：面對無法回答的問題，加上 guard prompt 後，hallucination 從 **4/10 降到 0/10**。結果是真的，也能重現，但幾乎不能拿來下標題。十題太少，而且「語料庫外」是容易的失敗模式；模型根本找不到相關段落，自然少了被誘導出合理錯答的機會。[^repo]
{ .pb-translation lang=zh-Hant }

The scaled rerun used 100 answerable and 100 adversarially unanswerable SQuAD 2.0 questions. Each near-miss question retrieves an on-topic passage that does *not* contain the answer. Here the guarded generator hallucinated on **48/100** questions; without the guard it failed on **78/100**.[^squad-report] The 0% result did not generalize. That reversal is the useful result.

擴大後的重跑使用 100 題可回答、100 題經對抗設計而無法回答的 SQuAD 2.0 問題。每個 near-miss 問題都會取回主題相關、卻不含答案的段落。這次有 guard 的 generator 在 **48/100** 題出現 hallucination，沒有 guard 時是 **78/100**。[^squad-report] 原本的 0% 無法泛化；這次反轉才是有用的結果。
{ .pb-translation lang=zh-Hant }

## The two evaluations were testing different problems｜兩次評估測的是不同問題

| Evaluation／評估 | Unanswerable set／無法回答資料集 | Guarded／有 guard | Unguarded／無 guard | What it measures／測量內容 |
| --- | ---: | ---: | ---: | --- |
| Initial harness／初始 harness | 10 out-of-corpus questions／10 題語料庫外問題 | 0/10 | 4/10 | Can the model abstain when evidence is plainly absent?／證據明顯不存在時，模型能否拒答？ |
| Scaled rerun／擴大重跑 | 100 adversarial near-misses／100 題對抗式 near-miss | 48/100 | 78/100 | Can it resist a relevant-looking passage that lacks the answer?／面對看似相關卻沒有答案的段落時，模型能否避免錯答？ |

The first result still describes those ten examples. It just cannot support “this guardrail solves hallucination.” The harder set changes both the point estimate and the mechanism: the context is no longer empty or irrelevant, so “answer only from context” can steer the model toward a wrong span instead of toward abstention.

第一個結果仍能描述那十題，卻不足以支持「這個 guardrail 解決了 hallucination」。較難的資料集同時改變估計值與失敗機制：context 不再是空白或不相關內容，「只能根據 context 回答」反而可能把模型推向錯誤 span，而不是拒答。
{ .pb-translation lang=zh-Hant }

## Define the failure before reporting the percentage｜回報百分比前先定義失敗

For the unanswerable slice, let $h_i=1$ when answer $i$ contains a claim unsupported by the retrieved passage. The reported hallucination rate is

在無法回答的資料切片中，若答案 $i$ 包含取回段落無法支持的主張，就令 $h_i=1$。Hallucination rate 定義如下：
{ .pb-translation lang=zh-Hant }

$$
\widehat{H}_{\text{unanswerable}} =
\frac{1}{N_{\text{unanswerable}}}
\sum_{i=1}^{N_{\text{unanswerable}}} h_i.
\tag{1}
$$

That denominator must travel with the percentage. A zero from $N=10$ and a zero from $N=1{,}000$ are not the same amount of evidence. The scaled report therefore includes Wilson 95% confidence intervals and paired exact McNemar tests where two gates score the same answers.[^squad-report]

百分比必須連同分母一起看。$N=10$ 的零和 $N=1{,}000$ 的零，證據強度完全不同。因此，擴大後的報告加入 Wilson 95% confidence intervals；兩個 gates 評分同一批答案時，也使用 paired exact McNemar tests。[^squad-report]
{ .pb-translation lang=zh-Hant }

Retrieval recall and hallucination rate also stay separate. An answer may retrieve the gold passage and still invent a claim; an unanswerable question may retrieve the intended near-miss perfectly and *still* require abstention.

Retrieval recall 與 hallucination rate 也要分開。即使取回 gold passage，答案仍可能捏造主張；無法回答的問題即使精準取回預定的 near-miss，仍然必須拒答。
{ .pb-translation lang=zh-Hant }

## Same-model validation shares the generator’s blind spots｜同模型驗證會共享 generator 的盲點

The original `validate()` step asked the same 8B model family to judge its own unguarded answers. It caught only one of four hallucinations in the small run. At scale, the plain self-judge reached **27% recall**. Switching to an independent 8B model family raised recall to **41%**, but 53 of 96 hallucinations escaped both judges.[^judge-report]

原本的 `validate()` 讓同一個 8B model family 判斷自己的 unguarded answers。小型實驗的四次 hallucination 只抓到一次；擴大後，plain self-judge 的 recall 是 **27%**。換成獨立的 8B model family 後，recall 升到 **41%**，但 96 次 hallucination 中仍有 53 次同時逃過兩個 judges。[^judge-report]
{ .pb-translation lang=zh-Hant }

| Gate on the same 200 answers／同一批 200 答案的 gate | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Same-family judge／同 family judge | 79% | 27% | 0.40 |
| Cross-family judge／跨 family judge | 74% | 41% | 0.52 |
| Same-family judge + chain of thought／同 family + CoT | 81% | 40% | 0.53 |
| Cross-family judge + chain of thought／跨 family + CoT | 62% | 49% | 0.55 |
| MiniCheck-FT5 770M | 66% | 36% | 0.47 |
| Cross-family judge **OR** MiniCheck／跨 family judge **或** MiniCheck | 63% | 55% | 0.59 |

The smallest model in that table gives the most interesting signal. MiniCheck is a 770M grounded NLI verifier: it compares the answer against evidence instead of deciding from its parametric memory. It recovered 12 of the 53 errors missed by both 8B judges—more blind spots than any generative-judge variant.[^minicheck][^judge-report]

表中最小的模型反而提供了最有意思的訊號。MiniCheck 是一個 770M grounded NLI verifier，會把答案和證據直接比對，不靠 parametric memory 判斷。兩個 8B judges 都漏掉的 53 個錯誤中，它補抓到 12 個，比任何 generative-judge variant 都多。[^minicheck][^judge-report]
{ .pb-translation lang=zh-Hant }

This is why “use a bigger judge” is not the default answer. A separate 70B experiment found no meaningful recall advantage over the cross-family 8B judge on its paired set, while the grounded verifier continued to catch errors the large parametric judge missed.[^repo]

因此，不能預設換成更大的 judge 就會改善結果。另一組 70B 實驗在 paired set 上，recall 並未明顯優於 cross-family 8B judge；grounded verifier 則繼續抓到大型 parametric judge 漏掉的錯誤。[^repo]
{ .pb-translation lang=zh-Hant }

## Better retrieval did not make abstention safer｜更好的 retrieval 沒有讓拒答更安全

Adding a cross-encoder reranker improved retrieval recall@3 from **85% to 96%** and substring answer accuracy from **70% to 81%**. Hallucination did not improve: guarded answers moved from 48% to 52%, and unguarded answers from 78% to 86%; neither change was statistically significant.[^rerank-report]

加入 cross-encoder reranker 後，retrieval recall@3 從 **85% 升到 96%**，substring answer accuracy 從 **70% 升到 81%**。Hallucination 沒有改善：guarded answers 從 48% 變成 52%，unguarded answers 從 78% 變成 86%；兩項變化都沒有統計顯著性。[^rerank-report]
{ .pb-translation lang=zh-Hant }

That result is easy to misread. The reranker worked—it surfaced more relevant passages. On an adversarially unanswerable question, however, a more relevant *answerless* passage gives the generator a better near-miss. Retrieval quality and abstention safety are different axes.

這個結果很容易被看反。Reranker 確實找到了更多相關段落；但對經對抗設計且無法回答的問題而言，更相關但沒有答案的段落，會提供更像答案的 near-miss。Retrieval quality 與 abstention safety 是兩條不同的軸。
{ .pb-translation lang=zh-Hant }

## What I would deploy from these results｜根據這些結果，我會怎麼部署

The measured gate with the best coverage is the union of two different failure detectors: block when either the cross-family judge or MiniCheck flags the answer. It reaches 55% recall on the common Phase-5 run, with 63% precision. That is not a solved system; 29 of 96 hallucinations still escaped every tested gate.

實測 coverage 最好的 gate 是兩種 failure detectors 的聯集：cross-family judge 或 MiniCheck 任一標記答案就阻擋。在共同的 Phase-5 run 上，recall 是 55%，precision 是 63%。這套系統仍未解決問題；96 次 hallucination 中，還有 29 次逃過所有測試過的 gates。
{ .pb-translation lang=zh-Hant }

So the production contract is explicit:

因此，production contract 要明確寫成以下五點：
{ .pb-translation lang=zh-Hant }

1. **Generate with an abstention instruction**, because it still cuts the hard-set failure rate by 30 points.<br><span class="pb-inline-translation" lang="zh-Hant">生成時加入 abstention instruction；它仍能讓 hard-set failure rate 降低 30 個百分點。</span>
2. **Verify against evidence**, not only with another generative model.<br><span class="pb-inline-translation" lang="zh-Hant">直接對照證據驗證，不能只加另一個 generative model。</span>
3. **Log the rejected and escaped cases** with retrieved spans so the next evaluation set is built from real failures.<br><span class="pb-inline-translation" lang="zh-Hant">連同 retrieved spans 記錄被擋下與逃脫的案例，讓下一版 evaluation set 來自真實失敗。</span>
4. **Report precision and recall**, not just the residual hallucination rate; an aggressive gate can look safe by blocking correct answers.<br><span class="pb-inline-translation" lang="zh-Hant">同時回報 precision 與 recall；只看 residual hallucination rate，可能把大量阻擋正確答案的 gate 誤認為安全。</span>
5. **Keep a deterministic adversarial set** and rerun paired comparisons whenever the model, retriever, prompt, or judge changes.<br><span class="pb-inline-translation" lang="zh-Hant">保留 deterministic adversarial set；model、retriever、prompt 或 judge 有變動時，重新執行 paired comparisons。</span>

The adversarial set changed the claim: the prompt guard reduced hallucination but did not eliminate it. Future evaluations should include relevant-looking passages that still lack the answer.

對抗資料集改寫了原本的結論：prompt guard 能降低 hallucination，卻無法消除它。後續評估必須加入看似相關、實際上仍不含答案的段落。
{ .pb-translation lang=zh-Hant }

→ Reproduce the runs, inspect row-level verdicts, and read the statistical caveats:<br><span class="pb-inline-translation" lang="zh-Hant">重現實驗、檢查逐列 verdict，並閱讀統計限制：</span>
[github.com/waynehacking8/nim-agent-blueprint](https://github.com/waynehacking8/nim-agent-blueprint)

[^repo]: [NIM Agent Blueprint](https://github.com/waynehacking8/nim-agent-blueprint), the primary repository containing the agent graph, datasets, committed row-level verdicts, reports, and reproduction commands.／主要 repo，包含 agent graph、資料集、逐列 verdict、報告與重現命令。
[^squad-report]: [SQuAD 2.0 evaluation report](https://github.com/waynehacking8/nim-agent-blueprint/blob/main/eval/report_squad.md), including the N=200 design, Wilson intervals, and paired comparisons.／包含 N=200 設計、Wilson intervals 與 paired comparisons。
[^judge-report]: [Judge-variant report](https://github.com/waynehacking8/nim-agent-blueprint/blob/main/eval/report_judge_variants.md), comparing same-family, cross-family, chain-of-thought, MiniCheck, and union gates on shared answers.／比較同 family、跨 family、CoT、MiniCheck 與 union gates。
[^rerank-report]: [Reranker ablation](https://github.com/waynehacking8/nim-agent-blueprint/blob/main/eval/report_rerank.md), with paired retrieval, accuracy, and hallucination results.／包含 paired retrieval、accuracy 與 hallucination 結果。
[^minicheck]: [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/abs/2404.10774), the grounded verifier used in the evaluation.／評估使用的 grounded verifier。
