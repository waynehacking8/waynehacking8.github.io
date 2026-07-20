---
description: "Federated learning from scratch (FedAvg / FedProx / SCAFFOLD), what breaks under Non-IID data, and how DP-SGD and secure aggregation add real privacy guarantees."
date: "2026-05-31"
updated: "2026-07-21"
language: "en"
image: "https://wayne.is-a.dev/assets/blog/federated-dp.webp"
tags:
  - Privacy
  - Federated Learning
  - Differential Privacy
---

# Notes on Federated Learning and Differential Privacy

*2026-05-31 · privacy-preserving ML*

<figure class="pb-article-hero">
  <img src="/assets/blog/federated-dp.webp" alt="Google Research 聯邦學習與差分隱私實驗圖" loading="eager" decoding="async">
  <figcaption>Federated learning with formal DP guarantees · <a href="https://research.google/blog/federated-learning-with-formal-differential-privacy-guarantees/">Source: Google Research</a></figcaption>
</figure>

Working notes on building federated learning (FL) from scratch, what actually breaks under
**Non-IID** data, and how **differential privacy (DP)** and **secure aggregation** fit on top —
including the honest negative results that the marketing slides leave out. They follow the
implementation in
[**federated-learning-lab**](https://github.com/waynehacking8/federated-learning-lab)[^fl-repo]
(FedAvg / FedProx / SCAFFOLD, DP-SGD, secure aggregation; 33/33 tests, literature
cross-validated).

## 1. What federated learning actually is

The data never moves. Instead of pooling everyone's data on one server, each client trains
locally and sends **model updates** to a server that aggregates them. The canonical loop
(**FedAvg**) is:

1. Server broadcasts the global model.
2. Each client does a few local SGD epochs on its own data.
3. Each client sends back its updated weights.
4. Server averages the weights (weighted by client data size) → new global model.[^fedavg]

That's it. The elegance is that raw data stays on-device; the difficulty is that the clients'
data distributions are **not** identical.

## 2. The Non-IID problem (where FedAvg starts to hurt)

FedAvg implicitly assumes every client sees roughly the same distribution. Real clients don't —
one hospital sees different cases than another, one phone's keyboard sees different language.
Under **Non-IID** data, each client's local optimum pulls in a different direction, so averaging
their updates produces **client drift**: the global model lands somewhere none of them wanted.

Two well-known fixes, both implemented and measured in the lab:

- **FedProx** — add a proximal term that penalises drifting too far from the global model.[^fedprox]
  Stabilises training when clients are heterogeneous.
- **SCAFFOLD** — track **control variates** (correction terms) that estimate and subtract the
  drift direction. More state to communicate, but corrects the bias FedProx only damps.[^scaffold]

The honest finding worth repeating: on a strongly Non-IID split (e.g. label-skewed MNIST), the
fancy methods **don't always beat plain FedAvg by much** — and sometimes the dominant lever is
just more communication rounds. Reporting the case where your method *doesn't* win is what
separates a lab from a brochure.

## 3. Differential privacy: the model still leaks

Keeping data on-device is **not** privacy. Model updates leak information about the data that
produced them — membership inference and gradient-inversion attacks reconstruct training samples
from gradients. To get a real guarantee you add **differential privacy**.

**DP-SGD** makes each training step private by:[^dpsgd]

1. **Per-sample gradient clipping** — bound each example's contribution to a max norm `C`.
2. **Gaussian noise** — add noise calibrated to `C` to the summed gradients.

The result is a formal **(ε, δ)** guarantee: the trained model is provably almost the same
whether or not any single example was in the data. The cost is the **privacy–utility
trade-off** — smaller ε (stronger privacy) means more noise and lower accuracy. There is no
free lunch; the contribution is *measuring* the curve, not claiming privacy is costless.

## 4. Secure aggregation: hide the individual update

DP bounds what the *final model* leaks. **Secure aggregation** addresses a different threat: a
curious server seeing each client's *individual* update. With secure aggregation, clients mask
their updates so the server can compute only the **sum** — no single client's contribution is
visible — yet the masks cancel in aggregate. DP (what the model leaks) and secure aggregation
(what the server sees) are **complementary**, not substitutes.[^secagg]

## 5. Why "from scratch" and "33/33 tests"

Privacy ML is exactly the domain where a subtly wrong implementation gives a *false* sense of
safety — a clipping bug or a miscalibrated noise multiplier silently voids the ε guarantee. So
the lab:

- implements each algorithm from scratch (FedAvg / FedProx / SCAFFOLD, plus FedPer /
  Byzantine-robust / FedAdam / FedLoRA),
- **cross-validates against the literature** so behaviour matches published results, and
- ships **33/33 passing tests** and explicit negative results.

For privacy and security work, the test suite and the reproduction *are* the credibility.

## Takeaway

Federated learning moves the model, not the data — but on-device ≠ private. Non-IID data breaks
naive averaging (FedProx/SCAFFOLD help, sometimes only a little); DP-SGD buys a formal (ε, δ)
guarantee at a measurable accuracy cost; secure aggregation hides individual updates from the
server. The trustworthy version of all three is the one with the tests and the honest curves.

→ From-scratch implementations, tests, and negative results:
[github.com/waynehacking8/federated-learning-lab](https://github.com/waynehacking8/federated-learning-lab)

[^fl-repo]: [Federated Learning Lab](https://github.com/waynehacking8/federated-learning-lab), the primary artifact for implementations, tests, and measured negative results discussed here.
[^fedavg]: [McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data”](https://proceedings.mlr.press/v54/mcmahan17a.html).
[^fedprox]: [Li et al., “Federated Optimization in Heterogeneous Networks”](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html).
[^scaffold]: [Karimireddy et al., “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning”](https://proceedings.mlr.press/v119/karimireddy20a.html).
[^dpsgd]: [Abadi et al., “Deep Learning with Differential Privacy”](https://dl.acm.org/doi/10.1145/2976749.2978318).
[^secagg]: [Bonawitz et al., “Practical Secure Aggregation for Privacy-Preserving Machine Learning”](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/).
