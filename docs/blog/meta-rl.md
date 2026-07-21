---
description: "Meta-RL as inference over tasks: recurrent policies, gradient adaptation, latent context, structured exploration, and the evaluation mistakes that hide memorization."
date: "2025-01-11"
updated: "2026-07-21"
language: "zh-Hant"
image: "https://wayne.is-a.dev/assets/blog/meta-rl.webp"
tags:
  - Machine Learning
  - Reinforcement Learning
  - Survey
---

# Meta-RL：讓 policy 在 episode 內學會更新自己

*2025-01-11 · updated 2026-07-21 · reinforcement learning / adaptation*

<figure class="pb-article-hero">
  <img src="/assets/blog/meta-rl.webp" alt="DeepMind meta-reinforcement-learning experiment visualization" loading="eager" decoding="async">
  <figcaption>Prefrontal cortex as a meta-RL system · <a href="https://deepmind.google/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/">Source: Google DeepMind</a></figcaption>
</figure>

一般的強化學習在固定 MDP 裡找一個 policy。Meta-RL 多了一層：訓練資料是一個**任務
分布**，agent 要從新任務的少量 trajectory 推斷「這次的 dynamics／reward 到底是
什麼」，再用 episode 內的經驗改變行為。權重可以不更新，但 policy 的有效狀態必須
更新。這就是把學習演算法本身塞進 agent。

接下來要回答三件事：任務資訊存在哪裡、適應如何更新，以及探索怎麼協助辨識任務。

## 問題設定：在任務分布上最佳化

令任務 $\mathcal{T}$ 從分布 $p(\mathcal{T})$ 取樣。Agent 先收集 context
$c_{1:k}=(s,a,r,s')_{1:k}$，再透過適應機制 $A_\phi$ 產生任務條件 policy：

$$
\pi_{\mathcal{T}} = A_\phi(c_{1:k}),
\qquad
\max_\phi\;
\mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}
\left[J_{\mathcal{T}}\!\left(A_\phi(c_{1:k})\right)\right].
\tag{1}
$$

Meta-training 調整的是 $\phi$：它可能是一組容易 fine-tune 的初始權重、一個 recurrent
state update，或一個從 context 推斷 latent task 的 encoder。Meta-test 則把新任務保留
在訓練分布之外，否則看到的只是記憶，不是適應。

## 三條路線把 task belief 放在不同位置

| 路線 | 適應發生的位置 | 優點 | 主要風險 |
| --- | --- | --- | --- |
| Gradient-based（MAML） | 對 policy 參數做少量 gradient steps | 更新規則明確，可套在不同模型 | 二階梯度成本、RL gradient variance |
| Recurrent（RL²） | RNN hidden state 吸收 reward/action history | 測試時不用更新權重 | hidden state 難解釋，容易記住訓練任務 |
| Latent-context（PEARL / VariBAD） | 從 trajectory 推斷 latent task $z$ | 能表示不確定性，適合 belief-space reasoning | posterior collapse、offline context shift |

### MAML：找一個「幾步就能改好」的起點

對每個任務先做 inner-loop 更新

$$
\theta'_{\mathcal{T}} =
\theta-\alpha\nabla_\theta\mathcal{L}_{\mathcal{T}}(\theta),
\tag{2}
$$

外層再讓更新後的 $\theta'_{\mathcal{T}}$ 在各任務上都好。MAML 要學的是一組只需少量
gradient steps 就能適應新任務的初始參數。RL 裡的難點是 inner/outer loop
都帶 sampling noise，二階項也昂貴；first-order MAML 和 Reptile 用較便宜的近似換掉
部分精確度。[^maml]

### RL²：把更新規則藏進 recurrent state

RL² 把前一步 action、reward、termination 與 observation 一起餵進 RNN。權重在
meta-test 不變，但 hidden state 隨 trajectory 更新；對外看起來就像 agent 在 episode
內執行一個自己學會的 RL algorithm。[^rl2]

RNN 能表示多種更新規則，但 hidden state 很難解釋，實驗也難以分辨網路是在推斷任務、
記住 task ID，還是利用 benchmark 的固定順序。任務 permutation、held-out dynamics
與 counterfactual reward tests 因此比平均 return 更重要。

### PEARL / VariBAD：顯式維護 task belief

PEARL 從 context 推斷 latent $z$，policy 以 $(s,z)$ 決策；採樣不同 $z$ 也提供一種
posterior sampling 式探索。[^pearl] VariBAD 則把 Bayes-adaptive MDP 的 belief update
寫成 variational inference 問題，讓 agent 同時學 task inference 與 control。[^varibad]

這類方法讓「我還不確定是哪個任務」成為模型的一部分，但 context distribution 很
敏感。Offline meta-RL 若只在行為 policy 收集的 context 上訓練，部署時由新 policy
產生的 context 可能落在 encoder 沒看過的區域。

## Meta-RL 的探索要辨識任務

普通 exploration 尋找可能帶來高 reward 的 state。Meta-RL 還要選出最能區分候選任務的
action；即使兩個 action 的即時 reward 相同，其中一個仍可能讓 posterior 更快收斂，進而
提高後續回報。

MAESN 在 latent space 注入 episode-consistent structured noise，讓整段 trajectory 形成
一致的探索策略。[^maesn]
更一般地，可以把探索價值寫成 task information gain：

$$
\text{IG}(a_t)=
\mathbb{E}\left[
D_{\mathrm{KL}}\!\left(
p(\mathcal{T}\mid c_{1:t+1})\,\|\,
p(\mathcal{T}\mid c_{1:t})
\right)
\right].
\tag{3}
$$

實作通常不會直接最佳化這個式子，但可用它檢查探索是否能泛化：如果 agent 只在
training tasks 上知道該試什麼，換掉 reward mapping 後探索就會崩。

## 評估時最容易藏住的四個問題

1. **Task leakage**：observation、episode length 或 reset pattern 暗示 task ID。
2. **只報 adaptation 後的 return**：沒有畫第 0、1、2 次 trial 的曲線，看不到適應
   速度與前期探索成本。
3. **Train/test 任務太相似**：只換連續參數的小範圍，無法區分 interpolation 與真正
   的結構泛化。
4. **平均值掩蓋 tail tasks**：mean return 很高，但少數 dynamics shift 完全失敗；至少
   應同時報 quantile、worst-group 與 seed variation。

一個可信的報告至少包含：held-out task construction、每個 trial 的 return、適應所用
interaction budget、與從頭訓練／domain randomization／oracle task-ID policy 的對照。

## 我會怎麼選方法

- 已有可微分 policy、每個新任務允許幾步更新：先從 first-order MAML 類 baseline 開始。
- 任務資訊天然存在長 trajectory，而且線上更新權重不方便：用 recurrent policy，但加
  強力的 leakage tests。
- 需要不確定性與 information-seeking exploration：選 latent-context / belief-based
  方法，並把 posterior calibration 納入評估。
- 只有離線資料：先處理 context shift；沒有 coverage，meta-learning 不會憑空創造新
  任務的證據。

評估最後仍要回到一件事：agent 在新任務的前幾次互動中更新了哪些判斷。若實驗無法
回答，再高的 final return 仍可能只是記住了任務分布。

[^maml]: [Finn et al., “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks”](https://proceedings.mlr.press/v70/finn17a.html).
[^rl2]: [Duan et al., “RL²: Fast Reinforcement Learning via Slow Reinforcement Learning”](https://arxiv.org/abs/1611.02779).
[^pearl]: [Rakelly et al., “Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables”](https://proceedings.mlr.press/v97/rakelly19a.html).
[^varibad]: [Zintgraf et al., “VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning”](https://jmlr.org/papers/v22/21-0657.html).
[^maesn]: [Gupta et al., “Meta-Reinforcement Learning of Structured Exploration Strategies”](https://proceedings.neurips.cc/paper/2018/hash/4de754248c196c85ee4fbdcee89179bd-Abstract.html).
