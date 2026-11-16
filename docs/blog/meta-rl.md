---
description: "Meta-RL as inference over tasks: recurrent policies, gradient adaptation, latent context, structured exploration, and the evaluation mistakes that hide memorization."
date: "2025-01-11"
updated: "2026-07-21"
language: "en"
image: "https://wayne.is-a.dev/assets/blog/meta-rl.webp"
tags:
  - Machine Learning
  - Reinforcement Learning
  - Survey
---

# Meta-RL: Teaching a Policy to Update Itself Within an Episode

*2025-01-11 · updated 2026-07-21 · reinforcement learning / adaptation*

<figure class="pb-article-hero">
  <img src="/assets/blog/meta-rl.webp" alt="DeepMind meta-reinforcement-learning experiment visualization" loading="eager" decoding="async">
  <figcaption>Prefrontal cortex as a meta-RL system · <a href="https://deepmind.google/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/">Source: Google DeepMind</a></figcaption>
</figure>

Ordinary reinforcement learning looks for one policy in a fixed MDP. Meta-RL adds a layer: the training data is a **task distribution**, and the agent has to infer from a few trajectories on a new task what the dynamics and reward are this time, then change its behavior using the experience within the episode. The weights may stay fixed, but the effective state of the policy has to update. That is what it means to put the learning algorithm inside the agent.

Three questions follow: where the task information is stored, how adaptation updates it, and how exploration helps identify the task.

## Problem setup: optimizing over a task distribution

Sample a task $\mathcal{T}$ from a distribution $p(\mathcal{T})$. The agent first collects context $c_{1:k}=(s,a,r,s')_{1:k}$, then an adaptation mechanism $A_\phi$ produces a task-conditioned policy:

$$
\pi_{\mathcal{T}} = A_\phi(c_{1:k}),
\qquad
\max_\phi\;
\mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}
\left[J_{\mathcal{T}}\!\left(A_\phi(c_{1:k})\right)\right].
\tag{1}
$$

Meta-training tunes $\phi$. It may be an initial set of weights that is easy to fine-tune, a recurrent state update, or an encoder that infers a latent task from context. Meta-testing keeps the new tasks outside the training distribution. Otherwise what you see is memorization, not adaptation.

## Three routes put the task belief in different places

| Route | Where adaptation happens | Strength | Main risk |
| --- | --- | --- | --- |
| Gradient-based (MAML) | A few gradient steps on the policy parameters | Explicit update rule, works with different models | Second-order gradient cost, RL gradient variance |
| Recurrent (RL²) | RNN hidden state absorbs the reward/action history | No weight updates at test time | Hidden state is hard to interpret and easily memorizes training tasks |
| Latent-context (PEARL / VariBAD) | Infer a latent task $z$ from the trajectory | Can represent uncertainty, suits belief-space reasoning | Posterior collapse, offline context shift |

### MAML: find a starting point that a few steps can fix

For each task, first do an inner-loop update

$$
\theta'_{\mathcal{T}} =
\theta-\alpha\nabla_\theta\mathcal{L}_{\mathcal{T}}(\theta),
\tag{2}
$$

and then the outer loop makes the updated $\theta'_{\mathcal{T}}$ good across all tasks. What MAML learns is a set of initial parameters that needs only a few gradient steps to adapt to a new task. The difficulty in RL is that both the inner and outer loops carry sampling noise, and the second-order term is expensive. First-order MAML and Reptile trade some exactness for a cheaper approximation.[^maml]

### RL²: hiding the update rule in the recurrent state

RL² feeds the previous action, reward, termination and the observation into an RNN together. The weights do not change at meta-test time, but the hidden state updates along the trajectory. From the outside it looks like the agent is running an RL algorithm that it learned itself, inside the episode.[^rl2]

An RNN can represent many update rules, but the hidden state is hard to interpret, and experiments struggle to tell whether the network is inferring the task, memorizing a task ID, or exploiting a fixed ordering in the benchmark. Task permutation, held-out dynamics and counterfactual reward tests therefore matter more than average return.

### PEARL / VariBAD: maintaining the task belief explicitly

PEARL infers a latent $z$ from context, and the policy acts on $(s,z)$. Sampling different $z$ also gives a form of posterior-sampling exploration.[^pearl] VariBAD writes the belief update of a Bayes-adaptive MDP as a variational inference problem, so the agent learns task inference and control together.[^varibad]

These methods make "I'm not sure yet which task this is" part of the model, but they are sensitive to the context distribution. If offline meta-RL trains only on contexts collected by the behavior policy, then at deployment the contexts produced by the new policy may land in regions the encoder has never seen.

## Exploration in meta-RL has to identify the task

Ordinary exploration looks for states that may bring high reward. Meta-RL also has to pick the action that best separates the candidate tasks. Even if two actions give the same immediate reward, one of them may make the posterior converge faster and so raise later returns.

MAESN injects episode-consistent structured noise in latent space, so a whole trajectory follows one consistent exploration strategy.[^maesn]
More generally, the value of exploration can be written as task information gain:

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

In practice you usually don't optimize this expression directly, but you can use it to check whether exploration generalizes. If the agent only knows what to try on the training tasks, exploration collapses once the reward mapping is swapped.

## Four problems that evaluations hide most easily

1. **Task leakage**: the observation, episode length or reset pattern hints at the task ID.
2. **Reporting only the return after adaptation**: without curves for trials 0, 1 and 2, you can't see the adaptation speed or the early exploration cost.
3. **Train and test tasks too similar**: varying continuous parameters over a small range can't separate interpolation from real structural generalization.
4. **Averages hide tail tasks**: the mean return is high, but a few dynamics shifts fail completely. At least report quantiles, worst-group results and seed variation too.

A credible report includes at least: how the held-out tasks were constructed, the return at each trial, the interaction budget used for adaptation, and comparisons against training from scratch, domain randomization and an oracle task-ID policy.

## How I would pick a method

- You already have a differentiable policy and a few update steps per new task are allowed: start from a first-order MAML-style baseline.
- The task information naturally lives in long trajectories and updating weights online is inconvenient: use a recurrent policy, but add strong leakage tests.
- You need uncertainty and information-seeking exploration: choose a latent-context or belief-based method, and include posterior calibration in the evaluation.
- You only have offline data: deal with context shift first. Without coverage, meta-learning will not create evidence for a new task out of nothing.

Evaluation comes back to one question in the end: what did the agent update in its judgment during the first few interactions on a new task? If the experiment can't answer that, even a high final return may only mean the task distribution was memorized.

[^maml]: [Finn et al., “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks”](https://proceedings.mlr.press/v70/finn17a.html).
[^rl2]: [Duan et al., “RL²: Fast Reinforcement Learning via Slow Reinforcement Learning”](https://arxiv.org/abs/1611.02779).
[^pearl]: [Rakelly et al., “Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables”](https://proceedings.mlr.press/v97/rakelly19a.html).
[^varibad]: [Zintgraf et al., “VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning”](https://jmlr.org/papers/v22/21-0657.html).
[^maesn]: [Gupta et al., “Meta-Reinforcement Learning of Structured Exploration Strategies”](https://proceedings.neurips.cc/paper/2018/hash/4de754248c196c85ee4fbdcee89179bd-Abstract.html).
