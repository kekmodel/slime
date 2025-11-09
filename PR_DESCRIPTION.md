# Add CISPO (Clipped IS-weight Policy Optimization)

## Summary

Add support for CISPO (Clipped IS-weight Policy Optimization) algorithm introduced in the [MiniMax-M1 paper](https://arxiv.org/abs/2506.13585).

## Background

CISPO addresses a critical limitation in PPO/GRPO where low-probability reasoning tokens (e.g., "However," "Recheck," "Wait") are clipped out after the first on-policy update. As stated in the paper:

> "These low-probability tokens, however, are often crucial for stabilizing entropy and facilitating scalable RL."

CISPO solves this by clipping the importance sampling weight instead of token updates, preserving gradient contributions from all tokens.

**CISPO Loss Function:**

$$\mathcal{L}^{\text{CISPO}}(\theta) = -\mathbb{E}_{(s_i, a_i) \sim \mathcal{D}} \left[ \sum_{t=1}^{T_i} \text{sg}(\hat{r}_{i,t}(\theta)) \cdot A_i \cdot \log \pi_\theta(a_{i,t} | s_{i,<t}) \right]$$

where:
- $\hat{r}\_{i,t}(\theta)$ = clip( $r_{i,t}(\theta)$, $1 - \epsilon_{\text{low}}^{\text{IS}}$, $1 + \epsilon_{\text{high}}^{\text{IS}}$ ) is the clipped importance sampling ratio
- $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} | s_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} | s_{i,<t})}$ is the per-token importance sampling ratio
- $A_i$ is the advantage estimate for trajectory $i$
- sg(·) denotes stop-gradient operation

**Key Features:**
- **Token-level IS**: Clips importance sampling weights per token (Equation 4)
- **Practical clipping**: As stated in the paper: "In our experiments, we did not impose a lower bound on the IS weight by setting $\epsilon_{\text{low}}^{\text{IS}}$ to a large value; instead, we only tuned $\epsilon_{\text{high}}^{\text{IS}}$."
- **Stop-gradient**: Applies sg(·) to clipped ratios, preserving gradients through log probabilities

## Changes

- **`slime/utils/ppo_utils.py`**: Add `compute_cispo_loss()` function
- **`slime/utils/arguments.py`**: Add `'cispo'` to `advantage_estimator` choices
- **`slime/ray/rollout.py`**: Add CISPO to reward normalization
- **`slime/backends/megatron_utils/loss.py`**: Use CISPO loss when `advantage_estimator='cispo'`

## Implementation Details

- **Separate loss function required**: Unlike GRPO/GSPO (which only differ in advantage estimation), CISPO requires a distinct loss computation due to stop-gradient and explicit log probability usage
- **Token-level IS**: Uses per-token importance sampling ratios (not sequence-level)
- **Stop-gradient on IS ratio**: `ratio.detach()` prevents gradient flow through the ratio
- **Explicit log probability**: Uses `ratio_sg * advantages * log_probs` instead of `ratio * advantages`
  - This ensures gradient flows through $\log \pi_\theta$ only, matching Equation 4: sg( $\hat{r}\_{i,t}$ ) · $A_{i,t}$ · log $\pi_\theta$
- **Upper-only clipping**: Lower bound not imposed in practice (only $\epsilon_{\text{high}}^{\text{IS}}$ is tuned)
- **Default eps_clip_high**: 5.0 (based on ScaleRL paper analysis)

## Testing

Tested on GSM8K with Qwen3-0.6B model.

**WandB Run**: https://wandb.ai/kekmodel/slime-cispo-test/runs/nf0gh60p?nw=nwuserkekmodel

Configuration:
- Model: Qwen3-0.6B
- Dataset: GSM8K
- Settings: Mean-centering (--disable-grpo-std-normalization)
- eps_clip_high: 5.0

## References

- **MiniMax-M1 Paper**: [Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
  - Section 3.1: CISPO algorithm definition and motivation
  - Equation 4: CISPO loss function with token-level IS and stop-gradient
  - Equation 5: Clipped IS weight definition

- **ScaleRL Paper**: [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786)
  - Section 3.2: Empirical comparison showing CISPO outperforms DAPO and GSPO
  - Demonstrates CISPO's robustness to IS-clipping parameter choices
