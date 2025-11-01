# slime Analysis Documents

This directory contains research, analysis, and documentation for implementing RL algorithms and optimization techniques in slime.

## 📚 Contents

### CISPO Implementation

- **[CISPO_PAPER_REVIEW.md](CISPO_PAPER_REVIEW.md)** (16KB)
  - Comprehensive review of MiniMax-M1's CISPO algorithm
  - Comparison with PPO, GRPO, DAPO
  - slime implementation verification
  - Test setup and validation metrics
  - **Key findings**: Sequence-level IS, stop-gradient mechanism, FP32 LM head

- **[TESTING_CISPO.md](TESTING_CISPO.md)** (4.8KB)
  - Testing guide for CISPO implementation
  - First-step validation checklist
  - Metrics to monitor

### GLM-4.5 Production Implementation

- **[GLM4.5_PAPER_REVIEW.md](GLM4.5_PAPER_REVIEW.md)** (NEW!)
  - Comprehensive review of GLM-4.5 (355B MoE, 32B activated)
  - **Uses slime framework** for RL training
  - GRPO with mean-centering (no KL term, no Z-Score)
  - FP8 inference + BF16 training strategy
  - Outcome supervision for agentic tasks
  - Dynamic temperature scheduling (향후 구현 가능)
  - **Key validation**: Mean-centering choice matches binary reward analysis
  - Performance: AIME 24 (91.0%), MATH-500 (98.2%), SWE-bench (64.2%)

### Binary Reward Analysis

- **[BINARY_REWARD_ANALYSIS.md](BINARY_REWARD_ANALYSIS.md)** (12KB)
  - Mathematical analysis of Mean-Centering vs Z-Score normalization
  - Comparison across 9 success rate scenarios (batch size 32)
  - Recommendations for binary reward tasks (GSM8K, MATH, Coding)
  - **Key finding**: Dr. GRPO (mean-centering) is more stable and efficient for binary rewards
  - **Validated by GLM-4.5**: Production model uses same approach

- **[analyze_binary_reward.py](analyze_binary_reward.py)** (5.5KB)
  - Python script for advantage comparison analysis
  - Generates tables and detailed case studies
  - Run: `python analyze_binary_reward.py`

### MiniMax-M1 Research

- **[MINIMAX_M1_REWARD_DESIGN.md](MINIMAX_M1_REWARD_DESIGN.md)** (27KB)
  - In-depth analysis of MiniMax-M1's reward design
  - Multi-stage training pipeline
  - Rule-based and model-based rewards
  - Verifier strategies for math/coding tasks

- **[MINIMAX_M1_VS_M2_COMPARISON.md](MINIMAX_M1_VS_M2_COMPARISON.md)** (15KB)
  - Comparison between MiniMax-M1 and M2 models
  - Architecture differences and performance analysis

---

## 🔑 Key Recommendations

### For Binary Reward Tasks (GSM8K, MATH, Coding)

```bash
--advantage-estimator cispo
--disable-grpo-std-normalization  # Dr. GRPO (mean-centering)
--attention-softmax-in-fp32       # FP32 precision for stability
--accumulate-allreduce-grads-in-fp32
--eps-clip-high 5.0
```

**Why?**
- Mean-centering is more stable than Z-Score for binary rewards
- Prevents 2.3-3x gradient amplification at extreme success rates
- Natural difficulty weighting (harder problems get more learning signal)
- Megatron automatically uses FP32 for LM head log-probs

### For Continuous Reward Tasks

```bash
--advantage-estimator cispo
# Use default Z-Score normalization
--attention-softmax-in-fp32
--accumulate-allreduce-grads-in-fp32
--eps-clip-high 5.0
```

**Why?**
- Z-Score normalizes different reward scales
- Standard practice for continuous reward distributions
- Megatron automatically uses FP32 for LM head log-probs

---

## 📊 Implementation Status

### Core RL Algorithms

| Feature | MiniMax-M1 | GLM-4.5 | slime | Status |
|---------|------------|---------|-------|--------|
| CISPO algorithm | ✅ | ❌ (uses GRPO) | ✅ | Verified |
| GRPO (no KL) | ❌ | ✅ | ✅ | Verified |
| Sequence-level IS | ✅ | ✅ | ✅ | Verified |
| Stop-gradient | ✅ (CISPO) | N/A | ✅ | Verified |
| **Mean-centering** | ❌ (uses Z-Score) | ✅ | ✅ | **Production-proven** |
| FP32 LM head | ✅ | ✅ | ✅ | Megatron built-in |
| FP8 inference | ❌ | ✅ | ✅ | Implemented |

### Advanced Features

| Feature | GLM-4.5 | slime | Status |
|---------|---------|-------|--------|
| Outcome supervision | ✅ | ✅ | `loss_mask` support |
| Colocated mode | ✅ | ✅ | Implemented |
| Disaggregated mode | ✅ | ✅ | Implemented |
| **Dynamic temperature** | ✅ | ❌ | 향후 구현 |
| Token-weighted loss | ✅ | ⚠️ | 확인 필요 |
| Iterative distillation | ✅ | ✅ | Offline retraining |

---

## 🔗 References

### Papers

- **GLM-4.5 Paper**: https://arxiv.org/html/2508.06471
  - 355B MoE, uses slime framework
  - Mean-centering validation
  - AIME 24: 91.0%, MATH-500: 98.2%

- **MiniMax-M1 Paper**: https://arxiv.org/html/2506.13585v1
  - 456B MoE, CISPO algorithm
  - Z-Score normalization (theoretical comparison)

- **Dr. GRPO Paper**: https://arxiv.org/pdf/2503.20783
  - Mean-centering for binary rewards
  - Theoretical foundation

### slime Implementation

- **Core RL**:
  - `slime/utils/ppo_utils.py` (CISPO loss)
  - `slime/ray/rollout.py` (reward normalization)
  - `slime/backends/megatron_utils/loss.py` (advantage computation)

- **Infrastructure**:
  - `slime/router/` (SGLang routing)
  - `slime/rollout/` (custom generation functions)

---

**Last Updated**: 2025-11-01
**Maintainer**: Claude Code
**Latest Addition**: GLM-4.5 Production Review
