# CISPO Testing Guide (H100 Single GPU)

This guide explains how to test the CISPO implementation on a remote server with 1x H100 GPU.

## Prerequisites

### 1. Server Setup
```bash
# Clone your fork
git clone https://github.com/kekmodel/slime.git
cd slime

# Checkout the dev branch (contains test scripts)
git checkout dev

# Or checkout feature/add-cispo for clean CISPO-only implementation
# git checkout feature/add-cispo
```

### 2. Environment Setup

Follow the main README for installation, or use Docker:

```bash
# Using Docker (recommended)
docker pull slimerl/slime:latest
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  slimerl/slime:latest bash
```

### 3. Download Test Data

```bash
# Download dapo-math-17k dataset
# Adjust path in test_cispo.sh if needed
huggingface-cli download --repo-type dataset \
  THUDM/dapo-math-17k \
  --local-dir dapo-math-17k
```

## Running the Test

### Quick Test (Recommended First)

```bash
# Make script executable
chmod +x tests/test_cispo.sh

# Run the test
bash tests/test_cispo.sh
```

**Expected Duration:** ~10-15 minutes for 2 rollouts with Qwen2.5-0.5B

### What the Test Does

1. **Cleanup**: Kills previous Ray/SGLang processes
2. **Launches Ray**: Single GPU cluster
3. **Runs Training**:
   - 2 rollouts for quick validation
   - Batch size 2 (reduced for single GPU)
   - 4 samples per prompt
   - CISPO with eps_clip_high=5.0

## Validation Checklist

### Critical Metrics (First Step Must Pass)

Check the logs for **rollout_id=0, step=0**:

```python
# These MUST be exactly 0 on the first training step
assert train/ppo_kl == 0.0
assert train/pg_clipfrac == 0.0
assert train/kl_loss == 0.0  # if using --use-kl-loss
```

**Why this matters:** This validates that the recomputed log probabilities match the rollout exactly, proving correctness.

### Monitor These Metrics

```bash
# Watch for these in the logs
train/loss          # Should decrease
train/pg_loss       # Policy gradient loss
train/ppo_kl        # KL divergence (0 on first step!)
train/pg_clipfrac   # Fraction of ratios clipped (CISPO specific)
train/entropy_loss  # Entropy of the policy
```

### CISPO-Specific Checks

1. **Ratio Truncation**: Check that `pg_clipfrac` is non-zero when ratios > 5.0
2. **Stop-Gradient**: Loss should still backpropagate (check grad norms)
3. **Sequence-Level IS**: Verify KL is averaged per sequence, not per token

## Expected Output

### Success Indicators

```
✓ Ray cluster started with 1 GPU
✓ SGLang engine initialized
✓ Rollout 0 completed
✓ train/ppo_kl: 0.0000 (step=0)  ← CRITICAL!
✓ train/pg_clipfrac: 0.0000 (step=0)  ← CRITICAL!
✓ Rollout 1 completed
✓ Training finished
```

### Failure Indicators

```
✗ train/ppo_kl: 0.0123 (step=0)  ← Should be exactly 0!
✗ CUDA Out of Memory  ← Reduce batch size
✗ AssertionError in model.py  ← Implementation bug
```

## WandB Integration (Optional)

For metric tracking:

```bash
# Set WandB key
export WANDB_API_KEY=your_key

# Add to ROLLOUT_ARGS in test_cispo.sh:
--wandb-project slime-cispo-test
--wandb-group cispo-validation
```

## Troubleshooting

### OOM (Out of Memory)

Reduce batch sizes in `test_cispo.sh`:

```bash
ROLLOUT_ARGS=(
   --rollout-batch-size 1  # Reduce from 2
   --global-batch-size 4   # Reduce from 8
)
```

### Ray Connection Issues

```bash
# Check Ray status
ray status

# Restart Ray
ray stop --force
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1
```

### Model Download Issues

If Qwen2.5-0.5B download fails, use a local model:

```bash
# Download first
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /path/to/model

# Update test_cispo.sh
CKPT_ARGS=(
   --hf-checkpoint /path/to/model
)
```

## Comparing with GSPO

To verify CISPO works differently from GSPO:

```bash
# Run GSPO test
bash tests/test_gspo.sh  # Note: requires 4 GPUs

# Run CISPO test
bash tests/test_cispo.sh  # Uses 1 GPU

# Compare WandB metrics:
# - CISPO should show different pg_clipfrac patterns
# - CISPO uses larger eps_clip_high (5.0 vs 0.00035)
# - Both should have ppo_kl=0 on first step
```

## Next Steps After Successful Test

1. **Capture Results**:
   ```bash
   # Save logs
   tail -100 ray_job_*.log > cispo_test_results.log

   # Screenshot WandB metrics if using
   ```

2. **Run Longer Test** (optional):
   ```bash
   # Edit test_cispo.sh:
   --num-rollout 10  # Increase from 2
   --rollout-batch-size 4  # Increase if memory allows
   ```

3. **Update PR**:
   - Add test results to PR description
   - Reference WandB run if available
   - Document any fixes needed

## Quick Reference

**Single Command Test:**
```bash
bash tests/test_cispo.sh 2>&1 | tee cispo_test.log
```

**Check Critical Metric:**
```bash
grep "train/ppo_kl" cispo_test.log | head -1
# Should show: train/ppo_kl: 0.0000
```

**Clean Everything:**
```bash
pkill -9 sglang; pkill -9 ray; pkill -9 python
ray stop --force
```
