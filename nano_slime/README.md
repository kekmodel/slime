# Nano SLIME

**Minimal GRPO Training Implementation with SGLang + FSDP/Megatron**

A learning-focused implementation of SLIME's core GRPO training loop.
Supports two backends:
- **FSDP**: HuggingFace + FSDP2 (simpler, recommended for learning)
- **Megatron**: Ray + Megatron-LM (production, supports TP/PP)

## Features

| Feature | File | Description |
|---------|------|-------------|
| GRPO | `slime/utils/ppo_utils.py` | Group Relative Policy Optimization |
| KL loss (k1/k2/k3) | `slime/utils/ppo_utils.py` | Multiple KL divergence formulations |
| unbiased_kl | `slime/utils/ppo_utils.py` | Off-policy importance sampling correction |
| adv wo std | `slime/rollout/reward.py` | GRPO without std normalization |
| use rollout logprobs | `slime/backends/training_utils/loss.py` | Off-policy training |
| filtering zero std | `slime/rollout/reward.py` | Filter uninformative groups |
| FSDP Actor | `slime/backends/fsdp/actor.py` | Distributed training with FSDP |
| Megatron Actor | `slime/backends/megatron/actor.py` | Ray + Megatron distributed training |
| SGLang Engine | `slime/backends/sglang/engine.py` | Fast inference with SGLang |
| Slime Router | `slime/router/router.py` | Load balancing for SGLang engines |
| Routing Replay | `slime/utils/routing_replay.py` | MoE expert routing consistency |
| TensorBoard | `slime/utils/tracking.py` | Training metrics logging |

## Quick Start

```bash
# Install dependencies
pip install torch transformers

# Mock mode (no GPU needed)
python train.py --mock --num-rollout 5

# ===== FSDP Backend (simpler, recommended for learning) =====
# Single GPU
python train.py --backend fsdp --hf-checkpoint meta-llama/Llama-3.2-1B

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 train.py --backend fsdp --hf-checkpoint meta-llama/Llama-3.2-1B

# ===== Megatron Backend (production, supports TP/PP) =====
# Requires: pip install ray megatron-core
python train.py --backend megatron --num-gpus 4 --hf-checkpoint meta-llama/Llama-3.2-1B

# With Tensor Parallelism
python train.py --backend megatron --num-gpus 8 --tp-size 2 --hf-checkpoint <model>
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│  1. Load prompts                                             │
│  2. SGLang.generate(prompts) → samples with log_probs       │
│  3. RewardModel(samples) → rewards                          │
│  4. post_process_rewards() → GRPO group normalization       │
│  5. FSDPActor.train(samples) → gradient update              │
│  6. FSDPActor.update_weights() → sync to SGLang             │
│  7. Repeat                                                   │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
nano_slime/
├── train.py                        # Main entry point
├── README.md
│
├── slime/
│   ├── utils/
│   │   ├── ppo_utils.py           # KL, GRPO, policy loss
│   │   ├── types.py               # Sample, RolloutBatch
│   │   ├── routing_replay.py      # MoE routing replay
│   │   └── tracking.py            # TensorBoard/WandB
│   │
│   ├── backends/
│   │   ├── fsdp/
│   │   │   ├── actor.py           # FSDP training actor
│   │   │   └── parallel.py        # Parallel state
│   │   ├── megatron/
│   │   │   └── actor.py           # Ray + Megatron training actor
│   │   ├── sglang/
│   │   │   └── engine.py          # SGLang inference wrapper
│   │   └── training_utils/
│   │       └── loss.py            # Advantage, policy loss
│   │
│   ├── rollout/
│   │   └── reward.py              # GRPO normalization
│   │
│   ├── ray/
│   │   ├── placement_group.py     # GPU placement
│   │   └── rollout.py             # RolloutManager
│   │
│   └── router/
│       └── router.py              # SlimeRouter
│
└── tests/
    ├── test_phase1_ppo_utils.py
    ├── test_phase2_loss.py
    └── test_phase3_reward.py
```

## Key Concepts

### GRPO (Group Relative Policy Optimization)

GRPO normalizes rewards **within groups** of samples from the same prompt:

```python
# Generate n samples per prompt
samples = engine.generate(prompts, n_samples=4)

# Group-wise normalization
rewards = rewards.reshape(-1, n_samples_per_prompt)  # [num_prompts, n_samples]
mean = rewards.mean(dim=-1, keepdim=True)           # Per-prompt baseline
rewards = rewards - mean                             # Relative quality
if grpo_std_normalization:
    std = rewards.std(dim=-1, keepdim=True)
    rewards = rewards / (std + 1e-6)
```

Why group-wise?
- Removes difficulty variation across prompts
- Learns relative quality within each prompt
- Return = Reward (simple, effective)

### KL Divergence Types

```python
# k1: Simple log ratio
kl = log_probs - ref_log_probs

# k2: Squared for numerical stability
kl = (log_probs - ref_log_probs) ** 2 / 2

# k3: Low variance, unbiased (recommended)
r = ref_log_probs - log_probs
kl = exp(-r) - 1 - r
```

### Off-policy Training with rollout_log_probs

```python
# Problem: log_probs at train time ≠ log_probs at rollout time
# Solution: Store rollout log_probs and use importance sampling

if use_rollout_logprobs:
    old_log_probs = rollout_log_probs  # From SGLang generation
else:
    old_log_probs = log_probs          # Re-computed at train time

# Importance ratio for unbiased gradients
ratio = exp(log_probs - old_log_probs)
```

## Architecture

### Backend: FSDP (Recommended for Learning)

```python
# HuggingFace model + FSDP for distributed training
model = AutoModelForCausalLM.from_pretrained(checkpoint)
model = FSDP(model, sharding_strategy=FULL_SHARD)

# Training step
logits = model(tokens)
loss = policy_loss_function(logits, advantages, old_log_probs)
loss.backward()
optimizer.step()
```

### Backend: Megatron (Production)

```python
# Ray actors for distributed training
import ray

# Create actors across GPUs
actors = [MegatronTrainRayActor.remote(world_size, rank) for rank in range(num_gpus)]
ray.get([a.init.remote(args, "actor", with_ref=True) for a in actors])

# Training step (parallel across all actors)
rollout_data_ref = ray.put(rollout_data)
futures = [a.train.remote(rollout_id, rollout_data_ref) for a in actors]
ray.get(futures)

# Megatron supports:
# - Tensor Parallelism (TP): Split model across GPUs
# - Pipeline Parallelism (PP): Split layers across stages
# - Data Parallelism (DP): Replicate for larger batches
```

### Inference (SGLang)

```python
# SGLang for fast sampling
engine = SGLangEngine(args)
samples = engine.generate(prompts, n_samples=4)

# Weight sync after training
engine.update_weights(actor.state_dict())
```

### Colocate Mode

When GPU memory is limited, share GPUs between training and inference:

```python
# Rollout phase: Training model on CPU
actor.sleep()  # model.cpu()
samples = engine.generate(prompts)

# Training phase: SGLang releases memory
engine.release_memory()
actor.wake_up()  # model.cuda()
actor.train(samples)
```

## Command Line Options

```bash
python train.py \
    --hf-checkpoint meta-llama/Llama-3.2-1B \
    --num-rollout 100 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 4 \
    --kl-coef 0.05 \
    --kl-loss-type k3 \
    --use-rollout-logprobs \
    --use-tensorboard \
    --lr 1e-6
```

## Testing

```bash
# Install pytest
pip install pytest

# Run tests
cd nano_slime
python -m pytest tests/ -v
```

## References

- Original SLIME: [THUDM/slime](https://github.com/THUDM/slime)
- GRPO Paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- PPO Paper: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- SGLang: [sgl-project/sglang](https://github.com/sgl-project/sglang)
