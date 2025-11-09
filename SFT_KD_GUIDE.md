# SFT and Knowledge Distillation Guide for slime

This guide explains how to use slime for Supervised Fine-Tuning (SFT) and Knowledge Distillation (KD), leveraging Megatron's distributed training infrastructure without the RL rollout overhead.

## Table of Contents
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [Knowledge Distillation (KD)](#knowledge-distillation-kd)
- [Comparison Table](#comparison-table)
- [Troubleshooting](#troubleshooting)

---

## Supervised Fine-Tuning (SFT)

### Overview

slime supports pure supervised fine-tuning mode where:
- No SGLang inference engines are needed
- No reward models or RL algorithms
- Direct supervised learning with standard cross-entropy loss
- Full Megatron distributed training capabilities (TP, PP, CP, etc.)

### Key Configuration

```bash
SFT_ARGS=(
   # Core SFT settings
   --loss-type sft_loss                        # Use SFT loss instead of policy_loss
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --disable-compute-advantages-and-returns    # Disable RL advantage calculation
   --debug-train-only                          # Skip SGLang initialization

   # Data settings
   --prompt-data /path/to/training_data.jsonl
   --input-key messages                        # Key for input in your dataset
   --loss-mask-type qwen                       # or qwen3, distill_qwen

   # Training settings
   --rollout-batch-size 128
   --global-batch-size 128
   --num-epoch 3
   --rollout-shuffle

   # Loss calculation
   --calculate-per-token-loss                  # Optional: per-token loss normalization
)
```

### Complete Example Script

```bash
#!/bin/bash

# Cleanup
pkill -9 sglang; sleep 3
ray stop --force
pkill -9 ray; pkill -9 python; sleep 3

set -ex

# Load model configuration
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

# Checkpoint paths
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Base/
   --ref-load /root/Qwen3-4B-Base_torch_dist     # Used as initial weights if --load is not set
   --load /root/Qwen3-4B-Base_slime/             # Resume from here if exists
   --save /root/Qwen3-4B-Base_slime/
   --save-interval 1000
)

# SFT-specific arguments
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/openhermes2_5.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# Megatron parallelism settings
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# Optimizer
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]}
```

### Data Format

Your training data should be in JSONL format with messages in OpenAI chat format:

```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
  ]
}
```

For multi-turn conversations:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "Tell me about AI."},
    {"role": "assistant", "content": "AI is..."}
  ]
}
```

### Loss Mask Types

The `--loss-mask-type` determines which tokens to compute loss on:

- **`qwen`**: For Qwen/Qwen2 models with multi-turn conversations
- **`qwen3`**: For Qwen3 models (different tokenization)
- **`distill_qwen`**: For distillation data (single prompt-response pairs)

**How loss masks work:**
- `loss_mask[i] = 0`: No loss on this token (prompts, special tokens)
- `loss_mask[i] = 1`: Compute loss on this token (assistant responses)

---

## Knowledge Distillation (KD)

### Overview

slime supports Knowledge Distillation by loading two models simultaneously:
- **Teacher model** (larger, frozen): Loaded via `--ref-load`
- **Student model** (smaller, trained): Loaded via `--load`

The framework computes KL divergence between teacher and student predictions.

### Method 1: KL Divergence Loss (Recommended)

This method uses the built-in KL loss functionality originally designed for RL.

#### Configuration

```bash
CKPT_ARGS=(
   --hf-checkpoint /path/to/student_init       # Student model architecture
   --ref-load /path/to/teacher_model_torch_dist    # Teacher model (frozen)
   --load /path/to/student_checkpoint          # Student checkpoint (optional)
   --save /path/to/student_output/
   --save-interval 100
)

DISTILL_ARGS=(
   # Data loading
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /path/to/training_data.jsonl
   --input-key messages
   --loss-mask-type qwen3

   # Distillation loss settings
   --loss-type policy_loss              # Can also use custom_loss
   --use-kl-loss                        # Enable KL divergence loss
   --kl-loss-coef 0.5                   # Weight for KL loss (0.0-1.0)
   --kl-loss-type k3                    # KL computation method: k1, k2, k3, low_var_kl

   # Disable RL components
   --disable-compute-advantages-and-returns
   --debug-train-only

   # Training settings
   --rollout-batch-size 64
   --global-batch-size 64
   --num-epoch 1
)
```

#### KL Loss Types

- **`k1`**: Simple log ratio: `log(π_student) - log(π_teacher)`
- **`k2`**: Better approximation of KL divergence
- **`k3`**: Most stable, recommended for distillation
- **`low_var_kl`**: Low variance KL estimator

#### Loss Computation

The final loss is:
```
total_loss = NLL_loss + kl_loss_coef * KL(π_student || π_teacher)
```

Where:
- `NLL_loss`: Standard cross-entropy loss on ground-truth tokens
- `KL(...)`: KL divergence between student and teacher log probabilities
- `kl_loss_coef`: Balance between matching labels and matching teacher (0.0-1.0)

**Recommended values:**
- `kl_loss_coef = 0.5`: Equal weight to hard labels and teacher
- `kl_loss_coef = 1.0`: Pure distillation (ignore hard labels)
- `kl_loss_coef = 0.1-0.3`: Mostly supervised, slight teacher guidance

### Method 2: Custom Distillation Loss

For advanced distillation techniques (temperature scaling, feature matching, etc.).

#### Step 1: Create Custom Loss Function

Create `my_distillation_loss.py`:

```python
import torch
import torch.nn.functional as F
from argparse import Namespace
from typing import Callable
from slime.utils.types import RolloutBatch

def distillation_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Custom Knowledge Distillation with temperature scaling.

    Args:
        args: Must include custom parameters like distill_temperature, distill_alpha
        batch: Contains "ref_log_probs" from teacher model
        logits: Student model logits [1, T, vocab_size]
        sum_of_sample_mean: Reduction function for per-sample averaging

    Returns:
        (loss, metrics_dict)
    """
    # Hyperparameters (can be passed via --custom-config-path)
    temperature = getattr(args, 'distill_temperature', 2.0)
    alpha = getattr(args, 'distill_alpha', 0.5)  # Weight for hard labels

    # Get response tokens and masks
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]

    # Student probabilities (with temperature)
    student_log_probs = F.log_softmax(logits / temperature, dim=-1)

    # Teacher probabilities (from reference model)
    teacher_log_probs = batch["ref_log_probs"]
    teacher_log_probs = torch.cat(teacher_log_probs, dim=0)
    teacher_probs = torch.exp(teacher_log_probs / temperature)

    # KL Divergence Loss (soft labels)
    # KL(teacher || student) = sum(teacher * log(teacher / student))
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none',
        log_target=False
    )
    kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary

    # Apply loss mask and average
    masked_kl = []
    for kl, mask in zip(kl_div.split(response_lengths), loss_masks):
        masked_kl.append(kl * mask)
    kl_loss = sum_of_sample_mean(torch.cat(masked_kl)) * (temperature ** 2)

    # Hard label loss (standard cross-entropy)
    if alpha > 0:
        from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
        log_probs_dict = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=False,
        )
        hard_log_probs = torch.cat(log_probs_dict["log_probs"], dim=0)
        ce_loss = -sum_of_sample_mean(hard_log_probs)

        total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
    else:
        total_loss = kl_loss
        ce_loss = torch.tensor(0.0)

    return total_loss, {
        "loss": total_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "ce_loss": ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else torch.tensor(ce_loss),
    }
```

#### Step 2: Configuration YAML (Optional)

Create `distill_config.yaml`:

```yaml
distill_temperature: 2.0
distill_alpha: 0.5
```

#### Step 3: Run Distillation

```bash
DISTILL_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path my_distillation_loss.distillation_loss_function
   --custom-config-path distill_config.yaml

   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /path/to/data.jsonl
   --input-key messages

   --disable-compute-advantages-and-returns
   --debug-train-only
)
```

### Complete Distillation Example: Qwen-7B → Qwen-1.8B

```bash
#!/bin/bash

set -ex

# Use student model architecture
source scripts/models/qwen3-1.8B.sh

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-1.8B/             # Student architecture
   --ref-load /root/Qwen3-7B_torch_dist/         # Teacher model (larger)
   --save /root/Qwen3-1.8B_distilled/
   --save-interval 100
)

DISTILL_ARGS=(
   # Data
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/training_corpus.jsonl
   --input-key messages
   --loss-mask-type qwen3
   --rollout-shuffle

   # Distillation settings
   --loss-type policy_loss
   --use-kl-loss
   --kl-loss-coef 0.5                # Balance hard labels and teacher
   --kl-loss-type k3

   # Training
   --rollout-batch-size 64
   --global-batch-size 64
   --num-epoch 1

   --disable-compute-advantages-and-returns
   --debug-train-only
)

# Student model parallelism (1.8B model)
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6                         # Lower LR for distillation
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.05
   --weight-decay 0.01
)

# Start Ray and submit job
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${DISTILL_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]}
```

### Key Implementation Details

**How two models are managed:**

1. **Weight Management** (`actor.py:81-86`):
   ```python
   self.weights = {
       "actor": {},    # Student model (trained)
       "ref": {}       # Teacher model (frozen)
   }
   ```

2. **Model Switching** (`actor.py:191-208`):
   ```python
   def compute_log_prob(self, model_tag: str, ...):
       self.update_gpu_params_dict(self.weights[model_tag])  # Switch weights
       return forward_only(...)  # Compute log_probs
   ```

3. **Training Flow** (`actor.py:253-336`):
   ```python
   def train_actor(self, rollout_id, rollout_data):
       # Step 1: Get teacher log probs
       rollout_data.update(
           self.compute_log_prob("ref", data_iterator, ...)
       )

       # Step 2: Get student log probs (old policy)
       rollout_data.update(
           self.compute_log_prob("actor", data_iterator, ...)
       )

       # Step 3: Train student with KL loss
       train(self.model, self.optimizer, ...)
   ```

---

## Comparison Table

| Aspect | RL (PPO/GRPO) | SFT | Knowledge Distillation |
|--------|---------------|-----|------------------------|
| **Purpose** | Learn from rewards | Learn from labels | Learn from teacher model |
| **Models** | Actor + Ref (optional) | Single model | Student + Teacher |
| **Loss** | `policy_loss` | `sft_loss` | `policy_loss` + KL or custom |
| **--loss-type** | `policy_loss` | `sft_loss` | `policy_loss` or `custom_loss` |
| **--use-kl-loss** | Optional (false) | Not used | **Required** (true) |
| **--kl-loss-coef** | Small (0.01-0.1) | N/A | Medium-Large (0.3-1.0) |
| **--ref-load** | Optional (old policy) | Not used | **Required** (teacher) |
| **Rollout** | SGLang generation | Data loading | Data loading |
| **--debug-train-only** | false | **true** | **true** |
| **--disable-compute-advantages-and-returns** | false | **true** | **true** |
| **Reward Model** | Required | Not used | Not used |
| **SGLang Engines** | Required | **Not needed** | **Not needed** |

---

## Troubleshooting

### SFT Issues

**Problem: "No SGLang engines found"**
- **Solution**: Add `--debug-train-only` flag

**Problem: "Cannot find log_probs in batch"**
- **Solution**: Make sure `--loss-type sft_loss` is set

**Problem: "Loss is NaN"**
- **Cause**: Loss mask might be all zeros
- **Solution**:
  - Check your `--loss-mask-type` matches your model
  - Verify data format with `--input-key messages`
  - Look at the sft_rollout debug output for mask values

**Problem: "Out of memory during SFT"**
- **Solution**:
  - Use `--use-dynamic-batch-size`
  - Reduce `--max-tokens-per-gpu`
  - Increase `--tensor-model-parallel-size`

### Distillation Issues

**Problem: "ref model not found"**
- **Solution**: Make sure `--ref-load` points to valid torch_dist checkpoint

**Problem: "KL divergence is too large"**
- **Cause**: Teacher and student distributions are very different
- **Solution**:
  - Lower `--kl-loss-coef` (try 0.1-0.3 first)
  - Use `--kl-loss-type k3` for stability
  - Check if teacher model loaded correctly

**Problem: "Student model not learning"**
- **Cause**: KL loss dominating, student just copies teacher
- **Solution**:
  - Increase `--kl-loss-coef` slightly
  - Make sure hard labels are included (check loss_mask)
  - Try custom loss with temperature scaling

**Problem: "Different model sizes cause errors"**
- **Solution**:
  - Student and teacher must have same vocabulary size
  - TP/PP settings in `MODEL_ARGS` should match **student** model size
  - Teacher is loaded with same parallelism as student

### General Tips

1. **Checkpoint Conversion**:
   - Always convert HF checkpoints to torch_dist format first
   - Use `tools/convert_hf_to_torch_dist.py`

2. **First Step Verification**:
   - For SFT: Loss should be reasonable (-log(1/vocab_size) ≈ 8-10)
   - For KD: `log_probs` and `ref_log_probs` should be similar initially

3. **Memory Management**:
   - Teacher model uses same GPU memory as student during log prob computation
   - Use smaller batch sizes if loading two large models
   - Consider using `--use-dynamic-batch-size`

4. **Logging**:
   - Enable `--use-wandb` to track loss curves
   - Check `train/loss`, `train/kl_loss` metrics
   - For KD, monitor both CE loss and KL loss separately

5. **Learning Rate**:
   - SFT: 1e-5 to 5e-5 typical
   - KD: 5e-6 to 1e-5 (lower than SFT since starting from pre-trained)

---

## References

- **slime Documentation**: [https://github.com/THUDM/slime](https://github.com/THUDM/slime)
- **Model Configs**: `scripts/models/`
- **SFT Rollout**: `slime/rollout/sft_rollout.py`
- **Loss Functions**: `slime/backends/megatron_utils/loss.py`
- **Mask Generation**: `slime/utils/mask_utils.py`

---

**Last Updated**: 2025-01-08
