# CLAUDE.md

## Project Overview

**slime** is an LLM post-training framework for RL scaling that powers GLM-4.5, GLM-4.6, and other models. It connects Megatron (training) with SGLang (inference) to enable high-performance RL training with flexible data generation workflows.

### Core Architecture

The framework follows a three-module design:

1. **Training (Megatron)**: Main training process that reads from the Data Buffer and syncs parameters to rollout after training
2. **Rollout (SGLang + router)**: Generates new data (including rewards/verifier outputs) and stores in Data Buffer
3. **Data Buffer**: Bridge module managing prompt initialization, custom data, and rollout generation methods

Key directories:
- `slime/backends/`: Backend implementations for Megatron, SGLang, and FSDP
- `slime/rollout/`: Rollout logic and reward model implementations
- `slime/router/`: SGLang routing and middleware
- `slime/utils/`: Shared utilities (arguments, distributed utils, timers, etc.)
- `scripts/`: Training scripts and model configurations
- `tools/`: Conversion scripts between HF and Megatron formats
- `examples/`: Usage examples (slime_gym, tool_calling, multi_agent, eval, etc.)

## Development Commands

### Environment Setup
```bash
source .venv/bin/activate
uv pip install -e .

# If .venv doesn't exist
uv venv --python 3.12
```

### Required Environment Variables
- `PYTHONPATH=/path/to/Megatron-LM` - Required for Megatron training
- `OPENROUTER_API_KEY` - For examples/tool_calling benchmarks

### Code Quality
```bash
# Install pre-commit hooks
apt install pre-commit -y
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files --show-diff-on-failure --color=always

```

### Code Style
- **Line length**: 119 (black)
- **Imports**: isort (black-compatible profile)
- **Linting**: ruff (E, F, B, UP rules)
- **Type hints**: modern syntax (`list`, `dict`, `X | None`)

### Testing
```bash
# Run all tests
pytest

# Run specific test markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m system        # System tests

# Run tests in a specific directory
pytest tests/ci/

# Run with verbose output and timing
pytest --verbose --durations=0
```

### Model Weight Conversion

See `tools/convert_hf_to_torch_dist.py` (HF → Megatron) and `tools/convert_torch_dist_to_hf.py` (Megatron → HF). Requires sourcing model config first: `source scripts/models/<model>.sh`

### Running Training
```bash
# Single-node
bash scripts/run-glm4-9B.sh

# Multi-node: See scripts/ for Ray cluster setup examples
```

## Architecture Concepts

### Parameter Categories

slime accepts three types of arguments:

1. **Megatron arguments**: Read via `PYTHONPATH`. Examples: `--tensor-model-parallel-size`, `--num-layers`
2. **SGLang arguments**: Must be prefixed with `--sglang-`. Example: `--sglang-mem-fraction-static`
3. **slime-specific arguments**: See `slime/utils/arguments.py`

### Training Loop Constraints

The rollout-training loop must satisfy:
```
(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)
```

- `--rollout-batch-size`: Number of prompts per sampling round
- `--n-samples-per-prompt`: Responses generated per prompt
- `--global-batch-size`: Samples for one optimizer.step()
- `--num-steps-per-rollout`: Parameter updates per sampled data batch (default: 1 for on-policy)

### Parallelism (Megatron)

Supports TP, PP, CP, EP, ETP. **Gotcha**: Always enable `--sequence-parallel` when using `--tensor-model-parallel-size`.

### Colocated vs Disaggregated Mode

**Disaggregated (default)**: Training and inference use separate GPU pools
```bash
--actor-num-nodes 1 \
--actor-num-gpus-per-node 4 \
--rollout-num-gpus 4
```

**Colocated**: Training and inference share GPUs
```bash
--actor-num-nodes 1 \
--actor-num-gpus-per-node 8 \
--colocate \
--sglang-mem-fraction-static 0.8  # Reduce SGLang memory usage
```

### Custom Rollout and Reward Functions

For multi-turn interactions and agentic RL:

1. **Custom generation**: `--custom-generate-function-path module.path:function_name`
   - Signature: `async def generate(args, sample: Sample, sampling_params) -> Sample:`
   - Must handle `loss_mask`: Set to 1 for model-generated tokens, 0 for tool/environment outputs

2. **Custom reward**: `--custom-rm-path module.path:reward_function`
   - Signature: `async def reward_func(args, sample: Sample, **kwargs) -> float:`

3. **Metadata passing**: Use `--metadata-key` to load structured data into `Sample.metadata`

### Dynamic Batching

Enable for better GPU utilization:
```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu 4608
```

## Debugging

### Separate Training/Inference Debugging

- `--debug-rollout-only`: Only initialize SGLang (for debugging inference)
- `--debug-train-only`: Only initialize Megatron (for debugging training)
- `--save-debug-rollout-data /path/data_{rollout_id}.pt`: Save rollout outputs
- `--load-debug-rollout-data /path/data_{rollout_id}.pt`: Load fixed rollout data for training tuning

### First Training Step Checks

1. **Rollout coherence**: If garbled, check:
   - Parameters loaded correctly (check Megatron logs)
   - All parameters mapped correctly (especially for `pp_size > 1`)
   - Special buffers in SGLang released properly

2. **Log probabilities**: `log_probs` and `ref_log_probs` should be exactly equal (KL=0) on first step
   - If not equal: May be non-deterministic kernels in Transformer Engine
   - For CP mode: Use `--attention-backend flash` to enforce Flash Attention
   - If values are large (>1): Check training config or data/template mismatch

3. **KL divergence and grad_norm**: Should be 0 and small respectively when `num_steps_per_rollout == 1`
   - For MoE: Must enable `--moe-permute-fusion`

### Common Issues

- **OOM on second step (colocated mode)**: Reduce `--sglang-mem-fraction-static`
- **Embedding conversion issues**: Manually set `--vocab-size` during torch_dist→HF conversion (Megatron pads embeddings)
- **Precision issues with Transformer Engine**: Use `--attention-backend flash`

## Model Configurations

Model configs are in `scripts/models/`. When using a model:
1. Source the appropriate config: `source scripts/models/glm4-9B.sh`
2. Verify parameters match your model version (especially `--rotary-base`)
3. Override if needed: `MODEL_ARGS+=(--rotary-base 10000)`

Note: slime uses data packing (varlen/thd), so `--seq-length` and `--max-positional-embedding` don't limit context length.

## Advanced Features

### Dynamic Sampling (DAPO-style)
```bash
--over-sampling-batch-size 64 \  # > rollout-batch-size
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

### Partial Rollout
Enable `--partial-rollout` to cache and continue aborted samples from dynamic sampling. Customize extraction via `--buffer-filter-path`.

### bf16 Training + fp8 Inference
Download FP8 model variant (e.g., `Qwen/Qwen3-4B-FP8`) and set:
```bash
--hf-checkpoint /path/to/Qwen3-4B-FP8
--ref-load /path/to/bf16_torch_dist  # Still use bf16 for training
```

## Entry Points

- `train.py` - Synchronous training (default)
- `train_async.py` - Asynchronous training (experimental)
- `slime/utils/arguments.py` - All argument definitions (reference)

