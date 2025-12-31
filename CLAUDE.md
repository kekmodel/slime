# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slime is an LLM post-training framework for reinforcement learning (RL) scaling. It connects Megatron-LM (training) with SGLang (inference) to enable high-performance RL training with flexible data generation workflows. The framework powers models like GLM-4.7, GLM-4.6, and GLM-4.5.

## Architecture

The system has three main components:
- **Training (Megatron/FSDP)**: Main training process using Megatron-LM or FSDP backend, reads data from Data Buffer, syncs parameters to rollout
- **Rollout (SGLang + router)**: Generates new data including rewards/verifier outputs, stores in Data Buffer
- **Data Buffer**: Bridge module managing prompt initialization, custom data, and rollout generation

Training loop (`train.py`): Rollout generation → Training step → Weight sync → Repeat

## Common Commands

### Installation
```bash
pip install -e .  # Development install
```

### Running Training
```bash
# Start Ray cluster first (multi-node)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   -- python3 train.py [args...]

# Example with script
bash scripts/run-glm4-9B.sh
```

### Code Quality
```bash
pre-commit install                                    # Setup hooks
pre-commit run --all-files --show-diff-on-failure   # Run all checks
```

### Running Tests
```bash
# Single test file (requires GPUs)
python tests/ci/gpu_lock_exec.py --count 8 -- python tests/test_quick_start_glm4_9B.py

# Run pytest directly
pytest tests/test_chunked_gae.py -v
```

### Weight Conversion
```bash
# HF to Megatron torch_dist
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} --hf-checkpoint /path/to/hf --save /path/to/torch_dist

# Megatron to HF
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/torch_dist/iter_xxx/ --output-dir /path/to/hf --origin-hf-dir /path/to/original/hf
```

## Key Arguments

Arguments are in three categories:
1. **Megatron args**: Standard Megatron params like `--tensor-model-parallel-size 2`
2. **SGLang args**: Prefixed with `--sglang-`, e.g., `--sglang-mem-fraction-static`
3. **slime args**: Defined in `slime/utils/arguments.py`

Critical training parameters:
- `--rollout-batch-size`: Number of prompts per rollout
- `--n-samples-per-prompt`: Responses generated per prompt
- `--global-batch-size`: Samples per optimizer step
- `--num-steps-per-rollout`: Training steps per rollout (default 1 for on-policy)
- `--colocate`: Share GPUs between training and inference
- `--train-backend`: `megatron` or `fsdp`

## Code Structure

```
slime/
├── backends/           # Training backends
│   ├── megatron_utils/ # Megatron integration, weight conversion
│   ├── fsdp_utils/     # FSDP backend
│   └── sglang_utils/   # SGLang engine management
├── ray/                # Ray actors and placement groups
├── rollout/            # Rollout generation, reward models, data sources
│   ├── rm_hub/         # Built-in reward models (deepscaler, math_utils, etc.)
│   └── filter_hub/     # Dynamic sampling filters
├── router/             # Request routing with middleware
└── utils/              # Arguments, logging, distributed utilities

slime_plugins/          # Model-specific plugins
├── mbridge/            # Megatron bridge implementations
├── models/             # Custom model implementations
└── rollout_buffer/     # External rollout buffer integration

scripts/
├── models/             # Model configuration shells (source these before training)
└── run-*.sh            # Example training scripts
```

## Extending slime

### Custom Rollout Function
```python
# Set via --rollout-function-path your.module:generate_rollout
async def generate_rollout(args, rollout_id, *, evaluation=False) -> list[list[Sample]]:
    # Must set: tokens, response_length, reward, truncated
    pass
```

### Custom Reward Model
```python
# Set via --custom-rm-path your.module:reward_func
async def reward_func(args, sample: Sample, **kwargs) -> float:
    pass
```

### Custom Generate Function (for multi-turn/agent scenarios)
```python
# Set via --custom-generate-function-path your.module:generate
async def generate(args, sample: Sample, sampling_params) -> Sample:
    # Handle loss_mask: 1 for model tokens, 0 for tool/env tokens
    pass
```

## Debugging

- `--debug-rollout-only`: Only initialize SGLang, skip Megatron
- `--debug-train-only`: Only initialize Megatron, skip SGLang
- `--save-debug-rollout-data path_{rollout_id}.pt`: Save rollout data
- `--load-debug-rollout-data path_{rollout_id}.pt`: Load rollout data for training debugging

## Supported Models

- **Qwen**: Qwen3 series (including MoE variants), Qwen2.5 series
- **DeepSeek**: V3, V3.1, R1
- **GLM**: GLM-4, GLM-4.5 (MoE)
- **Llama**: Llama 3
- **Other**: Moonlight, MIMO, Kimi-K2

Model configs are in `scripts/models/`. Source the appropriate script before training.
