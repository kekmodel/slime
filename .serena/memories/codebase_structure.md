# Codebase Structure

## Top-Level Directory Layout

```
slime/
├── slime/              # Main package
├── slime_plugins/      # Plugin system for extensions
├── scripts/            # Training scripts and model configs
├── tools/              # Conversion and utility scripts
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Example implementations
├── docker/             # Docker configurations
├── train.py            # Main training entry point
├── train_async.py      # Async training entry point
└── setup.py            # Package installation
```

## Core Package Structure (`slime/`)

### `slime/backends/`
Backend implementations for different frameworks:
- **megatron_utils/**: Megatron-LM integration
  - `actor.py`: Actor model implementation
  - `model.py`, `model_provider.py`: Model definitions
  - `checkpoint.py`: Checkpoint loading/saving
  - `loss.py`: Loss computation
  - `data.py`: Data handling for training
  - `arguments.py`: Megatron argument parsing
  - `config_mapping/`: Model config mappings
  - `megatron_to_hf/`: Conversion logic

- **sglang_utils/**: SGLang inference integration
- **fsdp_utils/**: FSDP (Fully Sharded Data Parallel) support

### `slime/rollout/`
Data generation and rollout logic:
- `sglang_rollout.py`: Main rollout implementation using SGLang
- `sft_rollout.py`: Supervised fine-tuning rollout
- `base_types.py`: Base types and classes for rollout
- **rm_hub/**: Reward model implementations
- **filter_hub/**: Dynamic sampling filters

### `slime/router/`
SGLang routing and load balancing:
- `router.py`: Main router implementation
- **middleware_hub/**: Middleware components for routing

### `slime/utils/`
Shared utilities:
- `arguments.py`: Argument parsing for slime
- `distributed_utils.py`: Distributed training utilities
- `data.py`: Data handling utilities
- `ppo_utils.py`: PPO algorithm utilities
- `timer.py`: Timing utilities
- `async_utils.py`: Async operation helpers
- `ray_utils.py`: Ray integration utilities
- `wandb_utils.py`, `tensorboard_utils.py`: Logging
- `memory_utils.py`: Memory management
- `health_monitor.py`: System health monitoring
- `types.py`: Type definitions

### `slime/ray/`
Ray cluster integration

## Scripts Directory (`scripts/`)

### `scripts/models/`
Model configuration files for supported architectures:
- `glm4-9B.sh`
- `qwen3-4B.sh`
- Various other model configs

Contains `MODEL_ARGS` arrays with Megatron hyperparameters that must be manually specified (since Megatron cannot read configs from checkpoints).

### Training Scripts
Example: `run-glm4-9B.sh`, `run-qwen3-4B.sh`, etc.

Each script typically contains:
- `CKPT_ARGS`: Checkpoint paths
- `ROLLOUT_ARGS`: Rollout/inference parameters
- `EVAL_ARGS`: Evaluation configuration
- `PERF_ARGS`: Parallelism and performance settings
- `GRPO_ARGS`: Algorithm-specific parameters
- `OPTIMIZER_ARGS`: Optimizer configuration
- `SGLANG_ARGS`: SGLang service parameters
- `MISC_ARGS`: Miscellaneous settings

## Tools Directory (`tools/`)

Conversion scripts:
- `convert_hf_to_torch_dist.py`: HuggingFace → Megatron conversion
- `convert_torch_dist_to_hf.py`: Megatron → HuggingFace conversion

## Examples Directory (`examples/`)

Demonstrates various use cases:
- `multi_agent/`: Multi-agent RL examples
- `fully_async/`: Fully asynchronous training
- `search-r1/`: Search-based RL (multi-turn interaction)
- `retool/`: Tool-using agents
- `reproducibility/`: Reproducible training setups

## Tests Directory (`tests/`)

- `ci/`: Continuous integration tests
- `test-*.sh`: Shell-based integration tests
- `test_*.py`: Python unit/integration tests
- `command_utils.py`: Utilities for test commands

## Entry Points

### `train.py`
Main synchronous training entry point. Typical workflow:
1. Parse arguments (Megatron + SGLang + slime)
2. Initialize Ray cluster
3. Start training actor (Megatron)
4. Start rollout engines (SGLang)
5. Run rollout-training loop

### `train_async.py`
Asynchronous training mode with decoupled rollout and training.

## Plugin System (`slime_plugins/`)

Extensibility mechanism for custom:
- Reward models
- Rollout strategies
- Filters
- Middleware
