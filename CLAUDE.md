# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slime is an LLM post-training framework for reinforcement learning (RL) scaling. It connects Megatron-LM (training) with SGLang (inference) via Ray to enable high-performance RL training. The framework powers models like GLM-4.7, GLM-4.6, and GLM-4.5.

## Architecture

Three main components orchestrated by Ray:
- **Training (Megatron/FSDP)**: Reads rollout data, computes log_probs/advantages, updates weights
- **Rollout (SGLang + router)**: Generates responses, computes rewards, returns Sample objects
- **Weight Sync**: Converts Megatron weights to HuggingFace format and sends to SGLang

Main loop in `train.py`:
```
for rollout_id in range(num_rollout):
    rollout_data = rollout_manager.generate()   # SGLang generates responses
    actor_model.async_train(rollout_data)       # Megatron trains on data
    actor_model.update_weights()                # Sync weights to SGLang
```

### Key Data Flow
```
DataSource → Prompts → SGLang (generate) → Sample objects → Reward Model
    → Training Data (tokens, log_probs, rewards, loss_mask) → Megatron (train)
```

## Ray Distributed Architecture

slime uses Ray as the distributed orchestration layer. Understanding Ray patterns is essential for working with the codebase.

### Core Ray Actors

```
┌─────────────────────────────────────────────────────────────────────┐
│                         train.py (Driver)                            │
│                                                                      │
│   ┌─────────────────────┐         ┌─────────────────────────────┐   │
│   │    RayTrainGroup    │         │      RolloutManager         │   │
│   │                     │         │      (@ray.remote)          │   │
│   │  ┌───────────────┐  │         │                             │   │
│   │  │TrainRayActor 0│──┼────────▶│  ┌─────────────────────┐   │   │
│   │  │  (GPU 0)      │  │         │  │   SGLangEngine 0    │   │   │
│   │  └───────────────┘  │         │  │   (GPU 4,5)         │   │   │
│   │  ┌───────────────┐  │         │  └─────────────────────┘   │   │
│   │  │TrainRayActor 1│  │         │  ┌─────────────────────┐   │   │
│   │  │  (GPU 1)      │  │         │  │   SGLangEngine 1    │   │   │
│   │  └───────────────┘  │         │  │   (GPU 6,7)         │   │   │
│   │       ...           │         │  └─────────────────────┘   │   │
│   └─────────────────────┘         └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Ray Components

| Component | File | Role |
|-----------|------|------|
| `RayTrainGroup` | `slime/ray/actor_group.py` | Manages multiple TrainRayActors as a group |
| `TrainRayActor` | `slime/ray/train_actor.py` | Base class for training actors (1 per GPU) |
| `MegatronTrainRayActor` | `slime/backends/megatron_utils/actor.py` | Megatron implementation of TrainRayActor |
| `RolloutManager` | `slime/ray/rollout.py` | Manages SGLang engines, routes requests |
| `SGLangEngine` | `slime/backends/sglang_utils/sglang_engine.py` | Wraps one SGLang server instance |

### Ray Patterns Used

**1. PlacementGroup for GPU Allocation** (`placement_group.py`):
```python
# Ensures actors are placed on specific GPUs
pg = placement_group([{"GPU": 1, "CPU": 1} for _ in range(num_gpus)])
actor = Actor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=gpu_index,
    )
).remote()
```

**2. Async Pattern for Parallel Execution**:
```python
# Launch all actors in parallel, wait for all to complete
handles = [actor.train.remote(data) for actor in actors]
results = ray.get(handles)
```

**3. Object Store for Data Transfer** (`ray.put`/`ray.get`):
```python
# Put large data in object store once, share reference
data_ref = ray.put(rollout_data)
# All actors receive same reference (zero-copy on same node)
[actor.train.remote(data_ref) for actor in actors]
```

### Communication Flow

```
1. Weight Update (Megatron → SGLang):
   TrainRayActor.update_weights()
   └─→ Gather weights from TP/PP ranks (NCCL)
   └─→ Convert to HuggingFace format
   └─→ Send to SGLangEngine via:
       - IPC (colocate, same machine)
       - NCCL broadcast (distributed, different machines)

2. Rollout Data (SGLang → Megatron):
   RolloutManager.generate()
   └─→ SGLangEngine generates responses (async HTTP to router)
   └─→ Compute rewards (batched async)
   └─→ Convert to training format
   └─→ Split by data_parallel_rank
   └─→ ray.put() each partition → return refs to driver
   └─→ Driver passes refs to TrainRayActors
```

### Actor Lifecycle

```python
# 1. Create placement groups (GPU allocation)
pgs = create_placement_groups(args)

# 2. Create RolloutManager (spawns SGLangEngines internally)
rollout_manager = RolloutManager.remote(args, pgs["rollout"])

# 3. Create training actors (via RayTrainGroup)
actor_model = RayTrainGroup(args, pg=pgs["actor"])
ray.get(actor_model.async_init(args, role="actor"))

# 4. Connect: actors get reference to rollout_manager
actor_model.set_rollout_manager(rollout_manager)

# 5. Initial weight sync
actor_model.update_weights()

# 6. Training loop
for rollout_id in range(num_rollout):
    data = ray.get(rollout_manager.generate.remote(rollout_id))
    ray.get(actor_model.async_train(rollout_id, data))
    actor_model.update_weights()
```

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
- `--n-samples-per-prompt`: Responses generated per prompt (for GRPO)
- `--global-batch-size`: Samples per optimizer step
- `--num-steps-per-rollout`: Training steps per rollout (default 1 for on-policy)
- `--colocate`: Share GPUs between training and inference (uses offload)
- `--train-backend`: `megatron` or `fsdp`

**Important constraint**: `rollout-batch-size × n-samples-per-prompt = global-batch-size × num-steps-per-rollout`

## Code Structure

### Key Entry Points (read in this order)
1. `train.py` - Main training loop, orchestrates everything
2. `slime/ray/rollout.py` - RolloutManager, manages SGLang engines
3. `slime/rollout/sglang_rollout.py` - Actual generation logic (generate_rollout)
4. `slime/backends/megatron_utils/actor.py` - MegatronTrainRayActor (training)
5. `slime/utils/ppo_utils.py` - PPO/GRPO algorithm implementation

### Directory Overview
```
slime/
├── backends/
│   ├── megatron_utils/
│   │   ├── actor.py           # MegatronTrainRayActor
│   │   ├── loss.py            # Log-prob and advantage computation
│   │   └── update_weight/     # Weight sync (Megatron→HF→SGLang)
│   ├── fsdp_utils/            # Alternative FSDP backend
│   └── sglang_utils/          # SGLang engine wrapper
├── ray/
│   ├── rollout.py             # RolloutManager (core!)
│   ├── placement_group.py     # GPU allocation
│   └── train_actor.py         # Base TrainRayActor class
├── rollout/
│   ├── sglang_rollout.py      # generate_rollout function
│   ├── rm_hub/                # Built-in reward models
│   └── filter_hub/            # Dynamic sampling filters (DAPO)
└── utils/
    ├── arguments.py           # All argument definitions
    ├── ppo_utils.py           # PPO/GRPO core algorithms
    └── types.py               # Sample class definition

scripts/models/                # Model configs (source before training)
```

### Core Data Structure: Sample
```python
Sample = {
    "prompt": str,              # Input prompt
    "response": str,            # Generated response
    "tokens": list[int],        # Token IDs (prompt + response)
    "response_length": int,     # Response token count
    "reward": float,            # Reward score
    "loss_mask": list[int],     # 1=train on this token, 0=skip
    "rollout_log_probs": list,  # Log probs from rollout (for importance sampling)
    "status": COMPLETED|TRUNCATED|ABORTED,
}
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

## GPU Allocation Modes

**Disaggregated (default)**: Separate GPUs for training and inference
```bash
--actor-num-gpus-per-node 4 --rollout-num-gpus 4  # 4 for train, 4 for inference
```

**Colocated**: Same GPUs, time-shared with offload
```bash
--colocate --actor-num-gpus-per-node 8  # All 8 GPUs shared
# Uses --sglang-mem-fraction-static 0.8 to leave room for Megatron init
```

## Debugging

- `--debug-rollout-only`: Only initialize SGLang, skip Megatron
- `--debug-train-only`: Only initialize Megatron, skip SGLang
- `--save-debug-rollout-data path_{rollout_id}.pt`: Save rollout data
- `--load-debug-rollout-data path_{rollout_id}.pt`: Load rollout data for training debugging

See [Debugging Guide](docs/en/developer_guide/debug.md) for detailed debugging tips.

## Synchronization Model

The training loop is **synchronous** - each phase completes before the next starts:

```
Rollout (generate) → Train (update) → Weight Sync → Next Rollout
       ↓                  ↓                ↓
   Wait for all      Wait for all     Wait for all
   responses         gradients        engines to load
```

This is required for on-policy RL correctness (rollout policy = training policy).

## Supported Models

- **Qwen**: Qwen3 series (including MoE variants), Qwen2.5 series
- **DeepSeek**: V3, V3.1, R1
- **GLM**: GLM-4, GLM-4.5 (MoE)
- **Llama**: Llama 3
- **Other**: Moonlight, MIMO, Kimi-K2

Model configs are in `scripts/models/`. Source the appropriate script before training.

## Documentation Links

- [Quick Start Guide](docs/en/get_started/quick_start.md)
- [Usage Documentation](docs/en/get_started/usage.md)
- [Debugging Guide](docs/en/developer_guide/debug.md)
- [Q&A](docs/en/get_started/qa.md)
