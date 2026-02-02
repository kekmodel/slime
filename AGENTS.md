# SLIME — PROJECT KNOWLEDGE BASE

**Generated:** 2025-02-02  
**Commit:** 81100143  
**Branch:** dev

## OVERVIEW

LLM post-training framework for RL scaling. Connects Megatron (training) ↔ SGLang (inference) via Ray orchestration. Three pillars: Training Actor, Rollout Manager, Data Buffer.

## STRUCTURE

```
slime/
├── train.py              # Main entry (sync training loop)
├── train_async.py        # Async entry (overlapped rollout/train)
├── slime/
│   ├── backends/         # Training engine abstractions
│   │   ├── megatron_utils/   # Primary: Megatron-LM integration
│   │   ├── fsdp_utils/       # Alternative: PyTorch FSDP2
│   │   └── sglang_utils/     # Inference engine wrapper
│   ├── ray/              # Ray actor lifecycle, placement groups
│   ├── rollout/          # Data generation, reward computation
│   │   ├── rm_hub/       # Built-in reward models
│   │   └── filter_hub/   # Dynamic sampling filters
│   ├── router/           # Load balancing across SGLang engines
│   └── utils/            # Shared utilities, arguments, types
├── slime_plugins/        # Contrib: model bridges, external integrations
├── scripts/              # Model configs, run scripts
├── tools/                # Weight conversion utilities
├── tests/                # GPU-locked test infrastructure
└── examples/             # Use-case implementations
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new model | `scripts/models/*.sh` + `slime/backends/megatron_utils/megatron_to_hf/` | Model args + converter |
| Custom rollout | `--rollout-function-path` | See `slime/rollout/sglang_rollout.py` |
| Custom reward | `--custom-rm-path` | Signature: `async def fn(args, sample: Sample) -> float` |
| Custom generate | `--custom-generate-function-path` | For multi-turn, tool calling |
| Weight conversion | `tools/convert_*.py` | HF↔Megatron torch_dist |
| Add backend | Inherit `TrainRayActor` | See `slime/ray/train_actor.py` |
| Debug training only | `--debug-train-only` | Skip SGLang init |
| Debug rollout only | `--debug-rollout-only` | Skip Megatron init |

## EXTENSION POINTS

Functions loaded via `slime.utils.misc.load_function(path)` — use dot notation: `module.submodule.function`

| Argument | Signature | Purpose |
|----------|-----------|---------|
| `--rollout-function-path` | `def fn(args, rollout_id, data_buffer, evaluation=False)` | Override entire rollout loop |
| `--custom-generate-function-path` | `async def fn(args, sample, sampling_params) -> Sample` | Custom generation (RAG, tools) |
| `--custom-rm-path` | `async def fn(args, sample: Sample \| list[Sample]) -> float \| list[float]` | Custom reward computation |
| `--dynamic-sampling-filter-path` | `def fn(args, samples: list[Sample]) -> bool` | Filter rollout samples |
| `--custom-convert-samples-to-train-data-path` | Custom | Sample → training data |
| `--slime-router-middleware-paths` | FastAPI middleware | Router extensions |

## CONVENTIONS

### Arguments
- **Megatron args**: Standard (e.g., `--tensor-model-parallel-size 2`)
- **SGLang args**: Prefix `--sglang-` (e.g., `--sglang-mem-fraction-static 0.7`)
- **slime args**: Defined in `slime/utils/arguments.py`

### Batch Size Relationship
```
(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)
```

### Code Style
- Black: line_length=119
- isort: black profile, `slime`/`slime_plugins` as first-party
- Ruff: E, F, B, UP rules (ignore E402, E501)
- pre-commit required: `pre-commit install`

## ANTI-PATTERNS

- **NEVER** pass iteration path to checkpoint loading — use base dir + `--ckpt-step`
- **NEVER** include tool/environment tokens in loss — set `loss_mask=0`
- **NEVER** skip weight sync before training — SGLang must have latest weights
- **NEVER** use `SLIME_BACKEND` env var — deprecated, use `--train-backend`
- **NEVER** disable Sequence Parallelism when using Tensor Parallelism
- **NEVER** train on HF checkpoints directly — convert to torch_dist first

## GOTCHAS

- **Weight sync order**: ALWAYS update inference weights first, then train
- **FSDP + tie_word_embedding**: Meta-tensor init hangs — known limitation
- **ROCm**: No apex gradient fusion — use `--no-gradient-accumulation-fusion`
- **First step check**: Verify `log_probs == ref_log_probs` (KL=0)
- **Data packing**: slime ALWAYS uses varlen/thd — `--seq-length` doesn't limit context

## COMMANDS

### Environment Setup (uv + venv)

```bash
# Create venv and install uv if not present
python -m venv .venv
source .venv/bin/activate

# Install uv (fast Python package installer)
pip install uv

# Install dependencies (auto-installs missing packages)
uv pip install -e . --no-deps
uv pip install -e ".[fsdp]" --no-deps  # FSDP extras

# Or install all requirements at once
uv pip install -r requirements.txt
```

### Code Style

```bash
pre-commit install
pre-commit run --all-files
```

### Test (GPU-locked)

```bash
python tests/ci/gpu_lock_exec.py --count <N> -- python tests/<test>.py
```

### Run Training

```bash
python train.py --train-backend megatron \
    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
    ...
```

### Weight Conversion

```bash
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} --hf-checkpoint /path/to/hf --save /path/to/torch_dist
```

## CI

PR labels trigger tests on self-hosted GPU runners:
- `run-ci-short` — 4 GPUs, quick tests
- `run-ci-megatron` — 8 GPUs, Megatron backend
- `run-ci-fsdp` — FSDP backend
- `run-ci-precision` — Numerical alignment

Test markers: `unit`, `integration`, `system`, `skipduringci`, `pleasefixme`

## MODELS

Supported via `scripts/models/*.sh`:
- GLM-4 series (9B, 32B, 4.5, 4.6, 4.7)
- Qwen series (2.5, 3, 3-MoE, 3-Next)
- DeepSeek V3 series (V3, V3.1, R1)
- Llama 3, Kimi-K2, Moonlight, MIMO
