# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-23
**Platform:** Python (Ray, PyTorch, Megatron-LM, SGLang)

## OVERVIEW
**slime** is a high-performance LLM post-training framework optimizing Reinforcement Learning (RL) scaling. It decouples **Training** (Megatron-LM/FSDP) from **Rollout** (SGLang) using **Ray** orchestration to maximize throughput and GPU utilization.

## STRUCTURE
```
.
├── slime/            # Core framework (Ray orchestration, RolloutManager)
├── slime_plugins/    # Extension points (mbridge, rollout buffers, custom models)
├── scripts/          # Operational entry points (training launchers, model configs)
├── examples/         # Usage patterns (PPO, GRPO, custom envs)
├── tests/            # Hybrid E2E (GPU-aware) and Unit test suite
├── tools/            # Utilities for checkpoint conversion (HF <-> Megatron)
└── docker/           # Build env with custom patches for Megatron/SGLang
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| **Launch Training** | `scripts/*.sh` | Use `ray job submit` via these scripts |
| **Add Model** | `slime_plugins/models/` | Register new model architectures here |
| **Modify Rollout** | `slime/ray/rollout.py` | Logic for data generation & SGLang integration |
| **Backend Logic** | `slime/backends/` | `TrainRayActor` impls for Megatron/FSDP |
| **CI/CD** | `.github/workflows/` | Auto-generated from `.j2` templates |

## ARCHITECTURE
- **Orchestration**: Ray Placement Groups manage separate actor sets for Training and Rollout.
- **Training**: Supports `Megatron-LM` (Tensor/Pipeline Parallel) and `FSDP2` (Fully Sharded).
- **Inference**: Uses `SGLang` for high-throughput generation during rollout.
- **Plugins**: `slime_plugins/mbridge` adapts HF models to Megatron-Core internals.

## CONVENTIONS
- **Ray-Centric**: Use `ray job submit` rather than direct `python` execution for distributed runs.
- **Patching**: Core dependencies (Megatron, SGLang) are heavily patched (`docker/patch/`).
- **Configs**: Arguments are grouped (CKPT, ROLLOUT, GRPO) in shell scripts for clarity.

## ANTI-PATTERNS (THIS PROJECT)
- **Direct Process Management**: DO NOT manually manage GPU processes; use Ray actors.
- **Blocking Calls**: Avoid blocking the main loop; use `train_async.py` patterns.
- **Hardcoded Paths**: Use `PYTHONPATH` and env vars (`SLIME_HOME`) for resource location.
- **Unpatched Deps**: DO NOT use vanilla `megatron-lm` without checking `docker/patch/`.

## COMMANDS
```bash
# Install
pip install -e .

# Run Tests (CI style)
python tests/ci/gpu_lock_exec.py --count 4 -- python tests/test_qwen_tiny.py

# Build Docker
docker build -t slimerl/slime:latest -f docker/Dockerfile .
```
