# MEGATRON BACKEND

Primary training backend. Bridges Megatron-LM distributed training with slime RL loop.

## STRUCTURE

```
megatron_utils/
├── actor.py              # MegatronTrainRayActor — main training actor
├── model.py              # Model wrapping, TensorBackuper for weight snapshots
├── loss.py               # RL loss functions, advantage estimators
├── data.py               # DataIterator, dynamic batch sizing
├── megatron_to_hf/       # Weight conversion: Megatron → HuggingFace
│   ├── __init__.py       # Dispatcher + stateful parameter pairing
│   ├── llama.py, qwen*.py, ...  # Model-specific converters
│   └── processors/       # Padding removal, quantization
└── update_weight/        # Weight sync: Training → Inference
    ├── hf_weight_iterator_base.py   # Factory: Direct vs Bridge mode
    ├── update_weight_from_tensor.py # Gloo/NCCL/Ray transport
    └── common.py         # Async all-gather, CPU offload
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add RL algorithm | `loss.py` | Add estimator to dispatch |
| Add model converter | `megatron_to_hf/<model>.py` | Regex-based param mapping |
| Debug weight sync | `update_weight/` | Check Gloo vs NCCL mode |
| Actor/ref/old_actor swap | `model.py` → `TensorBackuper` | Named snapshots |

## CONVENTIONS

### Model Conversion Pattern
```python
# In megatron_to_hf/<model>.py
def _convert_<model>_to_hf(key: str, value: torch.Tensor) -> Generator:
    # Use regex to extract layer index from Megatron naming
    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", key)
    layer_idx = int(match.group(1))
    # Yield (hf_name, tensor) pairs
    yield f"model.layers.{layer_idx}.self_attn.q_proj.weight", value
```

### Weight Iterator Factory
- **Direct mode**: Raw manual mapping for simple models
- **Bridge mode**: Monkey-patches Megatron model via `AutoBridge`

### Hidden Parameter Pairing
`_cached_tensors` dict in `__init__.py` handles Q-LoRA/MLA where Megatron yields params separately but SGLang needs them combined.

## ANTI-PATTERNS

- **NEVER** skip `remove_padding` before HF conversion — vocab size mismatch
- **NEVER** return schedule plan in `forward_only` mode
- **NEVER** modify model weights without `TensorBackuper` snapshot

## GOTCHAS

- **Multi-Protocol Sync**: Colocate uses Gloo CPU gather → Ray IPC. Distributed uses NCCL broadcast from rank DP=TP=PP=0.
- **FP8 Quantization**: Requires DeepGemm for `quant_weight_ue8m0` kernels
- **Late Imports**: `hf_weight_iterator_base.py` uses late imports to avoid circular deps with `slime_plugins`
- **Virtual Pipeline**: Layer offset calculation depends on `megatron.core.transformer.transformer_layer.get_transformer_layer_offset`
