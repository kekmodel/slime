# ROLLOUT MODULE

Core data generation and reward computation. Orchestrates SGLang inference for RL training data.

## STRUCTURE

```
rollout/
├── sglang_rollout.py     # Default rollout loop: generate → reward → filter
├── sft_rollout.py        # SFT-style rollout (no RL)
├── data_source.py        # Dataset iteration, prompt loading
├── rm_hub/               # Built-in reward models
│   ├── math_utils.py     # Math reward (boxed answers)
│   ├── f1.py             # F1 score reward
│   ├── gpqa.py           # GPQA benchmark
│   ├── deepscaler.py     # DeepScaler reward
│   └── ifbench.py        # IFBench reward
└── filter_hub/           # Dynamic sampling filters
    └── dynamic_sampling_filters.py
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Modify generation loop | `sglang_rollout.py` → `generate_and_rm()` | Core async logic |
| Add reward model | `rm_hub/<name>.py` + register in `__init__.py` | Match builtin signature |
| Add filter | `filter_hub/` | Return bool or `DynamicFilterOutput` |
| Multi-turn tools | Override `--custom-generate-function-path` | See `examples/tool_calling/` |

## KEY FUNCTIONS

### `generate_rollout(args, rollout_id, data_buffer, evaluation=False)`
Main entry. Returns `RolloutFnTrainOutput` or `RolloutFnEvalOutput`.

### `generate_and_rm(args, samples, sampling_params, ...)`
Core async loop:
1. Call `async_generate()` → SGLang inference
2. Call `async_rm()` or `batched_async_rm()` → reward computation
3. Apply dynamic filter if configured

### Reward Model Dispatch
```python
# In rm_hub/__init__.py
async def async_rm(args, sample: Sample, **kwargs) -> float:
    if args.custom_rm_path:
        return await load_function(args.custom_rm_path)(args, sample, **kwargs)
    # else dispatch to builtin based on args.rm_type
```

## CONVENTIONS

- **Batch vs Single**: `async_rm` for single sample, `batched_async_rm` for lists. Custom RM should handle both if `group_rm` might be enabled.
- **loss_mask**: Set to `0` for tokens that shouldn't contribute to loss (tool outputs, env responses)

## ANTI-PATTERNS

- **NEVER** include environment/tool tokens in loss calculation
- **NEVER** assume single-sample RM signature if `group_rm` enabled

## GOTCHAS

- **Partial Rollouts**: `generate_and_rm` handles streaming/partial completions
- **Reward Hacking**: `RewardHackingMonitor` in utils detects length exploitation
- **Dynamic Batching**: `min_num_micro_batches` calculated from `max_tokens_per_gpu`
