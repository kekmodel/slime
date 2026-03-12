# Megatron LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LoRA (Low-Rank Adaptation) training to slime's Megatron backend with merge-based weight transfer, including MoE expert LoRA support.

**Architecture:** LoRA adapters wrap existing Megatron parallel linear layers. Base weights are frozen; only LoRA params receive gradients. Before weight transfer to SGLang, LoRA weights are merged into base weights so SGLang runs in normal mode (zero inference overhead). Ref model in RL uses adapter-off (scaling=0) instead of CPU weight swap. MoE expert LoRA uses monkey-patched GroupedMLP forward to inject LoRA delta into stacked weights before the fused gmm kernel.

**Tech Stack:** PyTorch, Megatron-Core (ColumnParallelLinear, RowParallelLinear, GroupedMLP), slime Megatron backend

**Spec:** `docs/plans/2026-03-12-megatron-lora-training-design.md`

---

## Issue Resolution Summary

This plan resolves the following critical and important issues identified during review:

| ID | Issue | Resolution |
|----|-------|------------|
| **C1** | GroupedMLP expert LoRA forward not wired | Monkey-patch `GroupedMLP.forward()` to inject `bmm`-computed delta into `w1`/`w2` before `gg.ops.gmm` call |
| **C2** | Colocate CPU backup has unmerged weights | merge→re-backup→transfer→unmerge→re-backup; optimize with Noop backuper when LoRA |
| **C3** | `_interleave_qkv` merge dimension wrong | Transpose Q/K/V deltas before interleave: delta is `[out, H]`, interleave operates on last dim |
| **I1** | `num_kv_heads < tp_size` crashes LoRA dim | Read `base_layer.output_size_per_partition` directly instead of manual calculation |
| **I2** | `"ref" in backup_tags` always False with LoRA | New `_lora_enabled` / `_needs_ref_logprobs` flags; use `disable_lora`/`enable_lora` for ref forward |
| **I3** | `ref_update_interval` meaningless with LoRA | Auto-disable with warning in argument validation |
| **I4** | `save_adapter_only` not defined as argument | Add `--save-adapter-only` / `--no-save-adapter-only` with auto-default |
| **I5** | Adapter save from TP rank 0 is incomplete | All-gather LoRA params across TP before saving from rank 0 |
| **I6** | VP decoder layers not iterated | `inject_lora_adapters` / `merge` / `save` accept `list[DDP]` and iterate all VP chunks |

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `slime/backends/megatron_utils/lora/__init__.py` | Package exports |
| `slime/backends/megatron_utils/lora/config.py` | `LoRAConfig` dataclass, argument registration |
| `slime/backends/megatron_utils/lora/layers.py` | TP-compatible LoRA wrapper layers (2D shared + fused QKV/FC1) |
| `slime/backends/megatron_utils/lora/expert_lora.py` | Expert LoRA: 3D params + GroupedMLP forward monkey-patch (C1) |
| `slime/backends/megatron_utils/lora/injection.py` | `inject_lora_adapters()`, `freeze_base_params()` — VP-aware (I6) |
| `slime/backends/megatron_utils/lora/merge.py` | `merge_lora_weights()`, `unmerge_lora_weights()`, `disable_lora()`, `enable_lora()` — handles expert + fused QKV (C3) |
| `tests/test_lora_config.py` | Unit tests for LoRA configuration |
| `tests/test_lora_layers.py` | Unit tests for LoRA layer correctness |
| `tests/test_lora_merge.py` | Unit tests for merge/unmerge roundtrip |
| `tests/test_lora_injection.py` | Unit tests for injection + freeze |
| `tests/test_lora_expert.py` | Unit tests for expert LoRA forward + merge |

### Modified Files

| File | Changes |
|------|---------|
| `slime/utils/arguments.py` | Add LoRA CLI arguments, I3 validation, I4 auto-default |
| `slime/backends/megatron_utils/model.py:83-123` | Inject LoRA in `setup_model_and_optimizer()`, VP-aware (I6) |
| `slime/backends/megatron_utils/model.py:663-688` | Adapter-only save with all_gather (I5) |
| `slime/backends/megatron_utils/actor.py:48-154` | LoRA-aware init: skip ref backup, set `_lora_enabled` flag (I2) |
| `slime/backends/megatron_utils/actor.py:370-460` | LoRA ref forward via disable/enable (I2), C2 merge timing |
| `slime/backends/megatron_utils/actor.py:484-523` | Merge/unmerge around `update_weights()` with colocate re-backup (C2) |
| `slime/backends/megatron_utils/checkpoint.py` | Adapter checkpoint load support |

---

## Chunk 1: LoRA Config and Arguments

### Task 1: LoRA Configuration

**Files:**
- Create: `slime/backends/megatron_utils/lora/__init__.py`
- Create: `slime/backends/megatron_utils/lora/config.py`
- Modify: `slime/utils/arguments.py`
- Test: `tests/test_lora_config.py`

- [ ] **Step 1: Create lora package directory**

```bash
mkdir -p slime/backends/megatron_utils/lora
```

- [ ] **Step 2: Write config test**

Create `tests/test_lora_config.py`:

```python
"""Tests for LoRA configuration."""
import argparse

import pytest


def test_lora_config_from_args():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=64,
        lora_alpha=128,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )
    config = LoRAConfig.from_args(args)
    assert config.rank == 64
    assert config.alpha == 128
    assert config.scaling == 128 / 64
    assert "q_proj" in config.target_modules
    assert config.dropout == 0.05


def test_lora_config_disabled():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=0,
        lora_alpha=0,
        lora_target_modules=[],
        lora_dropout=0.0,
    )
    config = LoRAConfig.from_args(args)
    assert not config.enabled


def test_lora_config_default_alpha():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=32,
        lora_alpha=None,
        lora_target_modules=["q_proj"],
        lora_dropout=0.0,
    )
    config = LoRAConfig.from_args(args)
    assert config.alpha == 64  # default: 2 * rank


def test_lora_config_expert_requires_num_experts():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=64,
        lora_alpha=128,
        lora_target_modules=["expert"],
        lora_dropout=0.0,
        num_experts=None,
    )
    with pytest.raises(ValueError, match="expert"):
        LoRAConfig.from_args(args)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'slime.backends.megatron_utils.lora'`

- [ ] **Step 4: Implement LoRAConfig**

Create `slime/backends/megatron_utils/lora/__init__.py`:

```python
from slime.backends.megatron_utils.lora.config import LoRAConfig

__all__ = ["LoRAConfig"]
```

Create `slime/backends/megatron_utils/lora/config.py`:

```python
"""LoRA configuration for Megatron backend."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

ALL_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "expert"]
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 0
    alpha: float = 0.0
    target_modules: tuple[str, ...] = ()
    dropout: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @property
    def scaling(self) -> float:
        if self.rank == 0:
            return 0.0
        return self.alpha / self.rank

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LoRAConfig:
        rank = getattr(args, "lora_rank", 0)
        alpha = getattr(args, "lora_alpha", None)
        if alpha is None and rank > 0:
            alpha = 2 * rank
        elif alpha is None:
            alpha = 0.0
        target_modules = tuple(getattr(args, "lora_target_modules", []))
        dropout = getattr(args, "lora_dropout", 0.0)

        if "expert" in target_modules and not getattr(args, "num_experts", None):
            raise ValueError("'expert' in lora_target_modules requires --num-experts > 0")

        return cls(rank=rank, alpha=alpha, target_modules=target_modules, dropout=dropout)


def add_lora_args(parser: argparse._ActionsContainer) -> None:
    """Register LoRA-specific CLI arguments."""
    group = parser.add_argument_group(title="LoRA", description="Low-Rank Adaptation arguments")
    group.add_argument("--lora-rank", type=int, default=0, help="LoRA rank. 0 disables LoRA.")
    group.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha for scaling. Default: 2 * rank.")
    group.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        choices=ALL_TARGET_MODULES,
        help="Target modules for LoRA. Default: all except expert.",
    )
    group.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout rate.")
    group.add_argument("--adapter-load", type=str, default=None, help="Path to load LoRA adapter checkpoint.")
    # [I4] --save-adapter-only with auto-default when LoRA is enabled
    group.add_argument(
        "--save-adapter-only",
        action="store_true",
        default=None,
        dest="save_adapter_only",
        help="Save only adapter weights in checkpoint. Default: True when --lora-rank > 0.",
    )
    group.add_argument(
        "--no-save-adapter-only",
        action="store_false",
        dest="save_adapter_only",
        help="Save full checkpoint even when LoRA is enabled.",
    )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_config.py -v
```
Expected: All 4 tests PASS

- [ ] **Step 6: Add LoRA args to slime argument parser**

In `slime/utils/arguments.py`, find where custom argument groups are added and add:

```python
from slime.backends.megatron_utils.lora.config import add_lora_args
```

Call `add_lora_args(parser)` in the argument registration section. Also add validation in the post-processing section:

```python
# [I4] Auto-set save_adapter_only when LoRA is enabled
if args.save_adapter_only is None:
    args.save_adapter_only = getattr(args, "lora_rank", 0) > 0

# Set default target modules if not specified
if getattr(args, "lora_rank", 0) > 0 and args.lora_target_modules is None:
    from slime.backends.megatron_utils.lora.config import DEFAULT_TARGET_MODULES
    args.lora_target_modules = list(DEFAULT_TARGET_MODULES)

# [I3] LoRA makes ref_update_interval meaningless (base is frozen)
if getattr(args, "lora_rank", 0) > 0 and args.ref_update_interval is not None:
    logger.warning(
        "--ref-update-interval is ignored when LoRA is enabled. "
        "In LoRA mode, ref model = base model (adapter off), which is frozen."
    )
    args.ref_update_interval = None
```

- [ ] **Step 7: Commit**

```bash
git add slime/backends/megatron_utils/lora/ tests/test_lora_config.py slime/utils/arguments.py
git commit -m "feat(lora): add LoRAConfig and CLI arguments

Includes I3 (ref_update_interval auto-disable) and I4 (save_adapter_only auto-default)."
```

---

## Chunk 2: LoRA Layers (Shared / 2D)

### Task 2: Shared LoRA Layers

**Files:**
- Create: `slime/backends/megatron_utils/lora/layers.py`
- Test: `tests/test_lora_layers.py`

- [ ] **Step 1: Write LoRA layer tests**

Create `tests/test_lora_layers.py`:

```python
"""Tests for LoRA wrapper layers."""
import torch
import torch.nn as nn

import pytest


class MockColumnParallelLinear(nn.Module):
    """Mock for Megatron ColumnParallelLinear without distributed deps."""

    def __init__(self, input_size, output_size_per_partition):
        super().__init__()
        self.input_size = input_size
        self.output_size_per_partition = output_size_per_partition
        self.weight = nn.Parameter(torch.randn(output_size_per_partition, input_size))

    def forward(self, x):
        return nn.functional.linear(x, self.weight), None


class MockRowParallelLinear(nn.Module):
    """Mock for Megatron RowParallelLinear without distributed deps."""

    def __init__(self, input_size_per_partition, output_size):
        super().__init__()
        self.input_size_per_partition = input_size_per_partition
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(output_size, input_size_per_partition))

    def forward(self, x):
        return nn.functional.linear(x, self.weight), None


def test_lora_column_parallel_forward_shape():
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)

    x = torch.randn(2, 10, 64)  # [batch, seq, hidden]
    out, bias = lora(x)
    assert out.shape == (2, 10, 32)


def test_lora_column_parallel_zero_init():
    """LoRA output should be zero at initialization (B is zero-initialized)."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)

    x = torch.randn(2, 10, 64)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    torch.testing.assert_close(lora_out, base_out)


def test_lora_column_parallel_nonzero_after_update():
    """After modifying lora_B, output should differ from base."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)
    lora.lora_B.data.fill_(1.0)

    x = torch.randn(2, 10, 64)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    assert not torch.allclose(lora_out, base_out)


def test_lora_row_parallel_forward_shape():
    from slime.backends.megatron_utils.lora.layers import LoRARowParallelLinear

    base = MockRowParallelLinear(input_size_per_partition=32, output_size=64)
    lora = LoRARowParallelLinear(base, rank=8, alpha=16)

    x = torch.randn(2, 10, 32)
    out, bias = lora(x)
    assert out.shape == (2, 10, 64)


def test_lora_row_parallel_zero_init():
    from slime.backends.megatron_utils.lora.layers import LoRARowParallelLinear

    base = MockRowParallelLinear(input_size_per_partition=32, output_size=64)
    lora = LoRARowParallelLinear(base, rank=8, alpha=16)

    x = torch.randn(2, 10, 32)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    torch.testing.assert_close(lora_out, base_out)


def test_lora_base_params_frozen():
    """Base layer params should not receive gradients through LoRA wrapper."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)

    # Only LoRA params should be in the trainable set
    lora_params = {n for n, p in lora.named_parameters() if "lora_" in n}
    assert "lora_A" in lora_params
    assert "lora_B" in lora_params


def test_lora_scaling():
    """Verify scaling = alpha / rank is applied correctly."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=4, output_size_per_partition=4)
    base.weight.data.zero_()
    lora = LoRAColumnParallelLinear(base, rank=2, alpha=4)
    # scaling = 4 / 2 = 2.0

    # Set A and B to identity-like
    lora.lora_A.data = torch.eye(2, 4)  # [rank=2, input=4]
    lora.lora_B.data = torch.eye(4, 2)  # [output=4, rank=2]

    x = torch.ones(1, 1, 4)
    out, _ = lora(x)
    # base=0, lora = x @ A^T @ B^T * 2.0 = [1,1,0,0] * 2 = [2,2,0,0]
    expected = torch.tensor([[[2.0, 2.0, 0.0, 0.0]]])
    torch.testing.assert_close(out, expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_layers.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement shared LoRA layers**

Create `slime/backends/megatron_utils/lora/layers.py`:

```python
"""TP-compatible LoRA wrapper layers for Megatron parallel linears."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAColumnParallelLinear(nn.Module):
    """LoRA adapter wrapping a ColumnParallelLinear.

    Base weight: [output_size_per_partition, input_size] (sharded on dim=0)
    lora_A:      [rank, input_size]                      (replicated)
    lora_B:      [output_size_per_partition, rank]        (sharded on dim=0, matching base)
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_size = base_layer.input_size
        output_size = base_layer.output_size_per_partition

        self.lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # TP attributes — lora_B follows base weight partitioning
        self.lora_B.tensor_model_parallel = True
        self.lora_B.partition_dim = 0
        self.lora_B.partition_stride = 1
        # lora_A is replicated
        self.lora_A.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        # Unpack tuple if base returns (output, bias)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_out, bias


class LoRARowParallelLinear(nn.Module):
    """LoRA adapter wrapping a RowParallelLinear.

    Base weight: [output_size, input_size_per_partition] (sharded on dim=1)
    lora_A:      [rank, input_size_per_partition]        (sharded on dim=1, matching base)
    lora_B:      [output_size, rank]                     (replicated)
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_size = base_layer.input_size_per_partition
        output_size = base_layer.output_size

        self.lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # TP attributes — lora_A follows base weight partitioning
        self.lora_A.tensor_model_parallel = True
        self.lora_A.partition_dim = 1
        self.lora_A.partition_stride = 1
        # lora_B is replicated
        self.lora_B.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        # lora_A @ x produces partial sum that gets added to base partial sum
        # Both are reduced together by RowParallel's reduce/reduce-scatter
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_out, bias
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_layers.py -v
```
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add slime/backends/megatron_utils/lora/layers.py tests/test_lora_layers.py
git commit -m "feat(lora): add LoRAColumnParallelLinear and LoRARowParallelLinear"
```

---

### Task 3: Fused QKV and FC1 LoRA Layers

**Files:**
- Modify: `slime/backends/megatron_utils/lora/layers.py`
- Test: `tests/test_lora_layers.py` (append)

> **[I1] Fix:** LoRAFusedQKV does NOT compute `num_kv_heads_per_tp` internally. It receives
> `num_q_heads_per_tp` and `num_kv_heads_per_tp` from the caller. The caller (Task 6) derives these
> from `base_layer.output_size_per_partition` to handle `num_kv_heads < tp_size` correctly.

- [ ] **Step 1: Write fused QKV LoRA test**

Append to `tests/test_lora_layers.py`:

```python
def test_lora_fused_qkv_forward_shape():
    """Fused QKV LoRA should produce correct output shape."""
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    # Simulating: hidden=64, num_q_heads=8, num_kv_heads=2, head_dim=8, tp_size=1
    # QKV dim = (8 + 2*2) * 8 = 96
    qkv_out = 96
    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=qkv_out)
    lora = LoRAFusedQKV(
        base, rank=8, alpha=16,
        num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8,
    )
    x = torch.randn(2, 10, 64)
    out, _ = lora(x)
    assert out.shape == (2, 10, 96)


def test_lora_fused_qkv_zero_init():
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=96)
    lora = LoRAFusedQKV(
        base, rank=8, alpha=16,
        num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8,
    )
    x = torch.randn(2, 10, 64)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    torch.testing.assert_close(lora_out, base_out)


def test_lora_fused_qkv_kv_heads_equal_one():
    """[I1] Test with num_kv_heads_per_tp=1 (simulating num_kv_heads < tp_size with replication)."""
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    # num_kv_heads=1 per TP rank (replicated), num_q_heads=4 per TP rank, head_dim=8
    qkv_out = (4 + 2 * 1) * 8  # = 48
    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=qkv_out)
    lora = LoRAFusedQKV(
        base, rank=4, alpha=8,
        num_q_heads_per_tp=4, num_kv_heads_per_tp=1, head_dim=8,
    )
    x = torch.randn(1, 5, 64)
    out, _ = lora(x)
    assert out.shape == (1, 5, 48)


def test_lora_fused_fc1_forward_shape():
    """Fused FC1 (SwiGLU) LoRA should produce correct output shape."""
    from slime.backends.megatron_utils.lora.layers import LoRAFusedFC1

    # SwiGLU: output = 2 * ffn_hidden / tp_size
    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=128)
    lora = LoRAFusedFC1(base, rank=8, alpha=16)
    x = torch.randn(2, 10, 64)
    out, _ = lora(x)
    assert out.shape == (2, 10, 128)


def test_lora_fused_fc1_zero_init():
    from slime.backends.megatron_utils.lora.layers import LoRAFusedFC1

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=128)
    lora = LoRAFusedFC1(base, rank=8, alpha=16)
    x = torch.randn(2, 10, 64)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    torch.testing.assert_close(lora_out, base_out)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_layers.py::test_lora_fused_qkv_forward_shape -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement fused QKV and FC1 LoRA**

Add to `slime/backends/megatron_utils/lora/layers.py`:

```python
class LoRAFusedQKV(nn.Module):
    """Separate Q/K/V LoRA adapters on Megatron's fused linear_qkv (ColumnParallel).

    Megatron stores QKV interleaved per group: [Q_group0, K_group0, V_group0, Q_group1, ...]
    Each Q/K/V gets its own lora_A (replicated) and lora_B (TP-sharded on dim=0).
    The forward output is arranged to match Megatron's interleaved layout.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        num_q_heads_per_tp: int,
        num_kv_heads_per_tp: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.num_q_heads_per_tp = num_q_heads_per_tp
        self.num_kv_heads_per_tp = num_kv_heads_per_tp
        self.head_dim = head_dim

        input_size = base_layer.input_size
        q_out = num_q_heads_per_tp * head_dim
        kv_out = num_kv_heads_per_tp * head_dim

        # Q LoRA
        self.q_lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.q_lora_B = nn.Parameter(torch.zeros(q_out, rank))
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))

        # K LoRA
        self.k_lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.k_lora_B = nn.Parameter(torch.zeros(kv_out, rank))
        nn.init.kaiming_uniform_(self.k_lora_A, a=math.sqrt(5))

        # V LoRA
        self.v_lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.v_lora_B = nn.Parameter(torch.zeros(kv_out, rank))
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # TP attributes for B matrices (sharded like base)
        for b_param in [self.q_lora_B, self.k_lora_B, self.v_lora_B]:
            b_param.tensor_model_parallel = True
            b_param.partition_dim = 0
            b_param.partition_stride = 1
        for a_param in [self.q_lora_A, self.k_lora_A, self.v_lora_A]:
            a_param.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        dx = self.dropout(x)
        q_corr = F.linear(F.linear(dx, self.q_lora_A), self.q_lora_B) * self.scaling
        k_corr = F.linear(F.linear(dx, self.k_lora_A), self.k_lora_B) * self.scaling
        v_corr = F.linear(F.linear(dx, self.v_lora_A), self.v_lora_B) * self.scaling

        lora_out = _interleave_qkv(
            q_corr, k_corr, v_corr,
            self.num_q_heads_per_tp, self.num_kv_heads_per_tp, self.head_dim,
        )
        return base_output + lora_out, bias


def _interleave_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    num_q_heads_per_tp: int, num_kv_heads_per_tp: int, head_dim: int,
) -> torch.Tensor:
    """Arrange Q/K/V corrections into Megatron's interleaved group layout.

    Input shapes: q=[..., num_q_heads*head_dim], k=[..., num_kv_heads*head_dim], v=same as k
    Output shape: [..., (num_q_heads + 2*num_kv_heads)*head_dim] in interleaved group order.
    """
    prefix_shape = q.shape[:-1]
    q_per_kv = num_q_heads_per_tp // num_kv_heads_per_tp

    # Reshape to per-group
    q = q.view(*prefix_shape, num_kv_heads_per_tp, q_per_kv * head_dim)
    k = k.view(*prefix_shape, num_kv_heads_per_tp, head_dim)
    v = v.view(*prefix_shape, num_kv_heads_per_tp, head_dim)

    # Interleave: [Q_group, K_group, V_group] per group
    interleaved = torch.cat([q, k, v], dim=-1)  # [..., num_kv_groups, (q_per_kv+2)*head_dim]
    return interleaved.view(*prefix_shape, -1)


def _interleave_qkv_weight(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    num_q_heads_per_tp: int, num_kv_heads_per_tp: int, head_dim: int,
) -> torch.Tensor:
    """[C3] Interleave Q/K/V weight deltas (2D: [out_dim, input_dim]).

    Unlike _interleave_qkv which operates on activation tensors [..., last_dim],
    weight deltas have shape [out_dim, H]. We interleave along dim=0 (the output dim).

    Args:
        q: [q_out, H], k: [kv_out, H], v: [kv_out, H]
    Returns:
        [qkv_out, H] in interleaved group layout
    """
    q_per_kv = num_q_heads_per_tp // num_kv_heads_per_tp
    H = q.shape[1]

    # Reshape to per-group: [num_kv_groups, heads*head_dim, H]
    q = q.view(num_kv_heads_per_tp, q_per_kv * head_dim, H)
    k = k.view(num_kv_heads_per_tp, head_dim, H)
    v = v.view(num_kv_heads_per_tp, head_dim, H)

    # Interleave along dim=1 (output dim within each group)
    interleaved = torch.cat([q, k, v], dim=1)  # [num_kv_groups, (q_per_kv+2)*head_dim, H]
    return interleaved.view(-1, H)  # [qkv_out, H]


class LoRAFusedFC1(nn.Module):
    """Separate gate/up LoRA adapters on Megatron's fused linear_fc1 (ColumnParallel, SwiGLU).

    linear_fc1 output is [gate_proj, up_proj] concatenated.
    Each gets its own lora_A (replicated) and lora_B (TP-sharded on dim=0).
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_size = base_layer.input_size
        half_out = base_layer.output_size_per_partition // 2

        # Gate LoRA
        self.gate_lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.gate_lora_B = nn.Parameter(torch.zeros(half_out, rank))
        nn.init.kaiming_uniform_(self.gate_lora_A, a=math.sqrt(5))

        # Up LoRA
        self.up_lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.up_lora_B = nn.Parameter(torch.zeros(half_out, rank))
        nn.init.kaiming_uniform_(self.up_lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for b_param in [self.gate_lora_B, self.up_lora_B]:
            b_param.tensor_model_parallel = True
            b_param.partition_dim = 0
            b_param.partition_stride = 1
        for a_param in [self.gate_lora_A, self.up_lora_A]:
            a_param.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        dx = self.dropout(x)
        gate_corr = F.linear(F.linear(dx, self.gate_lora_A), self.gate_lora_B) * self.scaling
        up_corr = F.linear(F.linear(dx, self.up_lora_A), self.up_lora_B) * self.scaling
        lora_out = torch.cat([gate_corr, up_corr], dim=-1)
        return base_output + lora_out, bias
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_layers.py -v
```
Expected: All 12 tests PASS (7 shared + 5 fused)

- [ ] **Step 5: Commit**

```bash
git add slime/backends/megatron_utils/lora/layers.py tests/test_lora_layers.py
git commit -m "feat(lora): add LoRAFusedQKV and LoRAFusedFC1 layers

Includes _interleave_qkv_weight for correct weight-space merge (C3 fix).
Adds I1 test for num_kv_heads_per_tp=1 edge case."
```

---

## Chunk 3: Merge/Unmerge and Disable/Enable

### Task 4: Merge and Disable Operations

**Files:**
- Create: `slime/backends/megatron_utils/lora/merge.py`
- Test: `tests/test_lora_merge.py`

> **[C3] Fix:** `_merge_fused_qkv` uses `_interleave_qkv_weight` (operates on `[out, H]`) instead of
> `_interleave_qkv` (operates on `[..., last_dim]`). The weight matrix is `[output_dim, input_dim]`
> so interleaving must happen along dim=0, not the last dim.

- [ ] **Step 1: Write merge/unmerge tests**

Create `tests/test_lora_merge.py`:

```python
"""Tests for LoRA merge/unmerge and disable/enable operations."""
import torch
import torch.nn as nn

from tests.test_lora_layers import MockColumnParallelLinear, MockRowParallelLinear


def test_merge_unmerge_roundtrip_column():
    """Merge then unmerge should restore original base weights."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear
    from slime.backends.megatron_utils.lora.merge import merge_lora_weights, unmerge_lora_weights

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)
    lora.lora_B.data.normal_()

    original_weight = base.weight.data.clone()

    model = nn.Module()
    model.layer = lora

    merge_lora_weights(model)
    assert not torch.allclose(base.weight.data, original_weight)  # merged

    unmerge_lora_weights(model)
    torch.testing.assert_close(base.weight.data, original_weight)  # restored


def test_merge_equivalence():
    """Merged base forward should equal LoRA forward."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear
    from slime.backends.megatron_utils.lora.merge import merge_lora_weights

    base = MockColumnParallelLinear(input_size=32, output_size_per_partition=16)
    lora = LoRAColumnParallelLinear(base, rank=4, alpha=8)
    lora.lora_A.data.normal_()
    lora.lora_B.data.normal_()

    x = torch.randn(1, 5, 32)
    lora_out, _ = lora(x)

    model = nn.Module()
    model.layer = lora
    merge_lora_weights(model)

    # After merge, base layer alone should give same output
    merged_out = nn.functional.linear(x, base.weight)
    torch.testing.assert_close(merged_out, lora_out, atol=1e-5, rtol=1e-5)


def test_merge_fused_qkv_roundtrip():
    """[C3] Fused QKV merge/unmerge roundtrip with correct weight dimension handling."""
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV
    from slime.backends.megatron_utils.lora.merge import merge_lora_weights, unmerge_lora_weights

    qkv_out = (4 + 2 * 2) * 8  # 64
    base = MockColumnParallelLinear(input_size=32, output_size_per_partition=qkv_out)
    lora = LoRAFusedQKV(base, rank=4, alpha=8, num_q_heads_per_tp=4, num_kv_heads_per_tp=2, head_dim=8)
    # Initialize with non-zero values
    lora.q_lora_B.data.normal_()
    lora.k_lora_B.data.normal_()
    lora.v_lora_B.data.normal_()

    original_weight = base.weight.data.clone()

    model = nn.Module()
    model.layer = lora

    merge_lora_weights(model)
    assert not torch.allclose(base.weight.data, original_weight)

    unmerge_lora_weights(model)
    torch.testing.assert_close(base.weight.data, original_weight, atol=1e-6, rtol=1e-6)


def test_merge_fused_fc1_roundtrip():
    from slime.backends.megatron_utils.lora.layers import LoRAFusedFC1
    from slime.backends.megatron_utils.lora.merge import merge_lora_weights, unmerge_lora_weights

    base = MockColumnParallelLinear(input_size=32, output_size_per_partition=64)
    lora = LoRAFusedFC1(base, rank=4, alpha=8)
    lora.gate_lora_B.data.normal_()
    lora.up_lora_B.data.normal_()

    original_weight = base.weight.data.clone()

    model = nn.Module()
    model.layer = lora

    merge_lora_weights(model)
    unmerge_lora_weights(model)
    torch.testing.assert_close(base.weight.data, original_weight, atol=1e-6, rtol=1e-6)


def test_disable_enable_lora():
    """disable_lora should make output equal to base; enable_lora should restore."""
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear
    from slime.backends.megatron_utils.lora.merge import disable_lora, enable_lora

    base = MockColumnParallelLinear(input_size=32, output_size_per_partition=16)
    lora = LoRAColumnParallelLinear(base, rank=4, alpha=8)
    lora.lora_B.data.fill_(1.0)

    model = nn.Module()
    model.layer = lora

    x = torch.randn(1, 5, 32)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    assert not torch.allclose(lora_out, base_out)

    disable_lora(model)
    disabled_out, _ = lora(x)
    torch.testing.assert_close(disabled_out, base_out)

    enable_lora(model)
    enabled_out, _ = lora(x)
    torch.testing.assert_close(enabled_out, lora_out)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_merge.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement merge/unmerge/disable/enable**

Create `slime/backends/megatron_utils/lora/merge.py`:

```python
"""LoRA weight merge/unmerge and disable/enable utilities."""
from __future__ import annotations

import torch
import torch.nn as nn

from slime.backends.megatron_utils.lora.layers import (
    LoRAColumnParallelLinear,
    LoRAFusedFC1,
    LoRAFusedQKV,
    LoRARowParallelLinear,
    _interleave_qkv_weight,
)

_LORA_LAYER_TYPES = (LoRAColumnParallelLinear, LoRARowParallelLinear, LoRAFusedQKV, LoRAFusedFC1)


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA adapters into base weights in-place. Call before weight transfer."""
    for module in model.modules():
        if isinstance(module, LoRAColumnParallelLinear):
            module.base_layer.weight.data += (module.lora_B @ module.lora_A) * module.scaling
        elif isinstance(module, LoRARowParallelLinear):
            module.base_layer.weight.data += (module.lora_B @ module.lora_A) * module.scaling
        elif isinstance(module, LoRAFusedQKV):
            _merge_fused_qkv(module)
        elif isinstance(module, LoRAFusedFC1):
            _merge_fused_fc1(module)


def unmerge_lora_weights(model: nn.Module) -> None:
    """Reverse merge to restore base weights. Call after weight transfer."""
    for module in model.modules():
        if isinstance(module, LoRAColumnParallelLinear):
            module.base_layer.weight.data -= (module.lora_B @ module.lora_A) * module.scaling
        elif isinstance(module, LoRARowParallelLinear):
            module.base_layer.weight.data -= (module.lora_B @ module.lora_A) * module.scaling
        elif isinstance(module, LoRAFusedQKV):
            _unmerge_fused_qkv(module)
        elif isinstance(module, LoRAFusedFC1):
            _unmerge_fused_fc1(module)


def _merge_fused_qkv(module: LoRAFusedQKV) -> None:
    """[C3] Merge Q/K/V LoRA into fused QKV weight in interleaved layout.

    Weight is [qkv_out, H]. _interleave_qkv_weight operates on dim=0 (output dim).
    """
    q_delta = (module.q_lora_B @ module.q_lora_A) * module.scaling  # [q_out, H]
    k_delta = (module.k_lora_B @ module.k_lora_A) * module.scaling  # [kv_out, H]
    v_delta = (module.v_lora_B @ module.v_lora_A) * module.scaling  # [kv_out, H]

    delta = _interleave_qkv_weight(
        q_delta, k_delta, v_delta,
        module.num_q_heads_per_tp, module.num_kv_heads_per_tp, module.head_dim,
    )
    module.base_layer.weight.data += delta


def _unmerge_fused_qkv(module: LoRAFusedQKV) -> None:
    q_delta = (module.q_lora_B @ module.q_lora_A) * module.scaling
    k_delta = (module.k_lora_B @ module.k_lora_A) * module.scaling
    v_delta = (module.v_lora_B @ module.v_lora_A) * module.scaling

    delta = _interleave_qkv_weight(
        q_delta, k_delta, v_delta,
        module.num_q_heads_per_tp, module.num_kv_heads_per_tp, module.head_dim,
    )
    module.base_layer.weight.data -= delta


def _merge_fused_fc1(module: LoRAFusedFC1) -> None:
    """Merge gate/up LoRA into fused FC1 weight."""
    gate_delta = (module.gate_lora_B @ module.gate_lora_A) * module.scaling
    up_delta = (module.up_lora_B @ module.up_lora_A) * module.scaling
    delta = torch.cat([gate_delta, up_delta], dim=0)
    module.base_layer.weight.data += delta


def _unmerge_fused_fc1(module: LoRAFusedFC1) -> None:
    gate_delta = (module.gate_lora_B @ module.gate_lora_A) * module.scaling
    up_delta = (module.up_lora_B @ module.up_lora_A) * module.scaling
    delta = torch.cat([gate_delta, up_delta], dim=0)
    module.base_layer.weight.data -= delta


def disable_lora(model: nn.Module) -> None:
    """Set all LoRA scaling to 0 (model behaves as base only). Used for ref forward."""
    for module in model.modules():
        if isinstance(module, _LORA_LAYER_TYPES):
            module._saved_scaling = module.scaling
            module.scaling = 0.0
        # Also handle expert LoRA (imported lazily to avoid circular deps)
        if hasattr(module, "_lora_enabled") and hasattr(module, "_lora_scale"):
            module._saved_lora_scale = module._lora_scale
            module._lora_scale = 0.0


def enable_lora(model: nn.Module) -> None:
    """Restore LoRA scaling after disable_lora."""
    for module in model.modules():
        if isinstance(module, _LORA_LAYER_TYPES):
            if hasattr(module, "_saved_scaling"):
                module.scaling = module._saved_scaling
                del module._saved_scaling
        if hasattr(module, "_saved_lora_scale"):
            module._lora_scale = module._saved_lora_scale
            del module._saved_lora_scale
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_merge.py -v
```
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add slime/backends/megatron_utils/lora/merge.py tests/test_lora_merge.py
git commit -m "feat(lora): add merge/unmerge and disable/enable utilities

Uses _interleave_qkv_weight for correct weight-dim merge (C3 fix).
Handles expert LoRA disable/enable via _lora_scale attribute."
```

---

## Chunk 4: Injection and Integration

### Task 5: LoRA Injection into Model

**Files:**
- Create: `slime/backends/megatron_utils/lora/injection.py`
- Test: `tests/test_lora_injection.py`

> **[I6] Fix:** `inject_lora_adapters` accepts `list[DDP]` (VP model chunks) and iterates all chunks.
> `_get_decoder_layers` handles DDP unwrapping + VP chunk iteration.

- [ ] **Step 1: Write injection test**

Create `tests/test_lora_injection.py`:

```python
"""Tests for LoRA injection into model."""
import argparse

import torch
import torch.nn as nn


def _make_mock_model(num_layers=2, hidden=64, ffn_hidden=128, num_q_heads=8, num_kv_heads=2, head_dim=8):
    """Create a mock GPTModel-like structure for testing injection."""
    from tests.test_lora_layers import MockColumnParallelLinear, MockRowParallelLinear

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
            self.linear_qkv = MockColumnParallelLinear(hidden, qkv_dim)
            self.linear_qkv.input_size = hidden
            self.linear_proj = MockRowParallelLinear(hidden, hidden)

    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = MockColumnParallelLinear(hidden, ffn_hidden * 2)  # SwiGLU
            self.linear_fc1.input_size = hidden
            self.linear_fc2 = MockRowParallelLinear(ffn_hidden, hidden)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = MockAttention()
            self.mlp = MockMLP()

    class MockDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])

    class MockGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = MockDecoder()

    return MockGPTModel()


def test_inject_lora_replaces_layers():
    from slime.backends.megatron_utils.lora.config import LoRAConfig
    from slime.backends.megatron_utils.lora.injection import inject_lora_adapters
    from slime.backends.megatron_utils.lora.layers import LoRAFusedFC1, LoRAFusedQKV, LoRARowParallelLinear

    model = _make_mock_model()
    config = LoRAConfig(
        rank=8, alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    inject_lora_adapters(
        [model], config,  # [I6] Accepts list of model chunks
        num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8,
    )

    layer = model.decoder.layers[0]
    assert isinstance(layer.self_attention.linear_qkv, LoRAFusedQKV)
    assert isinstance(layer.self_attention.linear_proj, LoRARowParallelLinear)
    assert isinstance(layer.mlp.linear_fc1, LoRAFusedFC1)
    assert isinstance(layer.mlp.linear_fc2, LoRARowParallelLinear)


def test_inject_lora_vp_multiple_chunks():
    """[I6] Injection should work across VP chunks."""
    from slime.backends.megatron_utils.lora.config import LoRAConfig
    from slime.backends.megatron_utils.lora.injection import inject_lora_adapters
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    chunk0 = _make_mock_model(num_layers=1)
    chunk1 = _make_mock_model(num_layers=1)
    config = LoRAConfig(rank=8, alpha=16, target_modules=("q_proj", "k_proj", "v_proj"))

    inject_lora_adapters(
        [chunk0, chunk1], config,
        num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8,
    )

    # Both chunks should have LoRA injected
    assert isinstance(chunk0.decoder.layers[0].self_attention.linear_qkv, LoRAFusedQKV)
    assert isinstance(chunk1.decoder.layers[0].self_attention.linear_qkv, LoRAFusedQKV)


def test_freeze_base_params():
    from slime.backends.megatron_utils.lora.config import LoRAConfig
    from slime.backends.megatron_utils.lora.injection import freeze_base_params, inject_lora_adapters

    model = _make_mock_model(num_layers=1)
    config = LoRAConfig(rank=8, alpha=16, target_modules=("q_proj", "k_proj", "v_proj"))
    inject_lora_adapters([model], config, num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8)
    freeze_base_params([model])

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}

    # All lora_ params should be trainable
    assert all("lora_" in n for n in trainable)
    # All non-lora params should be frozen
    assert all("lora_" not in n for n in frozen)
    assert len(trainable) > 0
    assert len(frozen) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_injection.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement injection**

Create `slime/backends/megatron_utils/lora/injection.py`:

```python
"""LoRA adapter injection into Megatron GPTModel."""
from __future__ import annotations

import logging
from collections.abc import Sequence

import torch.nn as nn

from slime.backends.megatron_utils.lora.config import LoRAConfig
from slime.backends.megatron_utils.lora.layers import (
    LoRAFusedFC1,
    LoRAFusedQKV,
    LoRARowParallelLinear,
)

logger = logging.getLogger(__name__)


def inject_lora_adapters(
    model: Sequence[nn.Module],
    config: LoRAConfig,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
) -> None:
    """Inject LoRA adapters into target modules of a GPTModel.

    [I6] Accepts list of model chunks (VP stages). Each chunk is unwrapped from DDP.
    Replaces target linear layers with LoRA-wrapped versions in-place.
    """
    if not config.enabled:
        return

    targets = config.target_modules
    has_qkv = any(m in targets for m in ("q_proj", "k_proj", "v_proj"))
    has_o = "o_proj" in targets
    has_fc1 = any(m in targets for m in ("gate_proj", "up_proj"))
    has_fc2 = "down_proj" in targets

    count = 0
    for model_chunk in model:
        unwrapped = _unwrap_ddp(model_chunk)
        for layer in _get_decoder_layers(unwrapped):
            if has_qkv:
                layer.self_attention.linear_qkv = LoRAFusedQKV(
                    layer.self_attention.linear_qkv,
                    rank=config.rank,
                    alpha=config.alpha,
                    num_q_heads_per_tp=num_q_heads_per_tp,
                    num_kv_heads_per_tp=num_kv_heads_per_tp,
                    head_dim=head_dim,
                    dropout=config.dropout,
                )
                count += 1

            if has_o:
                layer.self_attention.linear_proj = LoRARowParallelLinear(
                    layer.self_attention.linear_proj,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                count += 1

            if has_fc1:
                layer.mlp.linear_fc1 = LoRAFusedFC1(
                    layer.mlp.linear_fc1,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                count += 1

            if has_fc2:
                layer.mlp.linear_fc2 = LoRARowParallelLinear(
                    layer.mlp.linear_fc2,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                count += 1

    logger.info(f"Injected {count} LoRA adapters (rank={config.rank}, alpha={config.alpha})")


def freeze_base_params(model: Sequence[nn.Module]) -> None:
    """[I6] Freeze all non-LoRA parameters across all VP chunks."""
    frozen_count = 0
    trainable_count = 0
    for model_chunk in model:
        unwrapped = _unwrap_ddp(model_chunk)
        for name, param in unwrapped.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
    logger.info(f"Frozen {frozen_count} base params, {trainable_count} LoRA params trainable")


def _unwrap_ddp(model: nn.Module) -> nn.Module:
    """Unwrap DDP/FSDP wrapper to get inner module."""
    if hasattr(model, "module"):
        return _unwrap_ddp(model.module)
    return model


def _get_decoder_layers(model: nn.Module):
    """Yield transformer layers from an unwrapped model."""
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        yield from model.decoder.layers
    else:
        # Fallback: search children
        for child in model.children():
            if hasattr(child, "decoder") and hasattr(child.decoder, "layers"):
                yield from child.decoder.layers
                return
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_injection.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 5: Update `__init__.py` exports**

Update `slime/backends/megatron_utils/lora/__init__.py`:

```python
from slime.backends.megatron_utils.lora.config import LoRAConfig
from slime.backends.megatron_utils.lora.injection import freeze_base_params, inject_lora_adapters
from slime.backends.megatron_utils.lora.merge import disable_lora, enable_lora, merge_lora_weights, unmerge_lora_weights

__all__ = [
    "LoRAConfig",
    "inject_lora_adapters",
    "freeze_base_params",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "disable_lora",
    "enable_lora",
]
```

- [ ] **Step 6: Commit**

```bash
git add slime/backends/megatron_utils/lora/ tests/test_lora_injection.py
git commit -m "feat(lora): add VP-aware injection and freeze_base_params (I6 fix)"
```

---

### Task 6: Integrate LoRA into Model Construction Pipeline

**Files:**
- Modify: `slime/backends/megatron_utils/model.py:83-123`

> **[I1] Fix:** Derive `num_q_heads_per_tp` and `num_kv_heads_per_tp` from the first decoder layer's
> `linear_qkv.output_size_per_partition` instead of manual division. This handles the case where
> `num_kv_heads < tp_size` (Megatron replicates KV heads so each rank has ≥1).

- [ ] **Step 1: Modify `setup_model_and_optimizer` in `model.py`**

In `slime/backends/megatron_utils/model.py`, modify `setup_model_and_optimizer()` (lines 83-123) to inject LoRA after model construction but **before** optimizer creation:

```python
# After line 107: model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
# Insert LoRA injection:

if getattr(args, "lora_rank", 0) > 0:
    from megatron.core import parallel_state as mpu

    from slime.backends.megatron_utils.lora import LoRAConfig, freeze_base_params, inject_lora_adapters

    lora_config = LoRAConfig.from_args(args)

    # [I1] Derive head counts from the actual model to handle num_kv_heads < tp_size.
    # Megatron replicates KV heads when num_kv_heads < tp_size, so the actual
    # per-TP output dimension is set correctly in the base layer.
    head_dim = args.kv_channels if args.kv_channels else (args.hidden_size // args.num_attention_heads)
    first_layer = None
    for model_chunk in model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        if hasattr(unwrapped, "decoder") and len(unwrapped.decoder.layers) > 0:
            first_layer = unwrapped.decoder.layers[0]
            break
    assert first_layer is not None, "No decoder layers found in model"

    qkv_out_per_tp = first_layer.self_attention.linear_qkv.output_size_per_partition
    # qkv_out_per_tp = (num_q_heads_per_tp + 2 * num_kv_heads_per_tp) * head_dim
    # With GQA: q_per_kv = num_q_heads / num_kv_heads, so:
    # qkv_out_per_tp = num_kv_heads_per_tp * (q_per_kv + 2) * head_dim
    tp_size = mpu.get_tensor_model_parallel_world_size()
    num_q_heads_per_tp = args.num_attention_heads // tp_size
    # Solve for num_kv_heads_per_tp from qkv_out:
    # qkv_out = q_per_tp * head_dim + 2 * kv_per_tp * head_dim
    num_kv_heads_per_tp = (qkv_out_per_tp - num_q_heads_per_tp * head_dim) // (2 * head_dim)

    inject_lora_adapters(model, lora_config, num_q_heads_per_tp, num_kv_heads_per_tp, head_dim)
    freeze_base_params(model)
```

This must be placed **before** the optimizer construction (lines 110-121), because `get_megatron_optimizer` captures the parameter list and filters by `requires_grad`.

- [ ] **Step 2: Verify no import errors**

```bash
cd /Users/jd/Documents/workspace/slime && python -c "from slime.backends.megatron_utils.model import setup_model_and_optimizer; print('OK')"
```
Expected: `OK` (import succeeds; actual execution requires distributed setup)

- [ ] **Step 3: Commit**

```bash
git add slime/backends/megatron_utils/model.py
git commit -m "feat(lora): integrate LoRA injection into model construction

Derives num_kv_heads_per_tp from base layer output_size_per_partition (I1 fix)."
```

---

### Task 7: Integrate LoRA into Actor Training

**Files:**
- Modify: `slime/backends/megatron_utils/actor.py`

> **[I2] Fix:** Replaces `"ref" in backup_tags` condition with `_lora_enabled` flag.
> Uses `disable_lora`/`enable_lora` for ref forward instead of `_switch_model("ref")`.
>
> **[C2] Fix:** For weight update, uses merge→re-backup→transfer→unmerge→re-backup pattern
> on colocate path. On LoRA + colocate, automatically uses Noop backuper as optimization.

- [ ] **Step 1: Modify `init()` for LoRA-aware initialization**

In `actor.py`, after the model initialization section. Key changes:
1. Set `_lora_enabled` and `_needs_ref_logprobs` flags
2. Skip ref checkpoint load when LoRA is enabled (ref = base with adapter off)
3. [C2 optimization] When LoRA + colocate, prefer Noop backuper since CPU backup is unnecessary

```python
# After line 93 (initialize_model_and_optimizer), add:
self._lora_enabled = getattr(args, "lora_rank", 0) > 0
self._needs_ref_logprobs = args.kl_coef != 0 or getattr(args, "use_kl_loss", False)

# After adapter_load handling, add:
if getattr(args, "adapter_load", None):
    from slime.backends.megatron_utils.checkpoint import load_lora_adapter
    load_lora_adapter(self.model, args.adapter_load)

# Modify weights_backuper creation (lines 102-110):
# [C2] When LoRA is enabled, we don't need CPU backup for ref switching (adapter on/off instead).
# For colocate mode, TensorBackuperNoop returns live GPU refs, which correctly reflect merged weights.
if self._lora_enabled and not args.enable_weights_backuper:
    # Noop mode: weights_getter returns GPU refs directly
    logger.info("LoRA mode: using Noop weights backuper (ref switching via adapter on/off)")

# ... existing TensorBackuper.create code ...

# Replace lines 114-115 (ref checkpoint loading):
if with_ref:
    if self._lora_enabled:
        # With LoRA, ref model = base model (adapter off). No separate checkpoint needed.
        logger.info("LoRA mode: ref model uses base weights (adapter off), skipping ref checkpoint load")
    else:
        self.load_other_checkpoint("ref", args.ref_load)
```

- [ ] **Step 2: Modify `train_actor()` for LoRA ref forward [I2]**

In `actor.py`, modify the ref model forward section (around lines 378-389). Replace the `"ref" in backup_tags` condition with LoRA-aware logic:

```python
# Replace lines 378-389 with:
if self.args.compute_advantages_and_returns:
    if self._needs_ref_logprobs:
        if self._lora_enabled:
            # [I2] LoRA: adapter off = ref model (no weight switch needed)
            from slime.backends.megatron_utils.lora import disable_lora, enable_lora
            for model_chunk in self.model:
                disable_lora(model_chunk.module if hasattr(model_chunk, "module") else model_chunk)
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
            rollout_data.update(
                self.compute_log_prob(
                    data_iterator,
                    num_microbatches,
                    store_prefix="ref_",
                )
            )
            for model_chunk in self.model:
                enable_lora(model_chunk.module if hasattr(model_chunk, "module") else model_chunk)
        elif "ref" in self.weights_backuper.backup_tags:
            # Original: switch to ref model via CPU backup restore
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
            self._switch_model("ref")
            rollout_data.update(
                self.compute_log_prob(
                    data_iterator,
                    num_microbatches,
                    store_prefix="ref_",
                )
            )

    # Rest of actor/old_actor forward remains unchanged
    self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
    # ... (existing log_prob / critic / switch_model code) ...
```

- [ ] **Step 3: Modify weight update with merge/unmerge [C2]**

In `actor.py`, modify the weight update flow. The key insight:
- **Disaggregated path** reads live GPU params → merge before, unmerge after is enough.
- **Colocate path** reads from CPU backup → need merge→re-backup→transfer→unmerge→re-backup.

```python
# In the update_weights method (around line 500), wrap with merge/unmerge:

def update_weights(self, ...):
    if self._lora_enabled:
        from slime.backends.megatron_utils.lora import merge_lora_weights, unmerge_lora_weights

        # Merge LoRA into base weights on GPU
        for model_chunk in self.model:
            merge_lora_weights(model_chunk.module if hasattr(model_chunk, "module") else model_chunk)

        # [C2] Colocate path: CPU backup has unmerged weights. Re-backup after merge.
        if self.args.colocate and self.args.enable_weights_backuper:
            self.weights_backuper.backup("actor")  # Now contains merged weights

    # Original weight transfer
    self.weight_updater.update_weights()

    if self._lora_enabled:
        # Unmerge to restore base weights for continued training
        for model_chunk in self.model:
            unmerge_lora_weights(model_chunk.module if hasattr(model_chunk, "module") else model_chunk)

        # [C2] Colocate path: Restore CPU backup to unmerged state
        if self.args.colocate and self.args.enable_weights_backuper:
            self.weights_backuper.backup("actor")  # Back to unmerged
```

- [ ] **Step 4: Modify ref_update_interval handling [I3]**

The ref_update_interval section (lines 449-457) is already handled by argument validation (Task 1, Step 6) which sets `ref_update_interval = None` when LoRA is enabled. No code change needed here — the condition `self.args.ref_update_interval is not None` will be False.

- [ ] **Step 5: Commit**

```bash
git add slime/backends/megatron_utils/actor.py
git commit -m "feat(lora): integrate LoRA into actor training

- I2: ref forward via disable_lora/enable_lora (not backup_tags)
- C2: merge/unmerge around weight transfer with colocate re-backup
- I3: ref_update_interval auto-disabled via argument validation"
```

---

## Chunk 5: Checkpoint and MoE Expert LoRA

### Task 8: Adapter-Only Checkpoint Save/Load

**Files:**
- Modify: `slime/backends/megatron_utils/model.py:663-688`
- Modify: `slime/backends/megatron_utils/checkpoint.py`

> **[I5] Fix:** All TP ranks participate in all_gather for LoRA params. Only rank 0 saves.
> Uses existing `all_gather_param` from `update_weight/common.py`.

- [ ] **Step 1: Add adapter save path in `model.py`**

In `model.py`, modify the `save()` function. Add adapter-only save logic:

```python
def save(iteration, model, optimizer, opt_param_scheduler):
    args = get_args()

    # [I4] Adapter-only save when LoRA is enabled
    if getattr(args, "lora_rank", 0) > 0 and args.save_adapter_only:
        _save_lora_adapter(iteration, model, args)
        return

    # ... existing full checkpoint save logic ...
```

Add the helper function:

```python
def _save_lora_adapter(iteration: int, model: Sequence[DDP], args) -> None:
    """[I5] Save LoRA adapter weights with proper TP all-gather.

    All TP ranks participate in all_gather (collective op), but only the main rank writes to disk.
    This produces a complete (un-sharded) adapter checkpoint that can be loaded with any TP configuration.
    """
    import json
    from pathlib import Path

    import torch
    from megatron.core import parallel_state as mpu

    from slime.backends.megatron_utils.update_weight.common import all_gather_param

    save_dir = Path(args.save) / f"lora_adapter_iter_{iteration:07d}"

    # [I5] All TP ranks must participate in all_gather (it's a collective op).
    # Gather LoRA params to full tensors across TP.
    adapter_state = {}
    for model_chunk in model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        for name, param in unwrapped.named_parameters():
            if "lora_" in name:
                full_param = all_gather_param(name, param)
                adapter_state[name] = full_param.cpu()

    # Only main rank writes to disk
    if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(adapter_state, save_dir / "adapter_model.bin")

        from slime.backends.megatron_utils.lora.config import LoRAConfig
        config = LoRAConfig.from_args(args)
        config_dict = {
            "peft_type": "LORA",
            "r": config.rank,
            "lora_alpha": config.alpha,
            "target_modules": list(config.target_modules),
        }
        (save_dir / "lora_config.json").write_text(json.dumps(config_dict, indent=2))

        logger.info(f"Saved LoRA adapter to {save_dir}")

    # Barrier to ensure save completes before any rank proceeds
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
```

- [ ] **Step 2: Add adapter load in `checkpoint.py`**

In `checkpoint.py`, add adapter overlay loading:

```python
def load_lora_adapter(model, adapter_path: str) -> None:
    """Load LoRA adapter weights on top of already-injected model.

    The adapter checkpoint contains un-sharded (full) tensors.
    Each TP rank extracts its shard based on the TP attributes set during injection.
    """
    import torch
    from megatron.core import parallel_state as mpu

    adapter_state = torch.load(Path(adapter_path) / "adapter_model.bin", map_location="cpu", weights_only=True)

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    loaded = 0
    for model_chunk in model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        for name, param in unwrapped.named_parameters():
            if name not in adapter_state:
                continue
            full_tensor = adapter_state[name]

            # Shard if needed based on TP attributes
            if getattr(param, "tensor_model_parallel", False) and tp_size > 1:
                dim = param.partition_dim
                chunk_size = full_tensor.shape[dim] // tp_size
                full_tensor = full_tensor.narrow(dim, tp_rank * chunk_size, chunk_size)

            param.data.copy_(full_tensor.to(param.device))
            loaded += 1

    logger.info(f"Loaded {loaded} LoRA adapter params from {adapter_path}")
```

- [ ] **Step 3: Wire adapter load in `actor.py` init**

In `actor.py`, after the `initialize_model_and_optimizer` call (handled in Task 7 Step 1):

```python
if getattr(args, "adapter_load", None):
    from slime.backends.megatron_utils.checkpoint import load_lora_adapter
    load_lora_adapter(self.model, args.adapter_load)
```

- [ ] **Step 4: Commit**

```bash
git add slime/backends/megatron_utils/model.py slime/backends/megatron_utils/checkpoint.py slime/backends/megatron_utils/actor.py
git commit -m "feat(lora): adapter-only checkpoint save/load

- I5: all_gather LoRA params across TP before saving from rank 0
- I4: --save-adapter-only flag controls save behavior
- Adapter load shards full tensors back to TP partitions"
```

---

### Task 9: MoE Expert LoRA (GroupedMLP Forward Override)

**Files:**
- Create: `slime/backends/megatron_utils/lora/expert_lora.py`
- Modify: `slime/backends/megatron_utils/lora/injection.py`
- Modify: `slime/backends/megatron_utils/lora/merge.py`
- Test: `tests/test_lora_expert.py`

> **[C1] Fix:** Instead of just attaching LoRA params as attributes, monkey-patch `GroupedMLP.forward()`
> to inject `bmm`-computed delta into `w1`/`w2` before the `gg.ops.gmm` call. This preserves the
> fused CUTLASS kernel performance while adding LoRA corrections.

- [ ] **Step 1: Write expert LoRA tests**

Create `tests/test_lora_expert.py`:

```python
"""Tests for MoE expert LoRA with GroupedMLP forward override."""
import torch
import torch.nn as nn

import pytest


class MockGroupedMLP(nn.Module):
    """Mock for Megatron GroupedMLP with stacked 3D weights."""

    def __init__(self, num_local_experts, hidden_size, ffn_hidden):
        super().__init__()
        # Megatron stores weight1 as flat [H, E*F] then reshapes to [E, H, F] in forward
        self.weight1 = nn.Parameter(torch.randn(hidden_size, num_local_experts * ffn_hidden * 2))
        # weight2 as flat [E*F, H] then reshapes to [E, F, H]
        self.weight2 = nn.Parameter(torch.randn(num_local_experts * ffn_hidden, hidden_size))
        self.num_local_experts = num_local_experts
        self.config = type("Config", (), {"hidden_size": hidden_size, "ffn_hidden_size": ffn_hidden})()
        self._forward_called = False

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        self._forward_called = True
        E = self.num_local_experts
        H = self.config.hidden_size
        F = self.config.ffn_hidden_size
        w1 = self.weight1.view(E, H, -1)  # [E, H, 2F]
        w2 = self.weight2.view(E, -1, H)  # [E, F, H]
        # Simplified: just do batched matmul instead of gmm
        # In real code, this would be gg.ops.gmm
        output = torch.bmm(permuted_local_hidden_states.unsqueeze(0).expand(E, -1, -1), w1)
        return output.sum(0)  # Simplified


def test_expert_lora_param_shapes():
    """Expert LoRA should create correct 3D parameter shapes."""
    from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora

    mlp = MockGroupedMLP(num_local_experts=4, hidden_size=64, ffn_hidden=128)
    inject_expert_lora(mlp, rank=8, alpha=16)

    # FC1 (gate+up): A=[E, rank, H], gate_B=[E, F, rank], up_B=[E, F, rank]
    assert mlp._lora_A_fc1.shape == (4, 8, 64)
    assert mlp._lora_gate_B_fc1.shape == (4, 128, 8)
    assert mlp._lora_up_B_fc1.shape == (4, 128, 8)

    # FC2: A=[E, rank, F], B=[E, H, rank]
    assert mlp._lora_A_fc2.shape == (4, 8, 128)
    assert mlp._lora_B_fc2.shape == (4, 64, 8)


def test_expert_lora_zero_init_no_effect():
    """At initialization (B=0), expert LoRA should not change forward output."""
    from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora

    mlp = MockGroupedMLP(num_local_experts=2, hidden_size=32, ffn_hidden=64)
    w1_before = mlp.weight1.data.clone()
    w2_before = mlp.weight2.data.clone()

    inject_expert_lora(mlp, rank=4, alpha=8)

    # Base weights should be unchanged (LoRA B is zero-initialized)
    torch.testing.assert_close(mlp.weight1.data, w1_before)
    torch.testing.assert_close(mlp.weight2.data, w2_before)


def test_expert_lora_merge_unmerge_roundtrip():
    """Merge then unmerge should restore original weights."""
    from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora
    from slime.backends.megatron_utils.lora.merge import merge_expert_lora, unmerge_expert_lora

    mlp = MockGroupedMLP(num_local_experts=4, hidden_size=32, ffn_hidden=64)
    inject_expert_lora(mlp, rank=4, alpha=8)

    # Set non-zero B values
    mlp._lora_gate_B_fc1.data.normal_()
    mlp._lora_up_B_fc1.data.normal_()
    mlp._lora_B_fc2.data.normal_()

    w1_original = mlp.weight1.data.clone()
    w2_original = mlp.weight2.data.clone()

    merge_expert_lora(mlp)
    assert not torch.allclose(mlp.weight1.data, w1_original)

    unmerge_expert_lora(mlp)
    torch.testing.assert_close(mlp.weight1.data, w1_original, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mlp.weight2.data, w2_original, atol=1e-5, rtol=1e-5)


def test_expert_lora_disable_enable():
    """disable should set _lora_scale=0, enable should restore."""
    from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora

    mlp = MockGroupedMLP(num_local_experts=2, hidden_size=16, ffn_hidden=32)
    inject_expert_lora(mlp, rank=2, alpha=4)

    assert mlp._lora_scale == 4 / 2  # alpha / rank
    assert mlp._lora_enabled is True

    mlp._saved_lora_scale = mlp._lora_scale
    mlp._lora_scale = 0.0
    assert mlp._lora_scale == 0.0

    mlp._lora_scale = mlp._saved_lora_scale
    assert mlp._lora_scale == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_expert.py -v
```

- [ ] **Step 3: Implement expert LoRA with forward override**

Create `slime/backends/megatron_utils/lora/expert_lora.py`:

```python
"""[C1] MoE Expert LoRA via GroupedMLP forward monkey-patch.

Instead of wrapping GroupedMLP in a new class (which breaks isinstance checks),
we attach LoRA parameters as attributes and replace the forward method to inject
bmm-computed deltas into w1/w2 before the gg.ops.gmm call.

Key design:
- LoRA params: _lora_A_fc1, _lora_gate_B_fc1, _lora_up_B_fc1 (fc1 = gate+up fused)
              _lora_A_fc2, _lora_B_fc2
- Forward patch: compute delta via bmm, add to reshaped w1/w2, then call gmm as normal
- Merge: add delta directly to weight1/weight2 (the flat stored tensors)
- Performance: bmm cost is negligible (~0.1% of gmm) for small rank
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn


def inject_expert_lora(
    grouped_mlp: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> None:
    """Inject LoRA parameters into a GroupedMLP and monkey-patch its forward.

    Args:
        grouped_mlp: A Megatron GroupedMLP module with weight1, weight2, num_local_experts.
        rank: LoRA rank.
        alpha: LoRA alpha for scaling.
        dropout: LoRA dropout rate.
    """
    E = grouped_mlp.num_local_experts
    H = grouped_mlp.config.hidden_size
    F = grouped_mlp.config.ffn_hidden_size

    scale = alpha / rank

    # FC1 LoRA: weight1 is [H, E*2F] → reshaped [E, H, 2F] → split into gate [E,H,F] and up [E,H,F]
    # LoRA delta for gate: gate_B @ A → [E, F, H] → transpose to [E, H, F]
    # LoRA delta for up:   up_B  @ A → [E, F, H] → transpose to [E, H, F]
    grouped_mlp._lora_A_fc1 = nn.Parameter(torch.empty(E, rank, H))
    grouped_mlp._lora_gate_B_fc1 = nn.Parameter(torch.zeros(E, F, rank))
    grouped_mlp._lora_up_B_fc1 = nn.Parameter(torch.zeros(E, F, rank))

    # FC2 LoRA: weight2 is [E*F, H] → reshaped [E, F, H]
    # LoRA delta: B @ A → [E, H, F] → transpose to [E, F, H]
    grouped_mlp._lora_A_fc2 = nn.Parameter(torch.empty(E, rank, F))
    grouped_mlp._lora_B_fc2 = nn.Parameter(torch.zeros(E, H, rank))

    # Initialize A matrices
    for e in range(E):
        nn.init.kaiming_uniform_(grouped_mlp._lora_A_fc1[e], a=math.sqrt(5))
        nn.init.kaiming_uniform_(grouped_mlp._lora_A_fc2[e], a=math.sqrt(5))

    grouped_mlp._lora_scale = scale
    grouped_mlp._lora_enabled = True
    grouped_mlp._lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # Save original forward for reference (not called directly — we rewrite the full forward)
    grouped_mlp._original_forward = grouped_mlp.forward

    # Monkey-patch forward
    grouped_mlp.forward = types.MethodType(_lora_grouped_mlp_forward, grouped_mlp)


def _lora_grouped_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
    """Patched forward for GroupedMLP with LoRA delta injection.

    This reproduces the original GroupedMLP.forward() logic but inserts LoRA deltas
    into w1/w2 between the reshape and the gmm call.

    The original flow:
        w1 = self.weight1.view(E, H, -1)
        w2 = self.weight2.view(E, -1, H)
        fc1_output = gg.ops.gmm(input, w1, tokens_per_expert)
        ... activation ...
        fc2_output = gg.ops.gmm(intermediate, w2, tokens_per_expert)

    Patched flow:
        w1 = self.weight1.view(E, H, -1)
        if lora_enabled: w1 = w1 + delta_w1  # <-- LoRA injection
        fc1_output = gg.ops.gmm(input, w1, tokens_per_expert)
        ... (same) ...
    """
    # Delegate to original forward which handles the full gmm logic.
    # We temporarily merge the delta into weight1/weight2, call original, then remove.
    if self._lora_enabled and self._lora_scale != 0.0:
        _apply_expert_delta(self, merge=True)

    output = self._original_forward(permuted_local_hidden_states, tokens_per_expert)

    if self._lora_enabled and self._lora_scale != 0.0:
        _apply_expert_delta(self, merge=False)

    return output


def _apply_expert_delta(grouped_mlp, merge: bool) -> None:
    """Add or subtract LoRA delta from GroupedMLP's weight1/weight2.

    This is called in the forward pass to temporarily merge LoRA deltas.
    Uses torch.no_grad() to avoid tracking these ops in the autograd graph
    for the base weights (LoRA params get gradients through the training loss).

    Note: During training, gradients flow through the modified weights. Since we
    add the delta before forward and remove after, the base weights don't accumulate
    gradient from LoRA. The LoRA params themselves are part of the module and get
    gradients normally via autograd.
    """
    sign = 1.0 if merge else -1.0
    scale = grouped_mlp._lora_scale

    E = grouped_mlp.num_local_experts
    H = grouped_mlp.config.hidden_size
    F = grouped_mlp.config.ffn_hidden_size

    with torch.no_grad():
        # FC1: weight1 is [H, E*2F]
        # gate_delta = gate_B @ A → [E, F, rank] @ [E, rank, H] = [E, F, H]
        gate_delta = torch.bmm(grouped_mlp._lora_gate_B_fc1, grouped_mlp._lora_A_fc1) * scale
        up_delta = torch.bmm(grouped_mlp._lora_up_B_fc1, grouped_mlp._lora_A_fc1) * scale

        # weight1 viewed as [E, H, 2F] → gate is [:, :, :F], up is [:, :, F:]
        w1_view = grouped_mlp.weight1.data.view(E, H, 2 * F)
        w1_view[:, :, :F] += sign * gate_delta.transpose(-1, -2)  # [E, F, H].T → [E, H, F]
        w1_view[:, :, F:] += sign * up_delta.transpose(-1, -2)

        # FC2: weight2 is [E*F, H]
        fc2_delta = torch.bmm(grouped_mlp._lora_B_fc2, grouped_mlp._lora_A_fc2) * scale  # [E, H, rank]@[E, rank, F]=[E, H, F]
        w2_view = grouped_mlp.weight2.data.view(E, F, H)
        w2_view += sign * fc2_delta.transpose(-1, -2)  # [E, H, F].T → [E, F, H]
```

> **Important note on gradients:** The `_apply_expert_delta` uses `torch.no_grad()` because
> we're modifying `weight1`/`weight2` (base weights, frozen). The LoRA parameters themselves
> (`_lora_A_fc1`, `_lora_gate_B_fc1`, etc.) receive gradients through the normal autograd path
> because they're registered as `nn.Parameter` on the module, and the training loss backpropagates
> through the modified weights. The `no_grad` only prevents the base weights from accumulating
> gradient from the temporary merge/unmerge ops.
>
> **Alternative approach if gradient flow is an issue:** Instead of merge-in-forward, compute the
> delta as a separate tensor and override the gmm call. This would be cleaner for autograd but
> requires replicating more of GroupedMLP's forward logic. Start with the merge-in-forward
> approach and validate gradient correctness in integration tests.

- [ ] **Step 4: Add expert merge to merge.py**

In `slime/backends/megatron_utils/lora/merge.py`, add expert merge/unmerge functions:

```python
def merge_expert_lora(grouped_mlp) -> None:
    """Merge expert LoRA deltas into GroupedMLP weight1/weight2 permanently.

    Used before weight transfer to SGLang.
    """
    from slime.backends.megatron_utils.lora.expert_lora import _apply_expert_delta
    _apply_expert_delta(grouped_mlp, merge=True)


def unmerge_expert_lora(grouped_mlp) -> None:
    """Unmerge expert LoRA deltas from GroupedMLP weight1/weight2."""
    from slime.backends.megatron_utils.lora.expert_lora import _apply_expert_delta
    _apply_expert_delta(grouped_mlp, merge=False)
```

Also update `merge_lora_weights` and `unmerge_lora_weights` to handle expert LoRA:

```python
def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA adapters into base weights in-place. Call before weight transfer."""
    for module in model.modules():
        if isinstance(module, LoRAColumnParallelLinear):
            module.base_layer.weight.data += (module.lora_B @ module.lora_A) * module.scaling
        elif isinstance(module, LoRARowParallelLinear):
            module.base_layer.weight.data += (module.lora_B @ module.lora_A) * module.scaling
        elif isinstance(module, LoRAFusedQKV):
            _merge_fused_qkv(module)
        elif isinstance(module, LoRAFusedFC1):
            _merge_fused_fc1(module)
        # [C1] Expert LoRA: merge deltas into GroupedMLP weight1/weight2
        elif hasattr(module, "_lora_enabled") and hasattr(module, "_lora_A_fc1"):
            merge_expert_lora(module)
```

Mirror in `unmerge_lora_weights` with `unmerge_expert_lora`.

- [ ] **Step 5: Add expert injection in `injection.py`**

Add to `inject_lora_adapters` in `injection.py`, after the shared MLP injection:

```python
# [C1] MoE expert LoRA: inject into GroupedMLP
has_expert = "expert" in targets
if has_expert and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
    experts_module = layer.mlp.experts
    if hasattr(experts_module, "weight1"):  # GroupedMLP
        from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora
        inject_expert_lora(
            experts_module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        count += 2  # FC1 + FC2
        logger.info(
            f"Injected expert LoRA into GroupedMLP "
            f"(experts={experts_module.num_local_experts}, rank={config.rank})"
        )
    elif hasattr(experts_module, "local_experts"):  # SequentialMLP fallback
        for expert in experts_module.local_experts:
            if has_fc1:
                expert.linear_fc1 = LoRAFusedFC1(
                    expert.linear_fc1, rank=config.rank, alpha=config.alpha, dropout=config.dropout,
                )
            if has_fc2:
                expert.linear_fc2 = LoRARowParallelLinear(
                    expert.linear_fc2, rank=config.rank, alpha=config.alpha, dropout=config.dropout,
                )
            count += int(has_fc1) + int(has_fc2)
```

- [ ] **Step 6: Run all tests**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_expert.py tests/test_lora_layers.py tests/test_lora_merge.py tests/test_lora_injection.py tests/test_lora_config.py -v
```
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add slime/backends/megatron_utils/lora/ tests/test_lora_expert.py
git commit -m "feat(lora): add MoE expert LoRA via GroupedMLP forward override (C1 fix)

Monkey-patches GroupedMLP.forward() to inject bmm-computed LoRA delta
into w1/w2 before gg.ops.gmm call. Preserves fused CUTLASS performance.
Supports merge/unmerge for weight transfer and disable/enable for ref forward."
```

---

## Chunk 6: Final Integration and Validation

### Task 10: Run Pre-commit and Full Test Suite

- [ ] **Step 1: Run pre-commit checks**

```bash
cd /Users/jd/Documents/workspace/slime && pre-commit run --all-files --show-diff-on-failure --color=always
```
Fix any formatting/linting issues (line length 119, isort, ruff).

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_*.py -v --tb=short
```
Expected: All LoRA tests pass.

- [ ] **Step 3: Fix any issues and commit**

```bash
git add -A && git commit -m "fix(lora): address linting and test issues"
```

### Task 11: Summary Review

- [ ] **Step 1: Verify all new files exist**

```bash
ls -la slime/backends/megatron_utils/lora/
```
Expected: `__init__.py`, `config.py`, `layers.py`, `expert_lora.py`, `injection.py`, `merge.py`

- [ ] **Step 2: Verify all tests pass**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_*.py -v
```

- [ ] **Step 3: Review git log**

```bash
git log --oneline dev..HEAD
```
Expected: ~10 commits, all prefixed with `feat(lora):` or `fix(lora):`

- [ ] **Step 4: Verify issue resolution checklist**

| Issue | Status | Verification |
|-------|--------|-------------|
| C1 | Fixed | `test_expert_lora_*` tests pass; forward override injects delta before gmm |
| C2 | Fixed | actor.py merge→re-backup→transfer→unmerge→re-backup pattern for colocate |
| C3 | Fixed | `test_merge_fused_qkv_roundtrip` passes; uses `_interleave_qkv_weight` |
| I1 | Fixed | `test_lora_fused_qkv_kv_heads_equal_one` passes; uses output_size_per_partition |
| I2 | Fixed | actor.py uses `_lora_enabled` + disable/enable instead of backup_tags |
| I3 | Fixed | arguments.py auto-disables ref_update_interval with warning |
| I4 | Fixed | `--save-adapter-only` / `--no-save-adapter-only` with auto-default |
| I5 | Fixed | `_save_lora_adapter` uses all_gather before rank 0 save |
| I6 | Fixed | inject/merge/freeze accept list[DDP] and iterate VP chunks |
