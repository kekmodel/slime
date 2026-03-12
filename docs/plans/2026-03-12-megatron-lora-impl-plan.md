# Megatron LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LoRA (Low-Rank Adaptation) training to slime's Megatron backend with merge-based weight transfer, including MoE expert LoRA support.

**Architecture:** LoRA adapters wrap existing Megatron parallel linear layers. Base weights are frozen; only LoRA params receive gradients. Before weight transfer to SGLang, LoRA weights are merged into base weights so SGLang runs in normal mode (zero inference overhead). Ref model in RL uses adapter-off (scaling=0) instead of CPU weight swap.

**Tech Stack:** PyTorch, Megatron-Core (ColumnParallelLinear, RowParallelLinear, GroupedMLP), slime Megatron backend

**Spec:** `docs/plans/2026-03-12-megatron-lora-training-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `slime/backends/megatron_utils/lora/__init__.py` | Package exports |
| `slime/backends/megatron_utils/lora/config.py` | `LoRAConfig` dataclass, argument registration |
| `slime/backends/megatron_utils/lora/layers.py` | TP-compatible LoRA wrapper layers (2D + 3D) |
| `slime/backends/megatron_utils/lora/injection.py` | `inject_lora_adapters()`, `freeze_base_params()` |
| `slime/backends/megatron_utils/lora/merge.py` | `merge_lora_weights()`, `unmerge_lora_weights()`, `disable_lora()`, `enable_lora()` |
| `tests/test_lora_layers.py` | Unit tests for LoRA layer correctness |
| `tests/test_lora_merge.py` | Unit tests for merge/unmerge roundtrip |
| `tests/test_lora_injection.py` | Unit tests for injection + freeze |

### Modified Files

| File | Changes |
|------|---------|
| `slime/utils/arguments.py` | Add LoRA CLI arguments |
| `slime/backends/megatron_utils/model.py:83-123` | Inject LoRA in `setup_model_and_optimizer()` |
| `slime/backends/megatron_utils/actor.py:370-460` | LoRA ref forward in `train_actor()` |
| `slime/backends/megatron_utils/actor.py:484-523` | Merge/unmerge around `update_weights()` |
| `slime/backends/megatron_utils/actor.py:48-154` | Skip ref backup in `init()` when LoRA |
| `slime/backends/megatron_utils/model.py:663-688` | Adapter-only save path in `save()` |
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
from dataclasses import dataclass, field

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
    group.add_argument(
        "--save-adapter-only",
        action="store_true",
        default=False,
        help="Save only adapter weights in checkpoint. Default when lora_rank > 0.",
    )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_config.py -v
```
Expected: All 4 tests PASS

- [ ] **Step 6: Add LoRA args to slime argument parser**

In `slime/utils/arguments.py`, find where custom argument groups are added (near the end of the `parse_args` function or in `_add_slime_args`) and add:

```python
from slime.backends.megatron_utils.lora.config import add_lora_args
```

Call `add_lora_args(parser)` in the argument registration section. Also add validation:
- If `--lora-rank > 0` and `--lora-target-modules` is None, set to `DEFAULT_TARGET_MODULES`
- `--lora-rank > 0` is incompatible with `--only-train-params-name-list` (if it exists in Megatron args)

- [ ] **Step 7: Commit**

```bash
git add slime/backends/megatron_utils/lora/ tests/test_lora_config.py slime/utils/arguments.py
git commit -m "feat(lora): add LoRAConfig and CLI arguments"
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
    # base=0, lora = x @ A^T @ B^T * 2.0 = [1,1,1,1] @ [[1,0],[0,1],[0,0],[0,0]] @ [[1,0,0,0],[0,1,0,0]] * 2
    # = [1,1,0,0] @ [[1,0,0,0],[0,1,0,0]] * 2 = [1,1,0,0] * 2 = [2,2,0,0]
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
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add slime/backends/megatron_utils/lora/layers.py tests/test_lora_layers.py
git commit -m "feat(lora): add LoRAFusedQKV and LoRAFusedFC1 layers"
```

---

## Chunk 3: Merge/Unmerge and Disable/Enable

### Task 4: Merge and Disable Operations

**Files:**
- Create: `slime/backends/megatron_utils/lora/merge.py`
- Test: `tests/test_lora_merge.py`

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

import torch.nn as nn

from slime.backends.megatron_utils.lora.layers import (
    LoRAColumnParallelLinear,
    LoRAFusedFC1,
    LoRAFusedQKV,
    LoRARowParallelLinear,
)

_LORA_LAYER_TYPES = (LoRAColumnParallelLinear, LoRARowParallelLinear, LoRAFusedQKV, LoRAFusedFC1)


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA adapters into base weights in-place. Call before weight transfer."""
    for module in model.modules():
        if isinstance(module, LoRAColumnParallelLinear):
            # W += B @ A * scale
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
    """Merge Q/K/V LoRA into fused QKV weight in interleaved layout."""
    from slime.backends.megatron_utils.lora.layers import _interleave_qkv

    q_delta = (module.q_lora_B @ module.q_lora_A) * module.scaling  # [q_out, H]
    k_delta = (module.k_lora_B @ module.k_lora_A) * module.scaling  # [kv_out, H]
    v_delta = (module.v_lora_B @ module.v_lora_A) * module.scaling  # [kv_out, H]

    # Interleave to match base weight layout: [QKV_dim/T, H]
    # _interleave_qkv expects [..., dim] format, here we use [dim, H] → treat H as batch
    delta = _interleave_qkv(
        q_delta.unsqueeze(0), k_delta.unsqueeze(0), v_delta.unsqueeze(0),
        module.num_q_heads_per_tp, module.num_kv_heads_per_tp, module.head_dim,
    ).squeeze(0)

    module.base_layer.weight.data += delta


def _unmerge_fused_qkv(module: LoRAFusedQKV) -> None:
    from slime.backends.megatron_utils.lora.layers import _interleave_qkv

    q_delta = (module.q_lora_B @ module.q_lora_A) * module.scaling
    k_delta = (module.k_lora_B @ module.k_lora_A) * module.scaling
    v_delta = (module.v_lora_B @ module.v_lora_A) * module.scaling

    delta = _interleave_qkv(
        q_delta.unsqueeze(0), k_delta.unsqueeze(0), v_delta.unsqueeze(0),
        module.num_q_heads_per_tp, module.num_kv_heads_per_tp, module.head_dim,
    ).squeeze(0)

    module.base_layer.weight.data -= delta


def _merge_fused_fc1(module: LoRAFusedFC1) -> None:
    """Merge gate/up LoRA into fused FC1 weight."""
    gate_delta = (module.gate_lora_B @ module.gate_lora_A) * module.scaling
    up_delta = (module.up_lora_B @ module.up_lora_A) * module.scaling
    import torch
    delta = torch.cat([gate_delta, up_delta], dim=0)
    module.base_layer.weight.data += delta


def _unmerge_fused_fc1(module: LoRAFusedFC1) -> None:
    gate_delta = (module.gate_lora_B @ module.gate_lora_A) * module.scaling
    up_delta = (module.up_lora_B @ module.up_lora_A) * module.scaling
    import torch
    delta = torch.cat([gate_delta, up_delta], dim=0)
    module.base_layer.weight.data -= delta


def disable_lora(model: nn.Module) -> None:
    """Set all LoRA scaling to 0 (model behaves as base only). Used for ref forward."""
    for module in model.modules():
        if isinstance(module, _LORA_LAYER_TYPES):
            module._saved_scaling = module.scaling
            module.scaling = 0.0


def enable_lora(model: nn.Module) -> None:
    """Restore LoRA scaling after disable_lora."""
    for module in model.modules():
        if isinstance(module, _LORA_LAYER_TYPES):
            if hasattr(module, "_saved_scaling"):
                module.scaling = module._saved_scaling
                del module._saved_scaling
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_merge.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add slime/backends/megatron_utils/lora/merge.py tests/test_lora_merge.py
git commit -m "feat(lora): add merge/unmerge and disable/enable utilities"
```

---

## Chunk 4: Injection and Integration

### Task 5: LoRA Injection into Model

**Files:**
- Create: `slime/backends/megatron_utils/lora/injection.py`
- Test: `tests/test_lora_injection.py`

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
        model, config,
        num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8,
    )

    layer = model.decoder.layers[0]
    assert isinstance(layer.self_attention.linear_qkv, LoRAFusedQKV)
    assert isinstance(layer.self_attention.linear_proj, LoRARowParallelLinear)
    assert isinstance(layer.mlp.linear_fc1, LoRAFusedFC1)
    assert isinstance(layer.mlp.linear_fc2, LoRARowParallelLinear)


def test_freeze_base_params():
    from slime.backends.megatron_utils.lora.config import LoRAConfig
    from slime.backends.megatron_utils.lora.injection import freeze_base_params, inject_lora_adapters

    model = _make_mock_model(num_layers=1)
    config = LoRAConfig(rank=8, alpha=16, target_modules=("q_proj", "k_proj", "v_proj"))
    inject_lora_adapters(model, config, num_q_heads_per_tp=8, num_kv_heads_per_tp=2, head_dim=8)
    freeze_base_params(model)

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

import torch.nn as nn

from slime.backends.megatron_utils.lora.config import LoRAConfig
from slime.backends.megatron_utils.lora.layers import (
    LoRAColumnParallelLinear,
    LoRAFusedFC1,
    LoRAFusedQKV,
    LoRARowParallelLinear,
)

logger = logging.getLogger(__name__)


def inject_lora_adapters(
    model: nn.Module,
    config: LoRAConfig,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
) -> None:
    """Inject LoRA adapters into target modules of a GPTModel.

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
    for layer in _get_decoder_layers(model):
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


def freeze_base_params(model: nn.Module) -> None:
    """Freeze all non-LoRA parameters."""
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"Frozen {frozen_count} base params, {trainable_count} LoRA params trainable")


def _get_decoder_layers(model: nn.Module):
    """Yield transformer layers from model, handling DDP wrapping."""
    # Handle DDP wrapper: model may be a list of DDP-wrapped chunks
    if hasattr(model, "decoder"):
        yield from model.decoder.layers
    elif hasattr(model, "module"):
        yield from _get_decoder_layers(model.module)
    else:
        # Try to find decoder.layers in any submodule
        for child in model.children():
            if hasattr(child, "decoder"):
                yield from child.decoder.layers
                return
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_injection.py -v
```
Expected: All 2 tests PASS

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
git commit -m "feat(lora): add injection and freeze_base_params"
```

---

### Task 6: Integrate LoRA into Model Construction Pipeline

**Files:**
- Modify: `slime/backends/megatron_utils/model.py:83-123`

- [ ] **Step 1: Modify `setup_model_and_optimizer` in `model.py`**

In `slime/backends/megatron_utils/model.py`, modify `setup_model_and_optimizer()` (lines 83-123) to inject LoRA after model construction but **before** optimizer creation:

```python
# After line 107: model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
# Insert LoRA injection:

if getattr(args, "lora_rank", 0) > 0:
    from megatron.core import parallel_state as mpu

    from slime.backends.megatron_utils.lora import LoRAConfig, freeze_base_params, inject_lora_adapters

    lora_config = LoRAConfig.from_args(args)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    num_q_heads_per_tp = args.num_attention_heads // tp_size
    num_kv_heads_per_tp = (args.num_query_groups or args.num_attention_heads) // tp_size
    head_dim = args.hidden_size // args.num_attention_heads

    for model_chunk in model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        inject_lora_adapters(unwrapped, lora_config, num_q_heads_per_tp, num_kv_heads_per_tp, head_dim)
        freeze_base_params(unwrapped)
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
git commit -m "feat(lora): integrate LoRA injection into model construction pipeline"
```

---

### Task 7: Integrate LoRA into Actor Training

**Files:**
- Modify: `slime/backends/megatron_utils/actor.py`

- [ ] **Step 1: Modify `init()` to skip ref backup when LoRA is enabled**

In `actor.py`, after line 112 (`self.weights_backuper.backup("actor")`), add LoRA-aware ref handling:

```python
# Replace lines 114-115:
#   if with_ref:
#       self.load_other_checkpoint("ref", args.ref_load)
# With:
if with_ref:
    if getattr(args, "lora_rank", 0) > 0:
        # With LoRA, ref model = base model (adapter off). No separate checkpoint needed.
        # We still set the tag to signal ref is available, but don't load a separate checkpoint.
        logger.info("LoRA mode: ref model uses base weights (adapter off), skipping ref checkpoint load")
    else:
        self.load_other_checkpoint("ref", args.ref_load)
```

- [ ] **Step 2: Modify `train_actor()` for LoRA ref forward**

In `actor.py`, modify the ref model forward section (around lines 379-389):

```python
# Replace line 382:
#   self._switch_model("ref")
# With LoRA-aware ref:
if getattr(self.args, "lora_rank", 0) > 0:
    from slime.backends.megatron_utils.lora import disable_lora
    disable_lora_models = [m.module if hasattr(m, "module") else m for m in self.model]
    for m in disable_lora_models:
        disable_lora(m)
else:
    self._switch_model("ref")
```

After the ref log_probs computation (around line 389), restore LoRA:

```python
# After the ref compute_log_prob block, before line 390:
if getattr(self.args, "lora_rank", 0) > 0:
    from slime.backends.megatron_utils.lora import enable_lora
    for m in disable_lora_models:
        enable_lora(m)
```

- [ ] **Step 3: Modify `update_weights()` to merge/unmerge around transfer**

In `actor.py`, modify `update_weights()` (around line 500):

```python
# Before line 500 (self.weight_updater.update_weights()):
lora_enabled = getattr(self.args, "lora_rank", 0) > 0
if lora_enabled:
    from slime.backends.megatron_utils.lora import merge_lora_weights, unmerge_lora_weights
    for model_chunk in self.model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        merge_lora_weights(unwrapped)

# Line 500: self.weight_updater.update_weights()

# After line 500:
if lora_enabled:
    for model_chunk in self.model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        unmerge_lora_weights(unwrapped)
```

- [ ] **Step 4: Commit**

```bash
git add slime/backends/megatron_utils/actor.py
git commit -m "feat(lora): integrate LoRA into actor training, ref forward, and weight update"
```

---

## Chunk 5: Checkpoint and MoE Expert LoRA

### Task 8: Adapter-Only Checkpoint Save/Load

**Files:**
- Modify: `slime/backends/megatron_utils/model.py:663-688`
- Modify: `slime/backends/megatron_utils/checkpoint.py`

- [ ] **Step 1: Add adapter save path in `model.py`**

In `model.py`, modify the `save()` function (lines 663-688). Add adapter-only save logic:

```python
def save(iteration, model, optimizer, opt_param_scheduler):
    args = get_args()

    if getattr(args, "lora_rank", 0) > 0 and getattr(args, "save_adapter_only", True):
        _save_lora_adapter(iteration, model, args)
        return

    # ... existing full checkpoint save logic ...
```

Add the helper function:

```python
def _save_lora_adapter(iteration: int, model: Sequence[DDP], args) -> None:
    """Save only LoRA adapter weights."""
    import json
    from pathlib import Path

    from megatron.core import parallel_state as mpu

    save_dir = Path(args.save) / f"lora_adapter_iter_{iteration:07d}"

    if mpu.get_data_parallel_rank() == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

        adapter_state = {}
        for model_chunk in model:
            unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
            for name, param in unwrapped.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.cpu()

        if mpu.get_tensor_model_parallel_rank() == 0:
            import torch
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
```

- [ ] **Step 2: Add adapter load in `checkpoint.py`**

In `checkpoint.py`, add adapter overlay loading. Add a function:

```python
def load_lora_adapter(model, adapter_path: str) -> None:
    """Load LoRA adapter weights on top of already-injected model."""
    import torch

    adapter_state = torch.load(Path(adapter_path) / "adapter_model.bin", map_location="cpu")
    for model_chunk in model:
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        missing, unexpected = [], []
        model_state = dict(unwrapped.named_parameters())
        for name, tensor in adapter_state.items():
            if name in model_state:
                model_state[name].data.copy_(tensor.to(model_state[name].device))
            else:
                missing.append(name)
        if missing:
            logger.warning(f"LoRA adapter keys not found in model: {missing}")
    logger.info(f"Loaded LoRA adapter from {adapter_path}")
```

- [ ] **Step 3: Wire adapter load in `actor.py` init**

In `actor.py`, after the `initialize_model_and_optimizer` call (line 91-93), add:

```python
if getattr(args, "adapter_load", None):
    from slime.backends.megatron_utils.checkpoint import load_lora_adapter
    load_lora_adapter(self.model, args.adapter_load)
```

- [ ] **Step 4: Commit**

```bash
git add slime/backends/megatron_utils/model.py slime/backends/megatron_utils/checkpoint.py slime/backends/megatron_utils/actor.py
git commit -m "feat(lora): add adapter-only checkpoint save/load"
```

---

### Task 9: MoE Expert LoRA (3D Batched)

**Files:**
- Modify: `slime/backends/megatron_utils/lora/layers.py`
- Modify: `slime/backends/megatron_utils/lora/injection.py`
- Modify: `slime/backends/megatron_utils/lora/merge.py`
- Test: `tests/test_lora_layers.py` (append expert tests)

- [ ] **Step 1: Write expert LoRA tests**

Append to `tests/test_lora_layers.py`:

```python
class MockGroupedMLP(nn.Module):
    """Mock for Megatron GroupedMLP with stacked 3D weights."""

    def __init__(self, num_local_experts, input_size, ffn_hidden):
        super().__init__()
        # weight1 = fc1 (gate+up fused): [num_experts, 2*ffn_hidden, input_size]
        self.weight1 = nn.Parameter(torch.randn(num_local_experts, 2 * ffn_hidden, input_size))
        # weight2 = fc2: [num_experts, input_size, ffn_hidden]
        self.weight2 = nn.Parameter(torch.randn(num_local_experts, input_size, ffn_hidden))
        self.num_local_experts = num_local_experts
        self.config = type("Config", (), {"hidden_size": input_size, "ffn_hidden_size": ffn_hidden})()

    def forward(self, x, tokens_per_expert):
        return x  # stub


def test_lora_expert_fc1_shapes():
    """Expert FC1 LoRA should have correct 3D parameter shapes."""
    from slime.backends.megatron_utils.lora.layers import LoRAGroupedExpertFC1

    mlp = MockGroupedMLP(num_local_experts=4, input_size=64, ffn_hidden=128)
    lora = LoRAGroupedExpertFC1(mlp, num_local_experts=4, rank=8, alpha=16, hidden_size=64, ffn_hidden_per_tp=128)

    assert lora.gate_lora_A.shape == (4, 8, 64)
    assert lora.gate_lora_B.shape == (4, 128, 8)
    assert lora.up_lora_A.shape == (4, 8, 64)
    assert lora.up_lora_B.shape == (4, 128, 8)


def test_lora_expert_fc2_shapes():
    from slime.backends.megatron_utils.lora.layers import LoRAGroupedExpertFC2

    mlp = MockGroupedMLP(num_local_experts=4, input_size=64, ffn_hidden=128)
    lora = LoRAGroupedExpertFC2(mlp, num_local_experts=4, rank=8, alpha=16, hidden_size=64, ffn_hidden_per_tp=128)

    assert lora.lora_A.shape == (4, 8, 128)
    assert lora.lora_B.shape == (4, 64, 8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_layers.py::test_lora_expert_fc1_shapes -v
```

- [ ] **Step 3: Implement expert LoRA layers**

Add to `slime/backends/megatron_utils/lora/layers.py`:

```python
class LoRAGroupedExpertFC1(nn.Module):
    """Per-expert LoRA on GroupedMLP weight1 (fc1, gate+up fused, 3D stacked).

    weight1: [num_local_experts, 2*ffn_hidden_per_tp, hidden_size]
    gate_lora_A/B and up_lora_A/B: 3D [num_local_experts, ...]
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_local_experts: int,
        rank: int,
        alpha: float,
        hidden_size: int,
        ffn_hidden_per_tp: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.num_local_experts = num_local_experts

        self.gate_lora_A = nn.Parameter(torch.empty(num_local_experts, rank, hidden_size))
        self.gate_lora_B = nn.Parameter(torch.zeros(num_local_experts, ffn_hidden_per_tp, rank))
        self.up_lora_A = nn.Parameter(torch.empty(num_local_experts, rank, hidden_size))
        self.up_lora_B = nn.Parameter(torch.zeros(num_local_experts, ffn_hidden_per_tp, rank))

        for A in [self.gate_lora_A, self.up_lora_A]:
            for e in range(num_local_experts):
                nn.init.kaiming_uniform_(A[e], a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()


class LoRAGroupedExpertFC2(nn.Module):
    """Per-expert LoRA on GroupedMLP weight2 (fc2, 3D stacked).

    weight2: [num_local_experts, hidden_size, ffn_hidden_per_tp]
    lora_A: [num_local_experts, rank, ffn_hidden_per_tp]  (TP-sharded input)
    lora_B: [num_local_experts, hidden_size, rank]         (replicated output)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_local_experts: int,
        rank: int,
        alpha: float,
        hidden_size: int,
        ffn_hidden_per_tp: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.num_local_experts = num_local_experts

        self.lora_A = nn.Parameter(torch.empty(num_local_experts, rank, ffn_hidden_per_tp))
        self.lora_B = nn.Parameter(torch.zeros(num_local_experts, hidden_size, rank))

        for e in range(num_local_experts):
            nn.init.kaiming_uniform_(self.lora_A[e], a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
```

- [ ] **Step 4: Add expert merge logic**

In `slime/backends/megatron_utils/lora/merge.py`, add expert merge support to `merge_lora_weights` and `unmerge_lora_weights`:

```python
# Add to imports
from slime.backends.megatron_utils.lora.layers import LoRAGroupedExpertFC1, LoRAGroupedExpertFC2

# In merge_lora_weights, add:
elif isinstance(module, LoRAGroupedExpertFC1):
    for e in range(module.num_local_experts):
        gate_delta = (module.gate_lora_B[e] @ module.gate_lora_A[e]) * module.scaling
        up_delta = (module.up_lora_B[e] @ module.up_lora_A[e]) * module.scaling
        import torch
        delta = torch.cat([gate_delta, up_delta], dim=0)
        module.base_layer.weight1.data[e] += delta
elif isinstance(module, LoRAGroupedExpertFC2):
    for e in range(module.num_local_experts):
        module.base_layer.weight2.data[e] += (module.lora_B[e] @ module.lora_A[e]) * module.scaling

# Mirror in unmerge_lora_weights with subtraction
```

Also update `_LORA_LAYER_TYPES` tuple to include expert types, and `disable_lora`/`enable_lora` to handle them.

- [ ] **Step 5: Add expert injection in `injection.py`**

Add expert injection support in `inject_lora_adapters`:

```python
# After shared MLP injection, handle MoE experts:
if "expert" in targets and hasattr(layer.mlp, "experts"):
    experts_module = layer.mlp.experts
    if hasattr(experts_module, "weight1"):  # GroupedMLP
        num_local_experts = experts_module.num_local_experts
        ffn_hidden_per_tp = experts_module.config.ffn_hidden_size // expt_tp_size
        experts_module.lora_fc1 = LoRAGroupedExpertFC1(
            experts_module, num_local_experts, config.rank, config.alpha,
            hidden_size, ffn_hidden_per_tp, config.dropout,
        )
        experts_module.lora_fc2 = LoRAGroupedExpertFC2(
            experts_module, num_local_experts, config.rank, config.alpha,
            hidden_size, ffn_hidden_per_tp, config.dropout,
        )
        count += 2
```

Note: Expert LoRA injection is more complex because `GroupedMLP.forward()` directly uses `weight1`/`weight2` tensors in grouped GEMM. The merge approach handles this by modifying `weight1`/`weight2` directly before transfer. During training, expert LoRA forward must hook into the GroupedMLP computation, which requires understanding the `grouped_gemm` call pattern. This may need adjustment based on the actual `GroupedMLP.forward()` implementation in the version of Megatron being used.

- [ ] **Step 6: Run all tests**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_layers.py tests/test_lora_merge.py tests/test_lora_injection.py tests/test_lora_config.py -v
```
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add slime/backends/megatron_utils/lora/ tests/
git commit -m "feat(lora): add MoE expert LoRA (3D batched) layers and merge"
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
Expected: `__init__.py`, `config.py`, `layers.py`, `injection.py`, `merge.py`

- [ ] **Step 2: Verify all tests pass**

```bash
cd /Users/jd/Documents/workspace/slime && python -m pytest tests/test_lora_*.py -v
```

- [ ] **Step 3: Review git log**

```bash
git log --oneline dev..HEAD
```
Expected: ~8-9 commits, all prefixed with `feat(lora):` or `fix(lora):`
