"""Tests for LoRA wrapper layers."""

import torch
import torch.nn as nn


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

    x = torch.randn(2, 10, 64)
    out, bias = lora(x)
    assert out.shape == (2, 10, 32)


def test_lora_column_parallel_zero_init():
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)

    x = torch.randn(2, 10, 64)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    torch.testing.assert_close(lora_out, base_out)


def test_lora_column_parallel_nonzero_after_update():
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
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=32)
    lora = LoRAColumnParallelLinear(base, rank=8, alpha=16)

    lora_params = {n for n, p in lora.named_parameters() if "lora_" in n}
    assert "lora_A" in lora_params
    assert "lora_B" in lora_params


def test_lora_scaling():
    from slime.backends.megatron_utils.lora.layers import LoRAColumnParallelLinear

    base = MockColumnParallelLinear(input_size=4, output_size_per_partition=4)
    base.weight.data.zero_()
    lora = LoRAColumnParallelLinear(base, rank=2, alpha=4)

    lora.lora_A.data = torch.eye(2, 4)
    lora.lora_B.data = torch.eye(4, 2)

    x = torch.ones(1, 1, 4)
    out, _ = lora(x)
    expected = torch.tensor([[[2.0, 2.0, 0.0, 0.0]]])
    torch.testing.assert_close(out, expected)


def test_lora_fused_qkv_forward_shape():
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    qkv_out = 96  # (8 + 2*2) * 8
    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=qkv_out)
    lora = LoRAFusedQKV(
        base,
        rank=8,
        alpha=16,
        num_q_heads_per_tp=8,
        num_kv_heads_per_tp=2,
        head_dim=8,
    )
    x = torch.randn(2, 10, 64)
    out, _ = lora(x)
    assert out.shape == (2, 10, 96)


def test_lora_fused_qkv_zero_init():
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=96)
    lora = LoRAFusedQKV(
        base,
        rank=8,
        alpha=16,
        num_q_heads_per_tp=8,
        num_kv_heads_per_tp=2,
        head_dim=8,
    )
    x = torch.randn(2, 10, 64)
    base_out, _ = base(x)
    lora_out, _ = lora(x)
    torch.testing.assert_close(lora_out, base_out)


def test_lora_fused_qkv_kv_heads_equal_one():
    """[I1] Test with num_kv_heads_per_tp=1 (num_kv_heads < tp_size with replication)."""
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV

    qkv_out = (4 + 2 * 1) * 8  # = 48
    base = MockColumnParallelLinear(input_size=64, output_size_per_partition=qkv_out)
    lora = LoRAFusedQKV(
        base,
        rank=4,
        alpha=8,
        num_q_heads_per_tp=4,
        num_kv_heads_per_tp=1,
        head_dim=8,
    )
    x = torch.randn(1, 5, 64)
    out, _ = lora(x)
    assert out.shape == (1, 5, 48)


def test_lora_fused_fc1_forward_shape():
    from slime.backends.megatron_utils.lora.layers import LoRAFusedFC1

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
