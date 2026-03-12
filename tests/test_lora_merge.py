"""Tests for LoRA merge/unmerge and disable/enable operations."""

import torch
import torch.nn as nn

from tests.test_lora_layers import MockColumnParallelLinear


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
    assert not torch.allclose(base.weight.data, original_weight)

    unmerge_lora_weights(model)
    torch.testing.assert_close(base.weight.data, original_weight)


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

    merged_out = nn.functional.linear(x, base.weight)
    torch.testing.assert_close(merged_out, lora_out, atol=1e-5, rtol=1e-5)


def test_merge_fused_qkv_roundtrip():
    """[C3] Fused QKV merge/unmerge roundtrip with correct weight dimension handling."""
    from slime.backends.megatron_utils.lora.layers import LoRAFusedQKV
    from slime.backends.megatron_utils.lora.merge import merge_lora_weights, unmerge_lora_weights

    qkv_out = (4 + 2 * 2) * 8  # 64
    base = MockColumnParallelLinear(input_size=32, output_size_per_partition=qkv_out)
    lora = LoRAFusedQKV(base, rank=4, alpha=8, num_q_heads_per_tp=4, num_kv_heads_per_tp=2, head_dim=8)
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
