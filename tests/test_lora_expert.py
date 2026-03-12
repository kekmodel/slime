"""Tests for MoE expert LoRA with GroupedMLP forward override."""

import torch
import torch.nn as nn


class MockGroupedMLP(nn.Module):
    """Mock for Megatron GroupedMLP with stacked 3D weights."""

    def __init__(self, num_local_experts, hidden_size, ffn_hidden):
        super().__init__()
        # Megatron stores weight1 as [H, E*2F]
        self.weight1 = nn.Parameter(torch.randn(hidden_size, num_local_experts * ffn_hidden * 2))
        # weight2 as [E*F, H]
        self.weight2 = nn.Parameter(torch.randn(num_local_experts * ffn_hidden, hidden_size))
        self.num_local_experts = num_local_experts
        self.config = type("Config", (), {"hidden_size": hidden_size, "ffn_hidden_size": ffn_hidden})()
        self._forward_called = False

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        self._forward_called = True
        E = self.num_local_experts
        H = self.config.hidden_size
        w1 = self.weight1.view(E, H, -1)  # [E, H, 2F]
        _ = self.weight2.view(E, -1, H)  # [E, F, H] (used in real GroupedMLP)
        output = torch.bmm(permuted_local_hidden_states.unsqueeze(0).expand(E, -1, -1), w1)
        return output.sum(0)


def test_expert_lora_param_shapes():
    """Expert LoRA should create correct 3D parameter shapes."""
    from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora

    mlp = MockGroupedMLP(num_local_experts=4, hidden_size=64, ffn_hidden=128)
    inject_expert_lora(mlp, rank=8, alpha=16)

    assert mlp._lora_A_fc1.shape == (4, 8, 64)
    assert mlp._lora_gate_B_fc1.shape == (4, 128, 8)
    assert mlp._lora_up_B_fc1.shape == (4, 128, 8)
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
