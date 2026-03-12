"""Tests for LoRA injection into model."""


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
        rank=8,
        alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    inject_lora_adapters(
        [model],
        config,  # [I6] Accepts list of model chunks
        num_q_heads_per_tp=8,
        num_kv_heads_per_tp=2,
        head_dim=8,
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
        [chunk0, chunk1],
        config,
        num_q_heads_per_tp=8,
        num_kv_heads_per_tp=2,
        head_dim=8,
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
