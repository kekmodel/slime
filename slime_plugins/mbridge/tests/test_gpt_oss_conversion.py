# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Numerical precision tests for GptOssBridge weight conversion.

Tests verify that:
1. HF → MCore → HF round-trip preserves weights within tolerance
2. Forward pass produces same outputs with converted weights
3. Backward pass produces same gradients with converted weights
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch


# Tolerance levels for numerical comparison
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-6  # Absolute tolerance
GRAD_RTOL = 1e-4  # Gradient relative tolerance (slightly looser)
GRAD_ATOL = 1e-5  # Gradient absolute tolerance


@dataclass
class MockHFConfig:
    """Mock HuggingFace config for GPT-OSS."""
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000.0
    attention_dropout: float = 0.0
    router_aux_loss_coef: float = 0.9
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {
                "type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
            }


@dataclass
class MockMCoreConfig:
    """Mock Megatron-Core config."""
    num_moe_experts: int = 128
    moe_grouped_gemm: bool = True
    qk_layernorm: bool = False


class TestQKVConversion:
    """Test QKV weight/bias conversion for GQA."""

    @pytest.fixture
    def bridge(self):
        """Create a GptOssBridge instance with mocked dependencies."""
        with patch('megatron.core.__version__', '0.14.0'):
            from slime_plugins.mbridge.gpt_oss import GptOssBridge

            mock_mpu = MagicMock()
            mock_mpu.tp_size = 1
            mock_mpu.pp_size = 1
            mock_mpu.ep_size = 1
            mock_mpu.etp_size = 1
            mock_mpu.vpp_size = None
            mock_mpu.cp_size = 1

            with patch.object(GptOssBridge, '_build_config', return_value=MockMCoreConfig()):
                bridge = GptOssBridge.__new__(GptOssBridge)
                bridge.hf_config = MockHFConfig()
                bridge.config = MockMCoreConfig()
                bridge.export_weights_buff = {}
                return bridge

    def test_qkv_weight_roundtrip(self, bridge):
        """Test QKV weight conversion preserves values."""
        hf_config = bridge.hf_config
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim

        # Create HF weights
        q_weight = torch.randn(num_heads * head_dim, hidden_size)
        k_weight = torch.randn(num_kv_heads * head_dim, hidden_size)
        v_weight = torch.randn(num_kv_heads * head_dim, hidden_size)
        hf_weights = [q_weight, k_weight, v_weight]

        # HF → MCore
        mcore_weight = bridge._merge_qkv_weights(hf_weights)

        # MCore → HF
        recovered_weights = bridge._split_qkv_weights(mcore_weight)

        # Verify round-trip
        for orig, recovered, name in zip(hf_weights, recovered_weights, ['Q', 'K', 'V']):
            assert torch.allclose(orig, recovered, rtol=RTOL, atol=ATOL), \
                f"{name} weight mismatch: max diff = {(orig - recovered).abs().max().item()}"

        print(f"✓ QKV weight round-trip: max diff Q={_max_diff(q_weight, recovered_weights[0]):.2e}, "
              f"K={_max_diff(k_weight, recovered_weights[1]):.2e}, "
              f"V={_max_diff(v_weight, recovered_weights[2]):.2e}")

    def test_qkv_bias_roundtrip(self, bridge):
        """Test QKV bias conversion preserves values."""
        hf_config = bridge.hf_config
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim

        # Create HF biases
        q_bias = torch.randn(num_heads * head_dim)
        k_bias = torch.randn(num_kv_heads * head_dim)
        v_bias = torch.randn(num_kv_heads * head_dim)
        hf_biases = [q_bias, k_bias, v_bias]

        # HF → MCore
        mcore_bias = bridge._merge_qkv_bias(hf_biases)

        # MCore → HF
        recovered_biases = bridge._split_qkv_bias(mcore_bias)

        # Verify round-trip
        for orig, recovered, name in zip(hf_biases, recovered_biases, ['Q', 'K', 'V']):
            assert torch.allclose(orig, recovered, rtol=RTOL, atol=ATOL), \
                f"{name} bias mismatch: max diff = {(orig - recovered).abs().max().item()}"

        print(f"✓ QKV bias round-trip: max diff Q={_max_diff(q_bias, recovered_biases[0]):.2e}, "
              f"K={_max_diff(k_bias, recovered_biases[1]):.2e}, "
              f"V={_max_diff(v_bias, recovered_biases[2]):.2e}")


class TestFusedExpertConversion:
    """Test fused expert weight conversion."""

    @pytest.fixture
    def bridge(self):
        """Create a GptOssBridge instance with mocked dependencies."""
        with patch('megatron.core.__version__', '0.14.0'):
            from slime_plugins.mbridge.gpt_oss import GptOssBridge

            mock_mpu = MagicMock()
            mock_mpu.tp_size = 1
            mock_mpu.pp_size = 1
            mock_mpu.ep_size = 1
            mock_mpu.etp_size = 1
            mock_mpu.vpp_size = None
            mock_mpu.cp_size = 1

            with patch.object(GptOssBridge, '_build_config', return_value=MockMCoreConfig()):
                bridge = GptOssBridge.__new__(GptOssBridge)
                bridge.hf_config = MockHFConfig()
                bridge.config = MockMCoreConfig()
                bridge.export_weights_buff = {}
                return bridge

    def test_expert_weight_slice(self, bridge):
        """Test slicing individual expert from fused tensor."""
        num_experts = 128
        hidden_size = bridge.hf_config.hidden_size
        intermediate_size = bridge.hf_config.intermediate_size

        # Create fused HF weight: (num_experts, hidden_size, 2*intermediate_size)
        fused_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size)

        # Test slicing each expert
        for expert_id in [0, 1, 63, 127]:  # Test corners and middle
            mcore_name = f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
            sliced = bridge._weight_to_mcore_format(mcore_name, [fused_weight])

            expected = fused_weight[expert_id]
            assert torch.allclose(sliced, expected, rtol=RTOL, atol=ATOL), \
                f"Expert {expert_id} slice mismatch"

        print(f"✓ Expert weight slicing: all {num_experts} experts correctly extracted")

    def test_expert_weight_accumulation(self, bridge):
        """Test accumulating experts back into fused tensor."""
        num_experts = bridge.config.num_moe_experts
        hidden_size = bridge.hf_config.hidden_size
        intermediate_size = bridge.hf_config.intermediate_size

        # Create individual expert weights
        expert_weights = [
            torch.randn(hidden_size, 2 * intermediate_size)
            for _ in range(num_experts)
        ]

        # Accumulate experts one by one
        bridge.export_weights_buff = {}
        for expert_id in range(num_experts):
            mcore_name = f"decoder.layers.5.mlp.experts.linear_fc1.weight{expert_id}"
            hf_names, hf_weights = bridge._accumulate_expert_weight(
                mcore_name, expert_weights[expert_id], "5"
            )

            # Should return empty until all experts are collected
            if expert_id < num_experts - 1:
                assert len(hf_names) == 0
                assert len(hf_weights) == 0
            else:
                # Last expert triggers fusion
                assert len(hf_names) == 1
                assert len(hf_weights) == 1
                assert hf_names[0] == "model.layers.5.mlp.experts.gate_up_proj"

                # Verify fused tensor matches original experts
                fused = hf_weights[0]
                for i, orig_weight in enumerate(expert_weights):
                    assert torch.allclose(fused[i], orig_weight, rtol=RTOL, atol=ATOL), \
                        f"Expert {i} mismatch in fused tensor"

        print(f"✓ Expert weight accumulation: {num_experts} experts correctly fused")

    def test_expert_roundtrip(self, bridge):
        """Test full round-trip: fused HF → per-expert MCore → fused HF."""
        num_experts = bridge.config.num_moe_experts
        hidden_size = bridge.hf_config.hidden_size
        intermediate_size = bridge.hf_config.intermediate_size

        # Original fused HF weight
        original_fused = torch.randn(num_experts, hidden_size, 2 * intermediate_size)

        # Step 1: HF → MCore (slice into individual experts)
        mcore_weights = []
        for expert_id in range(num_experts):
            mcore_name = f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
            sliced = bridge._weight_to_mcore_format(mcore_name, [original_fused])
            mcore_weights.append(sliced)

        # Step 2: MCore → HF (accumulate back into fused)
        bridge.export_weights_buff = {}
        recovered_fused = None
        for expert_id, mcore_weight in enumerate(mcore_weights):
            mcore_name = f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
            hf_names, hf_weights = bridge._accumulate_expert_weight(
                mcore_name, mcore_weight, "0"
            )
            if hf_weights:
                recovered_fused = hf_weights[0]

        # Verify round-trip
        assert recovered_fused is not None, "Failed to recover fused tensor"
        assert torch.allclose(original_fused, recovered_fused, rtol=RTOL, atol=ATOL), \
            f"Fused expert round-trip failed: max diff = {_max_diff(original_fused, recovered_fused):.2e}"

        print(f"✓ Expert round-trip: max diff = {_max_diff(original_fused, recovered_fused):.2e}")


class TestForwardPassEquivalence:
    """Test that converted weights produce same forward pass outputs."""

    def test_qkv_projection_forward(self):
        """Test QKV projection produces same output with original and converted weights."""
        # Config
        batch_size = 2
        seq_len = 128
        hidden_size = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = 32

        # Create mock config
        mock_config = MockHFConfig()
        mock_config.hidden_size = hidden_size
        mock_config.num_attention_heads = num_heads
        mock_config.num_key_value_heads = num_kv_heads
        mock_config.head_dim = head_dim

        # Create HF-style separate projections
        q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)

        # Get weights
        q_weight, k_weight, v_weight = q_proj.weight.data, k_proj.weight.data, v_proj.weight.data
        q_bias, k_bias, v_bias = q_proj.bias.data, k_proj.bias.data, v_proj.bias.data

        # Create mock bridge for conversion
        class MockBridge:
            def __init__(self):
                self.hf_config = mock_config

        bridge = MockBridge()

        # Import conversion functions
        from slime_plugins.mbridge.gpt_oss import GptOssBridge
        bridge._merge_qkv_weights = GptOssBridge._merge_qkv_weights.__get__(bridge)
        bridge._merge_qkv_bias = GptOssBridge._merge_qkv_bias.__get__(bridge)

        # Convert to MCore format
        mcore_qkv_weight = bridge._merge_qkv_weights([q_weight, k_weight, v_weight])
        mcore_qkv_bias = bridge._merge_qkv_bias([q_bias, k_bias, v_bias])

        # Create MCore-style fused projection
        qkv_proj = nn.Linear(hidden_size, mcore_qkv_weight.shape[0], bias=True)
        qkv_proj.weight.data = mcore_qkv_weight
        qkv_proj.bias.data = mcore_qkv_bias

        # Input
        x = torch.randn(batch_size, seq_len, hidden_size)

        # HF forward
        q_hf = q_proj(x)
        k_hf = k_proj(x)
        v_hf = v_proj(x)

        # MCore forward
        qkv_mcore = qkv_proj(x)

        # Split MCore output to match HF format
        # MCore interleaves: [Q_group0, K0, V0, Q_group1, K1, V1, ...]
        num_groups = num_heads // num_kv_heads
        qkv_reshaped = qkv_mcore.view(batch_size, seq_len, num_kv_heads, num_groups + 2, head_dim)
        q_mcore = qkv_reshaped[:, :, :, :num_groups, :].reshape(batch_size, seq_len, -1)
        k_mcore = qkv_reshaped[:, :, :, num_groups, :].reshape(batch_size, seq_len, -1)
        v_mcore = qkv_reshaped[:, :, :, num_groups + 1, :].reshape(batch_size, seq_len, -1)

        # Compare
        assert torch.allclose(q_hf, q_mcore, rtol=RTOL, atol=ATOL), \
            f"Q output mismatch: max diff = {_max_diff(q_hf, q_mcore):.2e}"
        assert torch.allclose(k_hf, k_mcore, rtol=RTOL, atol=ATOL), \
            f"K output mismatch: max diff = {_max_diff(k_hf, k_mcore):.2e}"
        assert torch.allclose(v_hf, v_mcore, rtol=RTOL, atol=ATOL), \
            f"V output mismatch: max diff = {_max_diff(v_hf, v_mcore):.2e}"

        print(f"✓ QKV forward pass: Q diff={_max_diff(q_hf, q_mcore):.2e}, "
              f"K diff={_max_diff(k_hf, k_mcore):.2e}, V diff={_max_diff(v_hf, v_mcore):.2e}")


class TestGradientEquivalence:
    """Test that gradients are preserved through weight conversion."""

    def test_qkv_gradient_equivalence(self):
        """Test gradients match for QKV with original and converted weights."""
        # Config
        batch_size = 2
        seq_len = 64
        hidden_size = 128
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32

        # Create mock config
        mock_config = MockHFConfig()
        mock_config.hidden_size = hidden_size
        mock_config.num_attention_heads = num_heads
        mock_config.num_key_value_heads = num_kv_heads
        mock_config.head_dim = head_dim

        # Create HF-style projections with same initialization
        torch.manual_seed(42)
        q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)

        # Get weights
        q_weight = q_proj.weight.data.clone()
        k_weight = k_proj.weight.data.clone()
        v_weight = v_proj.weight.data.clone()
        q_bias = q_proj.bias.data.clone()
        k_bias = k_proj.bias.data.clone()
        v_bias = v_proj.bias.data.clone()

        # Create mock bridge
        class MockBridge:
            def __init__(self):
                self.hf_config = mock_config

        bridge = MockBridge()
        from slime_plugins.mbridge.gpt_oss import GptOssBridge
        bridge._merge_qkv_weights = GptOssBridge._merge_qkv_weights.__get__(bridge)
        bridge._merge_qkv_bias = GptOssBridge._merge_qkv_bias.__get__(bridge)
        bridge._split_qkv_weights = GptOssBridge._split_qkv_weights.__get__(bridge)
        bridge._split_qkv_bias = GptOssBridge._split_qkv_bias.__get__(bridge)

        # Convert to MCore format
        mcore_qkv_weight = bridge._merge_qkv_weights([q_weight, k_weight, v_weight])
        mcore_qkv_bias = bridge._merge_qkv_bias([q_bias, k_bias, v_bias])

        # Create MCore-style fused projection
        qkv_proj = nn.Linear(hidden_size, mcore_qkv_weight.shape[0], bias=True)
        qkv_proj.weight.data = mcore_qkv_weight.clone()
        qkv_proj.bias.data = mcore_qkv_bias.clone()

        # Same input for both
        torch.manual_seed(123)
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)

        # HF forward + backward
        q_hf = q_proj(x)
        k_hf = k_proj(x)
        v_hf = v_proj(x)
        loss_hf = (q_hf.sum() + k_hf.sum() + v_hf.sum())
        loss_hf.backward()

        # MCore forward
        qkv_mcore = qkv_proj(x_clone)
        num_groups = num_heads // num_kv_heads
        qkv_reshaped = qkv_mcore.view(batch_size, seq_len, num_kv_heads, num_groups + 2, head_dim)
        q_mcore = qkv_reshaped[:, :, :, :num_groups, :].reshape(batch_size, seq_len, -1)
        k_mcore = qkv_reshaped[:, :, :, num_groups, :].reshape(batch_size, seq_len, -1)
        v_mcore = qkv_reshaped[:, :, :, num_groups + 1, :].reshape(batch_size, seq_len, -1)
        loss_mcore = (q_mcore.sum() + k_mcore.sum() + v_mcore.sum())
        loss_mcore.backward()

        # Compare input gradients
        assert torch.allclose(x.grad, x_clone.grad, rtol=GRAD_RTOL, atol=GRAD_ATOL), \
            f"Input gradient mismatch: max diff = {_max_diff(x.grad, x_clone.grad):.2e}"

        # Split MCore gradients and compare with HF gradients
        mcore_grad_qkv = qkv_proj.weight.grad
        recovered_grads = bridge._split_qkv_weights(mcore_grad_qkv)

        assert torch.allclose(q_proj.weight.grad, recovered_grads[0], rtol=GRAD_RTOL, atol=GRAD_ATOL), \
            f"Q weight gradient mismatch: max diff = {_max_diff(q_proj.weight.grad, recovered_grads[0]):.2e}"
        assert torch.allclose(k_proj.weight.grad, recovered_grads[1], rtol=GRAD_RTOL, atol=GRAD_ATOL), \
            f"K weight gradient mismatch: max diff = {_max_diff(k_proj.weight.grad, recovered_grads[1]):.2e}"
        assert torch.allclose(v_proj.weight.grad, recovered_grads[2], rtol=GRAD_RTOL, atol=GRAD_ATOL), \
            f"V weight gradient mismatch: max diff = {_max_diff(v_proj.weight.grad, recovered_grads[2]):.2e}"

        print(f"✓ QKV gradient equivalence: input grad diff={_max_diff(x.grad, x_clone.grad):.2e}, "
              f"weight grad diff Q={_max_diff(q_proj.weight.grad, recovered_grads[0]):.2e}")


class TestPrecisionLevels:
    """Test precision at different data types."""

    @pytest.mark.parametrize("dtype,rtol,atol", [
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 1e-3, 1e-4),
        (torch.bfloat16, 1e-2, 1e-3),
    ])
    def test_qkv_conversion_precision(self, dtype, rtol, atol):
        """Test QKV conversion at different precision levels."""
        hidden_size = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = 32

        mock_config = MockHFConfig()
        mock_config.hidden_size = hidden_size
        mock_config.num_attention_heads = num_heads
        mock_config.num_key_value_heads = num_kv_heads
        mock_config.head_dim = head_dim

        class MockBridge:
            def __init__(self):
                self.hf_config = mock_config

        bridge = MockBridge()
        from slime_plugins.mbridge.gpt_oss import GptOssBridge
        bridge._merge_qkv_weights = GptOssBridge._merge_qkv_weights.__get__(bridge)
        bridge._split_qkv_weights = GptOssBridge._split_qkv_weights.__get__(bridge)

        # Create weights in target dtype
        q_weight = torch.randn(num_heads * head_dim, hidden_size, dtype=dtype)
        k_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype)
        v_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype)

        # Round-trip
        mcore_weight = bridge._merge_qkv_weights([q_weight, k_weight, v_weight])
        recovered = bridge._split_qkv_weights(mcore_weight)

        # Verify with dtype-appropriate tolerance
        for orig, rec, name in zip([q_weight, k_weight, v_weight], recovered, ['Q', 'K', 'V']):
            max_diff = _max_diff(orig, rec)
            assert torch.allclose(orig, rec, rtol=rtol, atol=atol), \
                f"{name} {dtype} conversion failed: max diff = {max_diff:.2e}"

        print(f"✓ {dtype} precision: max diff = {max(map(lambda x: _max_diff(x[0], x[1]), zip([q_weight, k_weight, v_weight], recovered))):.2e}")


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculate maximum absolute difference between two tensors."""
    return (a - b).abs().max().item()


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("GptOssBridge Weight Conversion Precision Tests")
    print("=" * 60)
    print()

    # QKV tests
    print("--- QKV Conversion Tests ---")
    try:
        test_qkv = TestQKVConversion()
        bridge = test_qkv.bridge.__wrapped__(test_qkv)  # Get fixture
        test_qkv.test_qkv_weight_roundtrip(bridge)
        test_qkv.test_qkv_bias_roundtrip(bridge)
    except Exception as e:
        print(f"✗ QKV tests failed: {e}")

    print()

    # Expert tests
    print("--- Fused Expert Conversion Tests ---")
    try:
        test_expert = TestFusedExpertConversion()
        bridge = test_expert.bridge.__wrapped__(test_expert)
        test_expert.test_expert_weight_slice(bridge)
        test_expert.test_expert_weight_accumulation(bridge)
        test_expert.test_expert_roundtrip(bridge)
    except Exception as e:
        print(f"✗ Expert tests failed: {e}")

    print()

    # Forward pass tests
    print("--- Forward Pass Equivalence Tests ---")
    try:
        test_forward = TestForwardPassEquivalence()
        test_forward.test_qkv_projection_forward()
    except Exception as e:
        print(f"✗ Forward pass tests failed: {e}")

    print()

    # Gradient tests
    print("--- Gradient Equivalence Tests ---")
    try:
        test_grad = TestGradientEquivalence()
        test_grad.test_qkv_gradient_equivalence()
    except Exception as e:
        print(f"✗ Gradient tests failed: {e}")

    print()

    # Precision tests
    print("--- Multi-Precision Tests ---")
    try:
        test_precision = TestPrecisionLevels()
        for dtype, rtol, atol in [(torch.float32, 1e-5, 1e-6),
                                   (torch.float16, 1e-3, 1e-4),
                                   (torch.bfloat16, 1e-2, 1e-3)]:
            test_precision.test_qkv_conversion_precision(dtype, rtol, atol)
    except Exception as e:
        print(f"✗ Precision tests failed: {e}")

    print()
    print("=" * 60)
    print("Test Summary Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Run with pytest for full test suite
    # pytest slime_plugins/mbridge/tests/test_gpt_oss_conversion.py -v

    # Or run standalone
    run_all_tests()
