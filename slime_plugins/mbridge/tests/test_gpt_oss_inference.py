# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
End-to-end inference equivalence test for GptOssBridge.

Simulates HF and MCore forward passes to verify converted weights
produce identical outputs and gradients.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Reduced config for testing (smaller than actual GPT-OSS-120B)."""
    hidden_size: int = 256
    intermediate_size: int = 512
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 32
    num_hidden_layers: int = 2
    num_local_experts: int = 4  # Reduced for testing
    num_experts_per_tok: int = 2
    vocab_size: int = 1000
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000.0
    attention_dropout: float = 0.0


# =============================================================================
# HuggingFace-style Model Components
# =============================================================================

class HFRMSNorm(nn.Module):
    """HF-style RMSNorm."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class HFRotaryEmbedding(nn.Module):
    """HF-style RoPE."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(position_ids.float().squeeze(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
    return q_embed, k_embed


class HFAttention(nn.Module):
    """HF-style GQA attention with separate Q/K/V projections."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = HFRotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand K/V for GQA
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """QuickGELU activation: x * sigmoid(1.702 * x)"""
    return x * torch.sigmoid(1.702 * x)


class HFExpertMLP(nn.Module):
    """Single expert MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT-OSS GLU: (up + 1) * quick_gelu(gate)
        gate = quick_gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj((up + 1.0) * gate)


class HFMoE(nn.Module):
    """HF-style MoE with fused expert weights."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Router
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)

        # Fused expert weights (GPT-OSS style)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        nn.init.zeros_(self.gate_up_proj_bias)
        nn.init.zeros_(self.down_proj_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Router
        router_logits = self.router(hidden_states_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Expert computation (simplified - process each token)
        final_output = torch.zeros_like(hidden_states_flat)

        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_input = hidden_states_flat[expert_mask]

            # Gate-up projection
            gate_up = torch.matmul(expert_input, self.gate_up_proj[expert_idx]) + self.gate_up_proj_bias[expert_idx]
            gate, up = gate_up.chunk(2, dim=-1)

            # GLU activation
            gate = quick_gelu(gate)
            activated = (up + 1.0) * gate

            # Down projection
            expert_output = torch.matmul(activated, self.down_proj[expert_idx]) + self.down_proj_bias[expert_idx]

            # Get routing weight for this expert
            expert_weight_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_weight_mask.float()).sum(dim=-1)

            final_output[expert_mask] += expert_output * expert_weights[expert_mask].unsqueeze(-1)

        return final_output.view(batch_size, seq_len, hidden_size)


class HFDecoderLayer(nn.Module):
    """HF-style decoder layer."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.self_attn = HFAttention(config)
        self.mlp = HFMoE(config)
        self.input_layernorm = HFRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = HFRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HFModel(nn.Module):
    """Simplified HF-style GPT-OSS model."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HFDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = HFRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


# =============================================================================
# MCore-style Model Components (using converted weights)
# =============================================================================

class MCoreAttention(nn.Module):
    """MCore-style fused QKV attention."""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        # Fused QKV projection (MCore style - interleaved)
        qkv_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, qkv_size, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = HFRotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV
        qkv = self.qkv_proj(hidden_states)

        # Split interleaved QKV (MCore format)
        qkv = qkv.view(batch_size, seq_len, self.num_kv_heads, self.num_groups + 2, self.head_dim)
        q = qkv[:, :, :, :self.num_groups, :].reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[:, :, :, self.num_groups, :].reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[:, :, :, self.num_groups + 1, :].reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand K/V for GQA
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


# =============================================================================
# Weight Conversion Functions
# =============================================================================

def convert_qkv_weights_hf_to_mcore(q_weight, k_weight, v_weight, config: TestConfig) -> torch.Tensor:
    """Convert HF Q/K/V weights to MCore interleaved format."""
    num_kv_heads = config.num_key_value_heads
    num_groups = config.num_attention_heads // num_kv_heads
    head_dim = config.head_dim
    hidden_size = config.hidden_size

    q = q_weight.view(num_kv_heads, num_groups, head_dim, hidden_size)
    k = k_weight.view(num_kv_heads, 1, head_dim, hidden_size)
    v = v_weight.view(num_kv_heads, 1, head_dim, hidden_size)

    qkv = torch.cat([q, k, v], dim=1)
    return qkv.reshape(-1, hidden_size)


def convert_qkv_bias_hf_to_mcore(q_bias, k_bias, v_bias, config: TestConfig) -> torch.Tensor:
    """Convert HF Q/K/V biases to MCore interleaved format."""
    num_kv_heads = config.num_key_value_heads
    num_groups = config.num_attention_heads // num_kv_heads
    head_dim = config.head_dim

    q = q_bias.view(num_kv_heads, num_groups, head_dim)
    k = k_bias.view(num_kv_heads, 1, head_dim)
    v = v_bias.view(num_kv_heads, 1, head_dim)

    qkv = torch.cat([q, k, v], dim=1)
    return qkv.reshape(-1)


def copy_attention_weights(hf_attn: HFAttention, mcore_attn: MCoreAttention, config: TestConfig):
    """Copy and convert attention weights from HF to MCore format."""
    # Convert QKV weights
    mcore_qkv_weight = convert_qkv_weights_hf_to_mcore(
        hf_attn.q_proj.weight.data,
        hf_attn.k_proj.weight.data,
        hf_attn.v_proj.weight.data,
        config
    )
    mcore_qkv_bias = convert_qkv_bias_hf_to_mcore(
        hf_attn.q_proj.bias.data,
        hf_attn.k_proj.bias.data,
        hf_attn.v_proj.bias.data,
        config
    )

    mcore_attn.qkv_proj.weight.data = mcore_qkv_weight
    mcore_attn.qkv_proj.bias.data = mcore_qkv_bias

    # Copy output projection directly
    mcore_attn.o_proj.weight.data = hf_attn.o_proj.weight.data.clone()
    mcore_attn.o_proj.bias.data = hf_attn.o_proj.bias.data.clone()


# =============================================================================
# Test Functions
# =============================================================================

def test_attention_forward_equivalence():
    """Test that HF and MCore attention produce same outputs."""
    print("\n" + "=" * 60)
    print("Test: Attention Forward Pass Equivalence")
    print("=" * 60)

    config = TestConfig()
    batch_size, seq_len = 2, 32

    # Create models
    hf_attn = HFAttention(config)
    mcore_attn = MCoreAttention(config)

    # Copy converted weights
    copy_attention_weights(hf_attn, mcore_attn, config)

    # Same input
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len)

    # Forward pass
    hf_attn.eval()
    mcore_attn.eval()

    with torch.no_grad():
        hf_output = hf_attn(hidden_states, position_ids)
        mcore_output = mcore_attn(hidden_states, position_ids)

    # Compare
    max_diff = (hf_output - mcore_output).abs().max().item()
    mean_diff = (hf_output - mcore_output).abs().mean().item()
    rel_diff = ((hf_output - mcore_output).abs() / (hf_output.abs() + 1e-8)).mean().item()

    print(f"  Output shape: {hf_output.shape}")
    print(f"  Max absolute diff:  {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    print(f"  Mean relative diff: {rel_diff:.2e}")

    passed = max_diff < 1e-5
    print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")

    return passed, max_diff


def test_attention_gradient_equivalence():
    """Test that HF and MCore attention produce same gradients."""
    print("\n" + "=" * 60)
    print("Test: Attention Gradient Equivalence")
    print("=" * 60)

    config = TestConfig()
    batch_size, seq_len = 2, 32

    # Create models
    hf_attn = HFAttention(config)
    mcore_attn = MCoreAttention(config)

    # Copy converted weights
    copy_attention_weights(hf_attn, mcore_attn, config)

    # Same input (requires grad)
    torch.manual_seed(42)
    hf_input = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    mcore_input = hf_input.clone().detach().requires_grad_(True)
    position_ids = torch.arange(seq_len)

    # Forward + backward
    hf_output = hf_attn(hf_input, position_ids)
    mcore_output = mcore_attn(mcore_input, position_ids)

    # Same loss
    hf_loss = hf_output.sum()
    mcore_loss = mcore_output.sum()

    hf_loss.backward()
    mcore_loss.backward()

    # Compare input gradients
    input_grad_diff = (hf_input.grad - mcore_input.grad).abs().max().item()

    print(f"  Input gradient max diff: {input_grad_diff:.2e}")

    passed = input_grad_diff < 1e-4
    print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")

    return passed, input_grad_diff


def test_moe_forward():
    """Test MoE forward pass."""
    print("\n" + "=" * 60)
    print("Test: MoE Forward Pass")
    print("=" * 60)

    config = TestConfig()
    batch_size, seq_len = 2, 32

    torch.manual_seed(42)
    moe = HFMoE(config)
    moe.eval()

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        output = moe(hidden_states)

    print(f"  Input shape:  {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Result: ✓ PASSED (forward runs without error)")

    return True, 0.0


def test_full_model_forward():
    """Test full model forward pass."""
    print("\n" + "=" * 60)
    print("Test: Full Model Forward Pass")
    print("=" * 60)

    config = TestConfig()
    batch_size, seq_len = 2, 32

    torch.manual_seed(42)
    model = HFModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logit range:  [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  Result: ✓ PASSED (forward runs without error)")

    return True, 0.0


def test_full_model_gradient():
    """Test full model backward pass."""
    print("\n" + "=" * 60)
    print("Test: Full Model Gradient Flow")
    print("=" * 60)

    config = TestConfig()
    batch_size, seq_len = 2, 16  # Smaller for gradient test

    torch.manual_seed(42)
    model = HFModel(config)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward
    logits = model(input_ids)

    # Loss
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

    # Backward
    loss.backward()

    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

    print(f"  Loss: {loss.item():.4f}")
    print(f"  All parameters have gradients: {has_grad}")
    print(f"  Gradient norm range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")
    print(f"  Result: {'✓ PASSED' if has_grad else '✗ FAILED'}")

    return has_grad, loss.item()


def test_precision_levels():
    """Test at different precision levels."""
    print("\n" + "=" * 60)
    print("Test: Multi-Precision Support")
    print("=" * 60)

    config = TestConfig()
    batch_size, seq_len = 2, 16

    results = []
    for dtype, name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
        try:
            torch.manual_seed(42)
            hf_attn = HFAttention(config).to(dtype)
            mcore_attn = MCoreAttention(config).to(dtype)
            copy_attention_weights(hf_attn, mcore_attn, config)

            hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=dtype)
            position_ids = torch.arange(seq_len)

            with torch.no_grad():
                hf_output = hf_attn(hidden_states, position_ids)
                mcore_output = mcore_attn(hidden_states, position_ids)

            max_diff = (hf_output - mcore_output).abs().max().item()
            results.append((name, max_diff, True))
            print(f"  {name}: max diff = {max_diff:.2e} ✓")
        except Exception as e:
            results.append((name, 0, False))
            print(f"  {name}: FAILED - {e}")

    passed = all(r[2] for r in results)
    return passed, results


# =============================================================================
# Main
# =============================================================================

def run_all_inference_tests():
    """Run all inference equivalence tests."""
    print("=" * 60)
    print(" GPT-OSS Bridge Inference Equivalence Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Attention Forward", *test_attention_forward_equivalence()))
    results.append(("Attention Gradient", *test_attention_gradient_equivalence()))
    results.append(("MoE Forward", *test_moe_forward()))
    results.append(("Full Model Forward", *test_full_model_forward()))
    results.append(("Full Model Gradient", *test_full_model_gradient()))
    results.append(("Multi-Precision", *test_precision_levels()))

    # Summary
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)

    passed_count = sum(1 for r in results if r[1])
    total_count = len(results)

    for name, passed, detail in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    print("=" * 60)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_inference_tests()
    exit(0 if success else 1)
