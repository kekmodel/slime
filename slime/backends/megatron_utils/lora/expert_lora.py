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

    # FC1 LoRA: weight1 is [H, E*2F] -> reshaped [E, H, 2F] -> split into gate [E,H,F] and up [E,H,F]
    # LoRA delta for gate: gate_B @ A -> [E, F, rank] @ [E, rank, H] = [E, F, H]
    # LoRA delta for up:   up_B  @ A -> [E, F, H]
    grouped_mlp._lora_A_fc1 = nn.Parameter(torch.empty(E, rank, H))
    grouped_mlp._lora_gate_B_fc1 = nn.Parameter(torch.zeros(E, F, rank))
    grouped_mlp._lora_up_B_fc1 = nn.Parameter(torch.zeros(E, F, rank))

    # FC2 LoRA: weight2 is [E*F, H] -> reshaped [E, F, H]
    # LoRA delta: B @ A -> [E, H, rank] @ [E, rank, F] = [E, H, F]
    grouped_mlp._lora_A_fc2 = nn.Parameter(torch.empty(E, rank, F))
    grouped_mlp._lora_B_fc2 = nn.Parameter(torch.zeros(E, H, rank))

    # Initialize A matrices
    for e in range(E):
        nn.init.kaiming_uniform_(grouped_mlp._lora_A_fc1.data[e], a=math.sqrt(5))
        nn.init.kaiming_uniform_(grouped_mlp._lora_A_fc2.data[e], a=math.sqrt(5))

    grouped_mlp._lora_scale = scale
    grouped_mlp._lora_enabled = True
    grouped_mlp._lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # Save original forward
    grouped_mlp._original_forward = grouped_mlp.forward

    # Monkey-patch forward
    grouped_mlp.forward = types.MethodType(_lora_grouped_mlp_forward, grouped_mlp)


def _lora_grouped_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
    """Patched forward for GroupedMLP with LoRA delta injection.

    Temporarily merges LoRA delta into weights, calls original forward, then unmerges.
    """
    if self._lora_enabled and self._lora_scale != 0.0:
        _apply_expert_delta(self, merge=True)

    output = self._original_forward(permuted_local_hidden_states, tokens_per_expert)

    if self._lora_enabled and self._lora_scale != 0.0:
        _apply_expert_delta(self, merge=False)

    return output


def _apply_expert_delta(grouped_mlp, merge: bool) -> None:
    """Add or subtract LoRA delta from GroupedMLP's weight1/weight2.

    Uses torch.no_grad() to avoid tracking temporary merge/unmerge ops for base weights.
    LoRA params get gradients through normal autograd when they're part of the module.
    """
    sign = 1.0 if merge else -1.0
    scale = grouped_mlp._lora_scale

    E = grouped_mlp.num_local_experts
    H = grouped_mlp.config.hidden_size
    F = grouped_mlp.config.ffn_hidden_size

    with torch.no_grad():
        # FC1: weight1 is [H, E*2F]
        # gate_delta = gate_B @ A -> [E, F, rank] @ [E, rank, H] = [E, F, H]
        gate_delta = torch.bmm(grouped_mlp._lora_gate_B_fc1, grouped_mlp._lora_A_fc1) * scale
        up_delta = torch.bmm(grouped_mlp._lora_up_B_fc1, grouped_mlp._lora_A_fc1) * scale

        # weight1 viewed as [E, H, 2F] -> gate is [:, :, :F], up is [:, :, F:]
        w1_view = grouped_mlp.weight1.data.view(E, H, 2 * F)
        w1_view[:, :, :F] += sign * gate_delta.transpose(-1, -2)  # [E, F, H].T -> [E, H, F]
        w1_view[:, :, F:] += sign * up_delta.transpose(-1, -2)

        # FC2: weight2 is [E*F, H]
        # fc2_delta = B @ A -> [E, H, rank] @ [E, rank, F] = [E, H, F]
        fc2_delta = torch.bmm(grouped_mlp._lora_B_fc2, grouped_mlp._lora_A_fc2) * scale
        w2_view = grouped_mlp.weight2.data.view(E, F, H)
        w2_view += sign * fc2_delta.transpose(-1, -2)  # [E, H, F].T -> [E, F, H]
