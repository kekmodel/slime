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

    # Expert LoRA params are NOT TP-sharded (experts are EP-sharded, not TP-sharded within each rank).
    # Mark all as non-TP so all_gather_param in adapter save handles them correctly.
    for p in [
        grouped_mlp._lora_A_fc1,
        grouped_mlp._lora_gate_B_fc1,
        grouped_mlp._lora_up_B_fc1,
        grouped_mlp._lora_A_fc2,
        grouped_mlp._lora_B_fc2,
    ]:
        p.tensor_model_parallel = False

    grouped_mlp._lora_scale = scale
    grouped_mlp._lora_enabled = True
    grouped_mlp._lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # Save original forward
    grouped_mlp._original_forward = grouped_mlp.forward

    # Monkey-patch forward
    grouped_mlp.forward = types.MethodType(_lora_grouped_mlp_forward, grouped_mlp)


def _lora_grouped_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
    """Patched forward for GroupedMLP with LoRA delta injection.

    Computes LoRA deltas with gradient tracking and adds to cloned weights so that
    autograd can backpropagate through the LoRA parameters. Base weights (frozen) are
    detached to avoid unnecessary gradient computation.
    """
    if not self._lora_enabled or self._lora_scale == 0.0:
        return self._original_forward(permuted_local_hidden_states, tokens_per_expert)

    E = self.num_local_experts
    H = self.config.hidden_size
    F = self.config.ffn_hidden_size
    scale = self._lora_scale

    # Compute LoRA deltas WITH gradient tracking (no torch.no_grad!)
    gate_delta = torch.bmm(self._lora_gate_B_fc1, self._lora_A_fc1) * scale  # [E, F, H]
    up_delta = torch.bmm(self._lora_up_B_fc1, self._lora_A_fc1) * scale  # [E, F, H]
    fc2_delta = torch.bmm(self._lora_B_fc2, self._lora_A_fc2) * scale  # [E, H, F]

    # Build modified weight1: base (detached) + LoRA delta
    # weight1 is [H, E*2F], viewed as [E, H, 2F] where [:,:,:F]=gate, [:,:,F:]=up
    w1_base = self.weight1.detach().view(E, H, 2 * F)
    w1_gate = w1_base[:, :, :F] + gate_delta.transpose(-1, -2)  # [E, H, F]
    w1_up = w1_base[:, :, F:] + up_delta.transpose(-1, -2)  # [E, H, F]
    w1_modified = torch.cat([w1_gate, w1_up], dim=-1).reshape_as(self.weight1)  # [H, E*2F]

    # Build modified weight2: base (detached) + LoRA delta
    # weight2 is [E*F, H], viewed as [E, F, H]
    w2_base = self.weight2.detach().view(E, F, H)
    w2_modified = (w2_base + fc2_delta.transpose(-1, -2)).reshape_as(self.weight2)  # [E*F, H]

    # Temporarily swap weights via _parameters dict to bypass Module.__setattr__
    # (which rejects non-Parameter tensors). The original forward accesses self.weight1/weight2
    # which goes through __getattr__ -> _parameters lookup.
    orig_w1 = self._parameters["weight1"]
    orig_w2 = self._parameters["weight2"]
    self._parameters["weight1"] = w1_modified
    self._parameters["weight2"] = w2_modified
    try:
        output = self._original_forward(permuted_local_hidden_states, tokens_per_expert)
    finally:
        self._parameters["weight1"] = orig_w1
        self._parameters["weight2"] = orig_w2

    return output


def _apply_expert_delta(grouped_mlp, merge: bool) -> None:
    """Add or subtract LoRA delta from GroupedMLP's weight1/weight2 in-place.

    Used by merge/unmerge utilities (NOT the forward pass). Operates under no_grad
    since this modifies base weight .data directly for weight transfer to SGLang.
    """
    sign = 1.0 if merge else -1.0
    scale = grouped_mlp._lora_scale

    E = grouped_mlp.num_local_experts
    H = grouped_mlp.config.hidden_size
    F = grouped_mlp.config.ffn_hidden_size

    with torch.no_grad():
        # FC1: weight1 is [H, E*2F]
        gate_delta = torch.bmm(grouped_mlp._lora_gate_B_fc1, grouped_mlp._lora_A_fc1) * scale
        up_delta = torch.bmm(grouped_mlp._lora_up_B_fc1, grouped_mlp._lora_A_fc1) * scale

        w1_view = grouped_mlp.weight1.data.view(E, H, 2 * F)
        w1_view[:, :, :F] += sign * gate_delta.transpose(-1, -2)
        w1_view[:, :, F:] += sign * up_delta.transpose(-1, -2)

        # FC2: weight2 is [E*F, H]
        fc2_delta = torch.bmm(grouped_mlp._lora_B_fc2, grouped_mlp._lora_A_fc2) * scale
        w2_view = grouped_mlp.weight2.data.view(E, F, H)
        w2_view += sign * fc2_delta.transpose(-1, -2)
