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
        # Expert LoRA handled separately (hasattr check for _lora_A_fc1)
        elif hasattr(module, "_lora_enabled") and hasattr(module, "_lora_A_fc1"):
            from slime.backends.megatron_utils.lora.expert_lora import _apply_expert_delta

            _apply_expert_delta(module, merge=True)


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
        elif hasattr(module, "_lora_enabled") and hasattr(module, "_lora_A_fc1"):
            from slime.backends.megatron_utils.lora.expert_lora import _apply_expert_delta

            _apply_expert_delta(module, merge=False)


def _merge_fused_qkv(module: LoRAFusedQKV) -> None:
    """[C3] Merge Q/K/V LoRA into fused QKV weight. Uses _interleave_qkv_weight (dim=0)."""
    q_delta = (module.q_lora_B @ module.q_lora_A) * module.scaling  # [q_out, H]
    k_delta = (module.k_lora_B @ module.k_lora_A) * module.scaling  # [kv_out, H]
    v_delta = (module.v_lora_B @ module.v_lora_A) * module.scaling  # [kv_out, H]

    delta = _interleave_qkv_weight(
        q_delta,
        k_delta,
        v_delta,
        module.num_q_heads_per_tp,
        module.num_kv_heads_per_tp,
        module.head_dim,
    )
    module.base_layer.weight.data += delta


def _unmerge_fused_qkv(module: LoRAFusedQKV) -> None:
    q_delta = (module.q_lora_B @ module.q_lora_A) * module.scaling
    k_delta = (module.k_lora_B @ module.k_lora_A) * module.scaling
    v_delta = (module.v_lora_B @ module.v_lora_A) * module.scaling

    delta = _interleave_qkv_weight(
        q_delta,
        k_delta,
        v_delta,
        module.num_q_heads_per_tp,
        module.num_kv_heads_per_tp,
        module.head_dim,
    )
    module.base_layer.weight.data -= delta


def _merge_fused_fc1(module: LoRAFusedFC1) -> None:
    gate_delta = (module.gate_lora_B @ module.gate_lora_A) * module.scaling
    up_delta = (module.up_lora_B @ module.up_lora_A) * module.scaling
    delta = torch.cat([gate_delta, up_delta], dim=0)
    module.base_layer.weight.data += delta


def _unmerge_fused_fc1(module: LoRAFusedFC1) -> None:
    gate_delta = (module.gate_lora_B @ module.gate_lora_A) * module.scaling
    up_delta = (module.up_lora_B @ module.up_lora_A) * module.scaling
    delta = torch.cat([gate_delta, up_delta], dim=0)
    module.base_layer.weight.data -= delta


def merge_expert_lora(grouped_mlp) -> None:
    """Merge expert LoRA deltas into GroupedMLP weight1/weight2 permanently."""
    from slime.backends.megatron_utils.lora.expert_lora import _apply_expert_delta

    _apply_expert_delta(grouped_mlp, merge=True)


def unmerge_expert_lora(grouped_mlp) -> None:
    """Unmerge expert LoRA deltas from GroupedMLP weight1/weight2."""
    from slime.backends.megatron_utils.lora.expert_lora import _apply_expert_delta

    _apply_expert_delta(grouped_mlp, merge=False)


def disable_lora(model: nn.Module) -> None:
    """Set all LoRA scaling to 0 (model behaves as base only). Used for ref forward."""
    for module in model.modules():
        if isinstance(module, _LORA_LAYER_TYPES):
            module._saved_scaling = module.scaling
            module.scaling = 0.0
        # Expert LoRA
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
