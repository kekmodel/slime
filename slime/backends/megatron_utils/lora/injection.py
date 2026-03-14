"""LoRA adapter injection into Megatron GPTModel."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch.nn as nn

from slime.backends.megatron_utils.lora.config import LoRAConfig
from slime.backends.megatron_utils.lora.utils import unwrap_ddp
from slime.backends.megatron_utils.lora.layers import (
    LoRAColumnParallelLinear,
    LoRAFusedFC1,
    LoRAFusedQKV,
    LoRARowParallelLinear,
)

logger = logging.getLogger(__name__)


def inject_lora_adapters(
    model: Sequence[nn.Module],
    config: LoRAConfig,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
) -> None:
    """Inject LoRA adapters into target modules of a GPTModel.

    [I6] Accepts list of model chunks (VP stages). Each chunk is unwrapped from DDP.
    Replaces target linear layers with LoRA-wrapped versions in-place.
    """
    if not config.enabled:
        return

    targets = config.target_modules
    has_qkv = any(m in targets for m in ("q_proj", "k_proj", "v_proj"))
    has_o = "o_proj" in targets
    has_fc1 = any(m in targets for m in ("gate_proj", "up_proj"))
    has_fc2 = "down_proj" in targets
    has_expert = "expert" in targets

    count = 0
    for model_chunk in model:
        unwrapped = unwrap_ddp(model_chunk)
        for layer in _get_decoder_layers(unwrapped):
            if has_qkv:
                attn = layer.self_attention
                if hasattr(attn, "linear_qkv"):
                    # Standard fused QKV attention
                    attn.linear_qkv = LoRAFusedQKV(
                        attn.linear_qkv,
                        rank=config.rank,
                        alpha=config.alpha,
                        num_q_heads_per_tp=num_q_heads_per_tp,
                        num_kv_heads_per_tp=num_kv_heads_per_tp,
                        head_dim=head_dim,
                        dropout=config.dropout,
                    )
                    count += 1
                else:
                    # MLA (Multi-Latent Attention): separate Q and KV projections.
                    # Target the up-projections (ColumnParallelLinear) which expand from
                    # compressed latent dims to per-head dims — most impactful for LoRA.
                    has_q = "q_proj" in targets
                    has_kv = "k_proj" in targets or "v_proj" in targets

                    if has_q:
                        # Prefer linear_q_up_proj (compressed Q path), fall back to linear_q_proj (direct Q)
                        q_attr = "linear_q_up_proj" if hasattr(attn, "linear_q_up_proj") else "linear_q_proj"
                        q_layer = getattr(attn, q_attr, None)
                        if q_layer is not None:
                            setattr(
                                attn,
                                q_attr,
                                LoRAColumnParallelLinear(
                                    q_layer, rank=config.rank, alpha=config.alpha, dropout=config.dropout
                                ),
                            )
                            count += 1

                    if has_kv and hasattr(attn, "linear_kv_up_proj"):
                        attn.linear_kv_up_proj = LoRAColumnParallelLinear(
                            attn.linear_kv_up_proj, rank=config.rank, alpha=config.alpha, dropout=config.dropout
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

            # [C1] MoE expert LoRA: inject into GroupedMLP
            if has_expert and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                experts_module = layer.mlp.experts
                if hasattr(experts_module, "weight1"):  # GroupedMLP
                    from slime.backends.megatron_utils.lora.expert_lora import inject_expert_lora

                    inject_expert_lora(
                        experts_module,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    )
                    count += 2  # FC1 + FC2

    logger.info(f"Injected {count} LoRA adapters (rank={config.rank}, alpha={config.alpha})")


def freeze_base_params(model: Sequence[nn.Module]) -> None:
    """[I6] Freeze all non-LoRA parameters across all VP chunks."""
    frozen_count = 0
    trainable_count = 0
    for model_chunk in model:
        unwrapped = unwrap_ddp(model_chunk)
        for name, param in unwrapped.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
    logger.info(f"Frozen {frozen_count} base params, {trainable_count} LoRA params trainable")


def _get_decoder_layers(model: nn.Module):
    """Yield transformer layers from an unwrapped model."""
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        yield from model.decoder.layers
    else:
        # Fallback: search children
        for child in model.children():
            if hasattr(child, "decoder") and hasattr(child.decoder, "layers"):
                yield from child.decoder.layers
                return
