# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from mbridge.core import LLMBridge, register_model


@register_model("gpt_oss")
class GptOssBridge(LLMBridge):
    """
    Bridge implementation for GPT-OSS models (e.g., GPT-OSS-120B).

    This class extends LLMBridge to provide specific configurations and
    optimizations for GPT-OSS models, handling the conversion between
    Hugging Face GPT-OSS format and Megatron-Core.

    Key features:
    - MoE architecture with 128 experts, top-4 routing (fused grouped gemm)
    - GQA (Grouped Query Attention) with 64 heads, 8 KV heads
    - Learnable sink tokens for attention anchoring
    - Sliding window attention (128) mixed with full attention
    - RMSNorm with pre-norm structure
    - GLU gating with clamping (alpha=1.702, limit=7.0)
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        # GPT-OSS specific: learnable sink tokens for attention anchoring
        "self_attention.sinks": ["model.layers.{layer_number}.self_attn.sinks"],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        # MoE router with bias
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
        # Fused MoE experts (grouped gemm format - all experts in single tensor)
        "mlp.experts.linear_fc1.weight": ["model.layers.{layer_number}.mlp.experts.gate_up_proj"],
        "mlp.experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.experts.down_proj"],
    }

    def _build_config(self):
        """
        Build the configuration for GPT-OSS models.

        Configures GPT-OSS-specific parameters including MoE settings,
        GQA configuration, and sliding window attention.

        Returns:
            TransformerConfig: Configuration object for GPT-OSS models
        """
        hf_config = self.hf_config

        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific (fused grouped gemm)
            moe_ffn_hidden_size=hf_config.intermediate_size,
            moe_router_topk=hf_config.num_experts_per_tok,
            num_moe_experts=hf_config.num_local_experts,
            moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=False,
            # No shared experts in GPT-OSS
            moe_shared_expert_intermediate_size=None,
            moe_shared_expert_overlap=False,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # GPT-OSS specific
            add_qkv_bias=False,
            add_bias_linear=False,
            qk_layernorm=False,
            rotary_interleaved=False,
        )

    def _get_transformer_layer_spec(self):
        """
        Gets the transformer layer specification.

        Creates and returns a specification for the transformer layers based on
        the current configuration.

        Returns:
            TransformerLayerSpec: Specification for transformer layers
        """
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.config.num_moe_experts,
            moe_grouped_gemm=self.config.moe_grouped_gemm,
            qk_layernorm=self.config.qk_layernorm,
        )
        return transformer_layer_spec

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "self_attention" in mcore_weights_name or "sinks" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        """
        Map attention-related MCore weight names to HuggingFace weight names.

        Args:
            name: MCore weight name containing attention parameters

        Returns:
            list: Corresponding HuggingFace weight names
        """
        layer_number = name.split(".")[2]
        convert_names = []

        for keyword, mapping_names in self._ATTENTION_MAPPING.items():
            if keyword in name:
                convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break

        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported attention parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """
        Map MLP-related MCore weight names to HuggingFace weight names.

        Handles fused expert weights (grouped gemm format).

        Args:
            name: MCore weight name containing MLP parameters

        Returns:
            list: Corresponding HuggingFace weight names
        """
        layer_number = name.split(".")[2]
        convert_names = []

        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break

        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported MLP parameter name: {name}")
        return convert_names

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert HuggingFace weights to Megatron-Core format.

        Handles special cases:
        - QKV merging for GQA
        - Sink tokens passthrough
        - Fused expert weights for grouped gemm

        Args:
            mcore_weights_name: Target MCore weight name
            hf_weights: List of HuggingFace weight tensors

        Returns:
            torch.Tensor: Converted weight tensor for MCore
        """
        # Handle QKV merging for GQA
        if "self_attention.linear_qkv.weight" in mcore_weights_name:
            assert len(hf_weights) == 3, f"Expected 3 tensors for QKV, got {len(hf_weights)}"
            q, k, v = hf_weights

            num_key_value_heads = self.hf_config.num_key_value_heads
            num_attention_heads = self.hf_config.num_attention_heads
            hidden_size = self.hf_config.hidden_size
            head_dim = getattr(self.hf_config, "head_dim", hidden_size // num_attention_heads)

            # Reshape for GQA interleaving
            num_groups = num_attention_heads // num_key_value_heads

            # q: [num_heads * head_dim, hidden_size] -> [num_kv_heads, num_groups, head_dim, hidden_size]
            q = q.view(num_key_value_heads, num_groups, head_dim, -1)
            # k: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, 1, head_dim, hidden_size]
            k = k.view(num_key_value_heads, 1, head_dim, -1)
            # v: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, 1, head_dim, hidden_size]
            v = v.view(num_key_value_heads, 1, head_dim, -1)

            # Interleave: [Q_group, K, V] for each KV head group
            qkv = torch.cat([q, k, v], dim=1)  # [num_kv_heads, num_groups+2, head_dim, hidden]
            qkv = qkv.reshape(-1, hidden_size)

            return qkv.contiguous()

        # Handle sink tokens (passthrough)
        if "sinks" in mcore_weights_name:
            assert len(hf_weights) == 1
            return hf_weights[0]

        # Handle fused expert weights (already in grouped gemm format)
        if "mlp.experts" in mcore_weights_name:
            assert len(hf_weights) == 1
            return hf_weights[0]

        # Default: single weight tensor
        if len(hf_weights) == 1:
            return hf_weights[0]

        # gate_up fusion (if needed)
        if len(hf_weights) == 2:
            return torch.cat(hf_weights, dim=0)

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Convert Megatron-Core weights to HuggingFace format.

        Handles special cases:
        - QKV splitting for GQA export
        - Sink tokens passthrough
        - Fused expert weights

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor

        Returns:
            tuple: (list of HF weight names, list of HF weight tensors)
        """
        layer_number = mcore_weights_name.split(".")[2] if "layers" in mcore_weights_name else None

        # Handle QKV splitting for GQA
        if "self_attention.linear_qkv.weight" in mcore_weights_name:
            num_key_value_heads = self.hf_config.num_key_value_heads
            num_attention_heads = self.hf_config.num_attention_heads
            hidden_size = self.hf_config.hidden_size
            head_dim = getattr(self.hf_config, "head_dim", hidden_size // num_attention_heads)
            num_groups = num_attention_heads // num_key_value_heads

            # Reshape: [total_dim, hidden] -> [num_kv_heads, num_groups+2, head_dim, hidden]
            qkv = mcore_weights.view(num_key_value_heads, num_groups + 2, head_dim, -1)

            # Split back
            q = qkv[:, :num_groups, :, :].reshape(-1, hidden_size)
            k = qkv[:, num_groups, :, :].reshape(-1, hidden_size)
            v = qkv[:, num_groups + 1, :, :].reshape(-1, hidden_size)

            hf_names = [n.format(layer_number=layer_number) for n in self._ATTENTION_MAPPING["self_attention.linear_qkv.weight"]]

            return hf_names, [q.contiguous(), k.contiguous(), v.contiguous()]

        # Handle sink tokens (passthrough)
        if "sinks" in mcore_weights_name:
            hf_name = f"model.layers.{layer_number}.self_attn.sinks"
            return [hf_name], [mcore_weights]

        # Handle fused expert weights (passthrough - already in correct format)
        if "mlp.experts" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mlp(mcore_weights_name)
            return hf_names, [mcore_weights]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
