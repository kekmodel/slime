"""TP-compatible LoRA wrapper layers for Megatron parallel linears."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAColumnParallelLinear(nn.Module):
    """LoRA adapter wrapping a ColumnParallelLinear.

    Base weight: [output_size_per_partition, input_size] (sharded on dim=0)
    lora_A:      [rank, input_size]                      (replicated)
    lora_B:      [output_size_per_partition, rank]        (sharded on dim=0, matching base)
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_size = base_layer.input_size
        output_size = base_layer.output_size_per_partition

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # TP attributes — lora_B follows base weight partitioning
        self.lora_B.tensor_model_parallel = True
        self.lora_B.partition_dim = 0
        self.lora_B.partition_stride = 1
        # lora_A is replicated
        self.lora_A.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        # Early return when LoRA is disabled (scaling=0)
        if self.scaling == 0.0:
            return base_output, bias

        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_out, bias


class LoRARowParallelLinear(nn.Module):
    """LoRA adapter wrapping a RowParallelLinear.

    Base weight: [output_size, input_size_per_partition] (sharded on dim=1)
    lora_A:      [rank, input_size_per_partition]        (sharded on dim=1, matching base)
    lora_B:      [output_size, rank]                     (replicated)
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_size = base_layer.input_size_per_partition
        output_size = base_layer.output_size

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A.tensor_model_parallel = True
        self.lora_A.partition_dim = 1
        self.lora_A.partition_stride = 1
        self.lora_B.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        # Early return when LoRA is disabled (scaling=0)
        if self.scaling == 0.0:
            return base_output, bias

        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling

        # All-reduce lora_out across TP ranks to match base_output's already-reduced state.
        # RowParallelLinear reduces its output internally; LoRA output needs the same treatment.
        if hasattr(self.base_layer, "tp_group") and self.base_layer.tp_group is not None:
            if getattr(self.base_layer, "sequence_parallel", False):
                from megatron.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region

                lora_out = reduce_scatter_to_sequence_parallel_region(lora_out, group=self.base_layer.tp_group)
            elif not getattr(self.base_layer, "explicit_expert_comm", False):
                from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

                lora_out = reduce_from_tensor_model_parallel_region(lora_out, group=self.base_layer.tp_group)

        return base_output + lora_out, bias


def _interleave_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
) -> torch.Tensor:
    """Interleave Q, K, V activation corrections into Megatron's group layout.

    Megatron stores QKV fused as groups: [Q_0, K_0, V_0, Q_1, K_1, V_1, ...]
    where each group corresponds to a KV head and its associated Q heads.

    Args:
        q: [..., num_q_heads_per_tp * head_dim]
        k: [..., num_kv_heads_per_tp * head_dim]
        v: [..., num_kv_heads_per_tp * head_dim]
        num_q_heads_per_tp: number of Q heads per TP rank
        num_kv_heads_per_tp: number of KV heads per TP rank
        head_dim: dimension per head

    Returns:
        Interleaved tensor of shape [..., (num_q_heads_per_tp + 2*num_kv_heads_per_tp) * head_dim]
    """
    q_per_kv = num_q_heads_per_tp // num_kv_heads_per_tp

    # Reshape to expose head structure
    prefix = q.shape[:-1]
    q_heads = q.reshape(*prefix, num_kv_heads_per_tp, q_per_kv * head_dim)
    k_heads = k.reshape(*prefix, num_kv_heads_per_tp, head_dim)
    v_heads = v.reshape(*prefix, num_kv_heads_per_tp, head_dim)

    # Concatenate along head dim per KV group: [q_group, k, v]
    groups = torch.cat([q_heads, k_heads, v_heads], dim=-1)  # [..., num_kv_heads_per_tp, (q_per_kv+2)*head_dim]

    # Flatten back to [..., total_dim]
    return groups.reshape(*prefix, -1)


def _interleave_qkv_weight(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
) -> torch.Tensor:
    """Interleave Q, K, V weight corrections into Megatron's group layout.

    For weight tensors shaped [out_dim, in_dim], interleaves along dim=0.

    Args:
        q: [num_q_heads_per_tp * head_dim, in_dim]
        k: [num_kv_heads_per_tp * head_dim, in_dim]
        v: [num_kv_heads_per_tp * head_dim, in_dim]
        num_q_heads_per_tp: number of Q heads per TP rank
        num_kv_heads_per_tp: number of KV heads per TP rank
        head_dim: dimension per head

    Returns:
        Interleaved tensor of shape [(num_q_heads_per_tp + 2*num_kv_heads_per_tp) * head_dim, in_dim]
    """
    q_per_kv = num_q_heads_per_tp // num_kv_heads_per_tp
    in_dim = q.shape[1]

    # Reshape to expose head structure along output dim
    q_heads = q.reshape(num_kv_heads_per_tp, q_per_kv * head_dim, in_dim)
    k_heads = k.reshape(num_kv_heads_per_tp, head_dim, in_dim)
    v_heads = v.reshape(num_kv_heads_per_tp, head_dim, in_dim)

    # Concatenate along output head dim per KV group
    groups = torch.cat([q_heads, k_heads, v_heads], dim=1)  # [num_kv_heads, (q_per_kv+2)*head_dim, in_dim]

    # Flatten back
    return groups.reshape(-1, in_dim)


class LoRAFusedQKV(nn.Module):
    """LoRA adapter for Megatron's fused QKV ColumnParallelLinear.

    Megatron stores fused QKV weights in group layout:
        [Q_0, K_0, V_0, Q_1, K_1, V_1, ...] per KV head group.

    We use separate Q, K, V LoRA adapters and interleave their corrections
    to match the fused layout before adding to the base output.

    All lora_B parameters follow ColumnParallelLinear sharding (partition_dim=0).
    All lora_A parameters are replicated.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        num_q_heads_per_tp: int,
        num_kv_heads_per_tp: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.num_q_heads_per_tp = num_q_heads_per_tp
        self.num_kv_heads_per_tp = num_kv_heads_per_tp
        self.head_dim = head_dim

        input_size = base_layer.input_size
        q_out = num_q_heads_per_tp * head_dim
        kv_out = num_kv_heads_per_tp * head_dim

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Q adapter
        self.q_lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.q_lora_B = nn.Parameter(torch.zeros(q_out, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))

        # K adapter
        self.k_lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.k_lora_B = nn.Parameter(torch.zeros(kv_out, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.k_lora_A, a=math.sqrt(5))

        # V adapter
        self.v_lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.v_lora_B = nn.Parameter(torch.zeros(kv_out, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # TP attributes: all B matrices sharded on dim=0 (column parallel)
        for param in (self.q_lora_B, self.k_lora_B, self.v_lora_B):
            param.tensor_model_parallel = True
            param.partition_dim = 0
            param.partition_stride = 1

        # A matrices are replicated
        for param in (self.q_lora_A, self.k_lora_A, self.v_lora_A):
            param.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        # Early return when LoRA is disabled (scaling=0)
        if self.scaling == 0.0:
            return base_output, bias

        dropped = self.dropout(x)

        q_corr = F.linear(F.linear(dropped, self.q_lora_A), self.q_lora_B)
        k_corr = F.linear(F.linear(dropped, self.k_lora_A), self.k_lora_B)
        v_corr = F.linear(F.linear(dropped, self.v_lora_A), self.v_lora_B)

        lora_out = _interleave_qkv(
            q_corr,
            k_corr,
            v_corr,
            self.num_q_heads_per_tp,
            self.num_kv_heads_per_tp,
            self.head_dim,
        )

        return base_output + lora_out * self.scaling, bias


class LoRAFusedFC1(nn.Module):
    """LoRA adapter for Megatron's fused gate+up (FC1) ColumnParallelLinear.

    Megatron stores fused gate+up weights as [gate | up] concatenated along dim=0.
    We use separate gate and up LoRA adapters and concatenate their corrections.

    All lora_B parameters follow ColumnParallelLinear sharding (partition_dim=0).
    All lora_A parameters are replicated.
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_size = base_layer.input_size
        # Each half is half of the full output
        half_out = base_layer.output_size_per_partition // 2

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Gate adapter
        self.gate_lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.gate_lora_B = nn.Parameter(torch.zeros(half_out, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.gate_lora_A, a=math.sqrt(5))

        # Up adapter
        self.up_lora_A = nn.Parameter(torch.empty(rank, input_size, device=device, dtype=dtype))
        self.up_lora_B = nn.Parameter(torch.zeros(half_out, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.up_lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # TP attributes: B matrices sharded on dim=0
        for param in (self.gate_lora_B, self.up_lora_B):
            param.tensor_model_parallel = True
            param.partition_dim = 0
            param.partition_stride = 1

        # A matrices are replicated
        for param in (self.gate_lora_A, self.up_lora_A):
            param.tensor_model_parallel = False

    def forward(self, x, **kwargs):
        base_out = self.base_layer(x, **kwargs)
        if isinstance(base_out, tuple):
            base_output, bias = base_out
        else:
            base_output, bias = base_out, None

        # Early return when LoRA is disabled (scaling=0)
        if self.scaling == 0.0:
            return base_output, bias

        dropped = self.dropout(x)

        gate_corr = F.linear(F.linear(dropped, self.gate_lora_A), self.gate_lora_B)
        up_corr = F.linear(F.linear(dropped, self.up_lora_A), self.up_lora_B)

        # Concatenate [gate | up] to match Megatron's fused layout
        lora_out = torch.cat([gate_corr, up_corr], dim=-1)

        return base_output + lora_out * self.scaling, bias
