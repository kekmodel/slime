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

        self.lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank))
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

        self.lora_A = nn.Parameter(torch.empty(rank, input_size))
        self.lora_B = nn.Parameter(torch.zeros(output_size, rank))
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

        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_out, bias
