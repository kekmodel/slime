"""
Parallel State for FSDP

This file defines the parallel state for distributed training.
Simplified from slime/backends/fsdp_utils/parallel.py.
"""

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class ParallelState:
    """
    Parallel state for distributed training.

    Attributes:
        dp_size: Data parallel world size
        dp_rank: Data parallel rank
        dp_group: Process group for data parallel
        tp_size: Tensor parallel size (default 1 for FSDP)
        tp_rank: Tensor parallel rank
        tp_group: Tensor parallel process group
        cp_size: Context parallel size (default 1)
        cp_rank: Context parallel rank
    """
    dp_size: int = 1
    dp_rank: int = 0
    dp_group: Any = None
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: Any = None
    cp_size: int = 1
    cp_rank: int = 0
    dp_mesh: Any = None

    @property
    def dp_cp_size(self) -> int:
        """Combined DP and CP size."""
        return self.dp_size * self.cp_size

    @property
    def dp_cp_rank(self) -> int:
        """Combined DP and CP rank."""
        return self.dp_rank * self.cp_size + self.cp_rank


def create_parallel_state(args: Namespace) -> ParallelState:
    """
    Create parallel state from arguments.

    For FSDP, we use data parallelism across all GPUs.
    Tensor parallelism is handled by FSDP sharding.

    Args:
        args: Training arguments

    Returns:
        ParallelState object
    """
    if not dist.is_initialized():
        return ParallelState()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # For FSDP, all ranks are in the same DP group
    dp_group = dist.new_group(list(range(world_size)))

    # Create device mesh for FSDP2
    try:
        from torch.distributed.device_mesh import init_device_mesh
        dp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    except ImportError:
        dp_mesh = None

    return ParallelState(
        dp_size=world_size,
        dp_rank=rank,
        dp_group=dp_group,
        dp_mesh=dp_mesh,
    )
