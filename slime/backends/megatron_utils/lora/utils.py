"""Shared utilities for LoRA and Megatron model handling."""


def unwrap_ddp(model_chunk):
    """Strip one level of DDP wrapper (DistributedDataParallel.module)."""
    return model_chunk.module if hasattr(model_chunk, "module") else model_chunk
