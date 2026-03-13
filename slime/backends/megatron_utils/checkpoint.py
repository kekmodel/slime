import logging
import os
import re
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)


def load_lora_adapter(model, adapter_path: str) -> None:
    """Load LoRA adapter weights on top of already-injected model.

    The adapter checkpoint contains un-sharded (full) tensors.
    Each TP rank extracts its shard based on the TP attributes set during injection.
    PP > 1: loads per-PP-rank files (adapter_model_pp{rank:02d}.bin).
    EP > 1: extracts this EP rank's expert shard from the full expert set.
    """
    import torch
    from megatron.core import parallel_state as mpu

    adapter_dir = Path(adapter_path)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()

    # Determine which file to load based on PP configuration
    pp_file = adapter_dir / f"adapter_model_pp{pp_rank:02d}.bin"
    if pp_size > 1 and pp_file.exists():
        adapter_file = pp_file
    else:
        adapter_file = adapter_dir / "adapter_model.bin"

    adapter_state = torch.load(adapter_file, map_location="cpu", weights_only=True)

    # VP (virtual pipeline): checkpoint may use vp_stage-prefixed keys to avoid
    # name collisions when multiple VP chunks share the same PP rank.
    num_vp_chunks = len(model)

    loaded = 0
    for vp_stage, model_chunk in enumerate(model):
        unwrapped = model_chunk.module if hasattr(model_chunk, "module") else model_chunk
        for name, param in unwrapped.named_parameters():
            # Try VP-prefixed key first, fall back to raw name (backward compat)
            vp_name = f"vp_stages.{vp_stage}.{name}" if num_vp_chunks > 1 else name
            if vp_name in adapter_state:
                lookup_name = vp_name
            elif name in adapter_state:
                lookup_name = name
            else:
                continue
            full_tensor = adapter_state[lookup_name]

            # TP shard based on TP attributes
            if getattr(param, "tensor_model_parallel", False) and tp_size > 1:
                dim = param.partition_dim
                chunk_size = full_tensor.shape[dim] // tp_size
                full_tensor = full_tensor.narrow(dim, tp_rank * chunk_size, chunk_size)

            # EP shard for expert LoRA params (dim=0 is the expert dimension)
            if ep_size > 1 and ".experts." in name and "_lora_" in name:
                e_total = full_tensor.shape[0]
                e_local = e_total // ep_size
                full_tensor = full_tensor[ep_rank * e_local : (ep_rank + 1) * e_local]

            param.data.copy_(full_tensor.to(param.device))
            loaded += 1

    logger.info(f"Loaded {loaded} LoRA adapter params from {adapter_path}")
