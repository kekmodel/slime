"""
Megatron Actor for nano_slime

This file implements a Megatron-based training actor with Ray.
Based on slime/backends/megatron_utils/actor.py.

Key components:
1. Ray Actor for distributed training
2. Megatron model loading and parallelism
3. Forward/backward pass with pipeline parallelism
4. Weight synchronization with SGLang engines

Usage:
    # Create Ray actors
    actors = [MegatronTrainRayActor.remote(...) for _ in range(num_gpus)]
    ray.get([a.init.remote(args, "actor") for a in actors])

    # Training loop
    for rollout_id in range(num_rollouts):
        ray.get([a.train.remote(rollout_id, rollout_data_ref) for a in actors])
        ray.get([a.update_weights.remote() for a in actors])
"""

import logging
import os
from argparse import Namespace
from typing import Any

try:
    import ray
    HAS_RAY = True
except ImportError:
    ray = None
    HAS_RAY = False

import torch
import torch.distributed as dist

from ..training_utils.loss import compute_advantages_and_returns

logger = logging.getLogger(__name__)


def get_local_gpu_id() -> int:
    """Get local GPU ID from CUDA_VISIBLE_DEVICES or Ray."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        if HAS_RAY:
            gpu_ids = ray.get_gpu_ids()
            return gpu_ids[0] if gpu_ids else 0
        return 0
    else:
        if HAS_RAY:
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                return cvd.split(",").index(str(gpu_ids[0]))
        return 0


class MegatronTrainRayActor:
    """
    Megatron training actor with Ray.

    This class wraps Megatron model training in a Ray actor
    for distributed RLHF training.

    Key features:
    1. Tensor Parallelism (TP) - Split model across GPUs
    2. Pipeline Parallelism (PP) - Split layers across stages
    3. Data Parallelism (DP) - Replicate model for larger batches
    4. Weight backup/restore for ref model switching
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        master_addr: str | None = None,
        master_port: int | None = None,
    ):
        """
        Initialize Ray actor.

        Args:
            world_size: Total number of GPUs
            rank: This actor's rank
            master_addr: Master node address
            master_port: Master node port
        """
        self._world_size = world_size
        self._rank = rank
        self.master_addr = master_addr or os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = master_port or int(os.environ.get("MASTER_PORT", "29500"))

        # Set environment variables
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int:
        """
        Initialize Megatron model and optimizer.

        Args:
            args: Training arguments
            role: "actor" or "critic"
            with_ref: Whether to load reference model

        Returns:
            Starting rollout ID
        """
        self.args = args
        self.role = role
        self.with_ref = with_ref

        # Set CUDA device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(
                backend=getattr(args, "distributed_backend", "nccl"),
            )

        logger.info(f"[Rank {self._rank}] Initializing Megatron Actor")

        # Initialize Megatron
        self._init_megatron(args)

        # Initialize model and optimizer
        self.model, self.optimizer, self.opt_param_scheduler = self._init_model_and_optimizer(args, role)

        # Create parallel state
        self.parallel_state = self._create_parallel_state()

        # Weight backup for model switching (actor/ref)
        self.weights_backup = {}
        self._active_model_tag = "actor"

        if with_ref:
            # Backup actor weights and load ref
            self._backup_weights("actor")
            self._load_ref_model(args)

        # Setup rollout engine connection
        self.rollout_manager = None
        self.rollout_engines = []

        logger.info(f"[Rank {self._rank}] Megatron Actor initialized")
        return 0

    def _init_megatron(self, args: Namespace) -> None:
        """Initialize Megatron-LM."""
        try:
            from megatron.core import mpu
            from slime.backends.megatron_utils.initialize import init

            init(args)
            self.mpu = mpu
            logger.info(f"[Rank {self._rank}] Megatron initialized")
        except ImportError as e:
            logger.warning(f"Megatron not available: {e}")
            logger.warning("Using mock Megatron for demonstration")
            self.mpu = None

    def _init_model_and_optimizer(
        self,
        args: Namespace,
        role: str,
    ) -> tuple[Any, Any, Any]:
        """
        Initialize model and optimizer.

        For actual Megatron, this calls initialize_model_and_optimizer.
        For nano_slime demo, we create a simple model.
        """
        try:
            from slime.backends.megatron_utils.model import initialize_model_and_optimizer

            model, optimizer, scheduler, _ = initialize_model_and_optimizer(args, role)
            return model, optimizer, scheduler
        except ImportError:
            # Mock model for demonstration
            logger.warning("Using mock model (Megatron not available)")
            return None, None, None

    def _create_parallel_state(self) -> Any:
        """Create parallel state from Megatron."""
        try:
            from slime.backends.megatron_utils.parallel import create_megatron_parallel_state

            return create_megatron_parallel_state(model=self.model)
        except ImportError:
            # Return simple parallel state
            from slime.utils.types import ParallelState

            return ParallelState(
                dp_size=self._world_size,
                dp_rank=self._rank,
            )

    def _backup_weights(self, tag: str) -> None:
        """Backup current model weights."""
        if self.model is None:
            return
        self.weights_backup[tag] = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }

    def _restore_weights(self, tag: str) -> None:
        """Restore weights from backup."""
        if self.model is None or tag not in self.weights_backup:
            return
        self.model.load_state_dict(self.weights_backup[tag])
        self._active_model_tag = tag

    def _switch_model(self, target_tag: str) -> None:
        """Switch to a different model (actor/ref)."""
        if target_tag == self._active_model_tag:
            return
        self._restore_weights(target_tag)

    def _load_ref_model(self, args: Namespace) -> None:
        """Load reference model weights."""
        ref_path = getattr(args, "ref_load", None) or getattr(args, "load", None)
        if ref_path:
            logger.info(f"[Rank {self._rank}] Loading ref model from {ref_path}")
            # In real Megatron, this loads checkpoint
            self._backup_weights("ref")

    def train(self, rollout_id: int, rollout_data_ref: Any) -> dict:
        """
        Run one training iteration.

        Args:
            rollout_id: Current rollout iteration
            rollout_data_ref: Ray object reference to rollout data

        Returns:
            Training metrics
        """
        if self.args.offload_train:
            self.wake_up()

        logger.info(f"[Rank {self._rank}] Training rollout {rollout_id}")

        # Get rollout data
        if HAS_RAY and hasattr(rollout_data_ref, 'get'):
            rollout_data = ray.get(rollout_data_ref.get())
        else:
            rollout_data = rollout_data_ref

        # Compute ref log_probs if needed
        if self.with_ref and "ref" in self.weights_backup:
            self._switch_model("ref")
            ref_log_probs = self._compute_log_probs(rollout_data, prefix="ref_")
            rollout_data.update(ref_log_probs)
            self._switch_model("actor")

        # Compute actor log_probs
        if not self.args.use_rollout_logprobs:
            actor_log_probs = self._compute_log_probs(rollout_data, prefix="")
            rollout_data.update(actor_log_probs)

        # Compute advantages and returns
        compute_advantages_and_returns(self.args, self.parallel_state, rollout_data)

        # Training step
        metrics = self._train_step(rollout_id, rollout_data)

        # Backup updated weights
        self._backup_weights("actor")

        if self.args.offload_train:
            self.sleep()

        return metrics

    def _compute_log_probs(
        self,
        rollout_data: dict,
        prefix: str = "",
    ) -> dict:
        """
        Compute log probabilities using Megatron forward.

        Args:
            rollout_data: Data containing tokens
            prefix: Key prefix ("ref_" for ref model)

        Returns:
            Dict with log_probs
        """
        if self.model is None:
            # Mock log_probs
            tokens = rollout_data.get("tokens", [])
            return {
                f"{prefix}log_probs": [
                    torch.randn(len(t) if hasattr(t, "__len__") else 100) - 5.0
                    for t in tokens
                ]
            }

        try:
            from slime.backends.megatron_utils.model import forward_only
            from slime.backends.training_utils.loss import get_log_probs_and_entropy

            # Create data iterator
            from slime.backends.training_utils.data import get_data_iterator

            data_iterator, num_microbatches = get_data_iterator(
                self.args, self.model, self.parallel_state, rollout_data
            )

            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                self.parallel_state,
                store_prefix=prefix,
            )
        except ImportError:
            # Mock implementation
            tokens = rollout_data.get("tokens", [])
            return {
                f"{prefix}log_probs": [
                    torch.randn(len(t) if hasattr(t, "__len__") else 100) - 5.0
                    for t in tokens
                ]
            }

    def _train_step(self, rollout_id: int, rollout_data: dict) -> dict:
        """
        Execute one training step.

        Args:
            rollout_id: Rollout iteration
            rollout_data: Training data

        Returns:
            Training metrics
        """
        if self.model is None:
            # Mock training
            import random
            return {
                "train/loss": random.uniform(0.1, 0.5),
                "train/pg_loss": random.uniform(0.05, 0.3),
                "train/rollout_id": rollout_id,
            }

        try:
            from slime.backends.megatron_utils.model import train
            from slime.backends.training_utils.data import get_data_iterator

            data_iterator, num_microbatches = get_data_iterator(
                self.args, self.model, self.parallel_state, rollout_data
            )

            # Set loss type
            self.args.loss_type = "policy_loss"

            train(
                rollout_id,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                data_iterator,
                num_microbatches,
                self.parallel_state,
            )

            return {"train/rollout_id": rollout_id}
        except ImportError:
            import random
            return {
                "train/loss": random.uniform(0.1, 0.5),
                "train/rollout_id": rollout_id,
            }

    def update_weights(self) -> None:
        """
        Synchronize weights to rollout engines.

        For Megatron, this involves:
        1. Gathering weights from TP/PP groups
        2. Converting to HF format
        3. Sending to SGLang engines
        """
        if not self.rollout_engines:
            logger.warning("No rollout engines to update")
            return

        logger.info(f"[Rank {self._rank}] Updating weights to rollout engines")

        try:
            from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers

            # Get weights (this handles TP/PP gathering)
            state_dict = dict(named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=True,
            ))

            # Update engines (only rank 0 does actual update)
            if self._rank == 0:
                for engine in self.rollout_engines:
                    if HAS_RAY:
                        ray.get(engine.update_weights.remote(state_dict))
                    else:
                        engine.update_weights(state_dict)

        except ImportError:
            # Mock update
            if self._rank == 0:
                for engine in self.rollout_engines:
                    if hasattr(engine, "update_weights"):
                        engine.update_weights({})

        if dist.is_initialized():
            dist.barrier()

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Save checkpoint."""
        if self.model is None:
            return

        try:
            from slime.backends.megatron_utils.model import save

            save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)
        except ImportError:
            logger.warning("Megatron save not available")

    def sleep(self) -> None:
        """Offload model to CPU for colocate mode."""
        if not getattr(self.args, "offload_train", False):
            return

        try:
            from torch_memory_saver import torch_memory_saver
            torch_memory_saver.pause()
        except ImportError:
            if self.model is not None:
                self.model.cpu()
                torch.cuda.empty_cache()

    def wake_up(self) -> None:
        """Move model back to GPU."""
        if not getattr(self.args, "offload_train", False):
            return

        try:
            from torch_memory_saver import torch_memory_saver
            torch_memory_saver.resume()
        except ImportError:
            if self.model is not None:
                self.model.cuda()

    def set_rollout_manager(self, rollout_manager: Any) -> None:
        """Set rollout manager reference."""
        self.rollout_manager = rollout_manager

    def register_rollout_engines(self, engines: list) -> None:
        """Register rollout engines for weight updates."""
        self.rollout_engines = engines


# Ray actor wrapper
if HAS_RAY:
    @ray.remote(num_gpus=1)
    class MegatronTrainRayActorRemote(MegatronTrainRayActor):
        """Ray remote actor wrapper."""
        pass


def create_megatron_actors(
    args: Namespace,
    num_gpus: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
) -> list:
    """
    Create Megatron Ray actors.

    Args:
        args: Training arguments
        num_gpus: Number of GPUs to use
        master_addr: Master node address
        master_port: Master node port

    Returns:
        List of Ray actor handles
    """
    if not HAS_RAY:
        raise ImportError("Ray is required for Megatron actors")

    actors = []
    for rank in range(num_gpus):
        actor = MegatronTrainRayActorRemote.remote(
            world_size=num_gpus,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        actors.append(actor)

    return actors


def init_megatron_actors(
    actors: list,
    args: Namespace,
    role: str = "actor",
    with_ref: bool = False,
) -> list:
    """
    Initialize all Megatron actors.

    Args:
        actors: List of Ray actor handles
        args: Training arguments
        role: "actor" or "critic"
        with_ref: Whether to load reference model

    Returns:
        List of starting rollout IDs
    """
    if not HAS_RAY:
        raise ImportError("Ray is required for Megatron actors")

    futures = [
        actor.init.remote(args, role, with_ref)
        for actor in actors
    ]
    return ray.get(futures)
