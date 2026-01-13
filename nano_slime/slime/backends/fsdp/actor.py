"""
FSDP Actor for nano_slime

This file implements a minimal FSDP-based training actor.
Based on slime/backends/fsdp_utils/actor.py.

Key components:
1. HuggingFace model loading
2. FSDP2 wrapping for distributed training
3. Forward/backward pass
4. Weight synchronization with SGLang engines

Usage:
    actor = FSDPActor(args)
    actor.init()
    actor.train(rollout_id, rollout_data)
    actor.update_weights()
"""

import logging
import os
from argparse import Namespace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..training_utils.loss import compute_advantages_and_returns, policy_loss_function
from .parallel import ParallelState, create_parallel_state

logger = logging.getLogger(__name__)


class FSDPActor:
    """
    Minimal FSDP training actor for GRPO.

    Responsibilities:
    1. Load HuggingFace model
    2. Wrap with FSDP for distributed training
    3. Compute log_probs for actor and ref model
    4. Run policy gradient training
    5. Sync weights to rollout engines
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.model = None
        self.ref_model = None
        self.optimizer = None
        self.parallel_state = None
        self.rollout_engines = []

    def init(self) -> None:
        """
        Initialize FSDP actor.

        Steps:
        1. Initialize distributed
        2. Create parallel state
        3. Load model
        4. Wrap with FSDP
        5. Create optimizer
        """
        # Initialize distributed if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.cuda.set_device(self.device)

        logger.info(f"[Rank {self.rank}] Initializing FSDP Actor")

        # Create parallel state
        self.parallel_state = create_parallel_state(self.args)

        # Load config and tokenizer
        self.config = AutoConfig.from_pretrained(
            self.args.hf_checkpoint,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.hf_checkpoint,
            trust_remote_code=True
        )

        # Load and wrap model
        self.model = self._load_and_wrap_model(self.args.hf_checkpoint)

        # Load ref model if needed (for KL computation)
        if getattr(self.args, "use_ref_model", True):
            ref_path = getattr(self.args, "ref_checkpoint", self.args.hf_checkpoint)
            self.ref_model = self._load_and_wrap_model(ref_path, is_ref=True)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(getattr(self.args, "adam_beta1", 0.9),
                   getattr(self.args, "adam_beta2", 0.999)),
            eps=getattr(self.args, "adam_eps", 1e-8),
            weight_decay=getattr(self.args, "weight_decay", 0.01),
        )

        self.global_step = 0
        logger.info(f"[Rank {self.rank}] FSDP Actor initialized")

    def _load_and_wrap_model(
        self,
        checkpoint_path: str,
        is_ref: bool = False
    ) -> FSDP:
        """
        Load HuggingFace model and wrap with FSDP.

        Args:
            checkpoint_path: Path to HF checkpoint
            is_ref: If True, this is a reference model (frozen)

        Returns:
            FSDP-wrapped model
        """
        logger.info(f"[Rank {self.rank}] Loading model from {checkpoint_path}")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=getattr(self.args, "attn_implementation", "flash_attention_2"),
        )

        if is_ref:
            # Freeze ref model
            for param in model.parameters():
                param.requires_grad = False

        # FSDP configuration
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

        # Get transformer layers for auto_wrap
        layer_cls = model._no_split_modules if hasattr(model, "_no_split_modules") else None

        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=self.device,
            use_orig_params=True,
        )

        if not is_ref:
            fsdp_model.train()
        else:
            fsdp_model.eval()

        logger.info(f"[Rank {self.rank}] Model wrapped with FSDP")
        return fsdp_model

    def compute_log_probs(
        self,
        model: FSDP,
        tokens: torch.Tensor,
        response_lengths: list[int],
    ) -> list[torch.Tensor]:
        """
        Compute log probabilities for response tokens.

        Args:
            model: FSDP model
            tokens: Input token ids [batch, seq_len]
            response_lengths: Length of response for each sample

        Returns:
            List of log_probs tensors, one per sample
        """
        with torch.no_grad() if not model.training else torch.enable_grad():
            # Forward pass
            outputs = model(input_ids=tokens)
            logits = outputs.logits.float()  # [batch, seq_len, vocab]

            # Apply temperature
            temperature = getattr(self.args, "rollout_temperature", 1.0)
            logits = logits / temperature

            # Compute log probs
            log_probs_all = torch.log_softmax(logits, dim=-1)

            # Extract response log probs for each sample
            batch_size = tokens.size(0)
            log_probs_list = []

            for i in range(batch_size):
                response_len = response_lengths[i]
                # Get indices for response tokens
                # log_probs for token t comes from logits at position t-1
                start_idx = tokens.size(1) - response_len

                # Gather log probs for actual tokens
                token_indices = tokens[i, start_idx:]  # [response_len]
                logits_slice = log_probs_all[i, start_idx-1:-1]  # [response_len, vocab]

                # Gather actual token log probs
                log_probs = logits_slice.gather(
                    dim=-1,
                    index=token_indices.unsqueeze(-1)
                ).squeeze(-1)  # [response_len]

                log_probs_list.append(log_probs)

            return log_probs_list

    def train(self, rollout_id: int, rollout_data: dict) -> dict:
        """
        Run one training iteration.

        Args:
            rollout_id: Current rollout iteration
            rollout_data: Dict containing:
                - tokens: list of token tensors
                - rewards: list of rewards
                - response_lengths: list of response lengths
                - loss_masks: list of loss masks
                - rollout_log_probs (optional): log_probs from rollout

        Returns:
            Training metrics dict
        """
        logger.info(f"[Rank {self.rank}] Training rollout {rollout_id}")

        # Get data from rollout
        tokens = rollout_data["tokens"]
        rewards = rollout_data["rewards"]
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data.get("loss_masks", None)

        # Stack tokens for batch processing
        if isinstance(tokens, list):
            # Pad to same length
            max_len = max(len(t) for t in tokens)
            padded_tokens = []
            for t in tokens:
                if len(t) < max_len:
                    padding = torch.zeros(max_len - len(t), dtype=t.dtype, device=t.device)
                    t = torch.cat([t, padding])
                padded_tokens.append(t)
            tokens_tensor = torch.stack(padded_tokens).to(self.device)
        else:
            tokens_tensor = tokens.to(self.device)

        # Compute ref log probs
        if self.ref_model is not None:
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(
                    self.ref_model,
                    tokens_tensor,
                    response_lengths
                )
        else:
            ref_log_probs = None

        # Compute actor log probs (for old policy)
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(
                self.model,
                tokens_tensor,
                response_lengths
            )

        # Create batch for loss function
        batch = {
            "tokens": tokens_tensor,
            "unconcat_tokens": [tokens_tensor[i] for i in range(tokens_tensor.size(0))],
            "rewards": rewards,
            "response_lengths": response_lengths,
            "total_lengths": [len(t) for t in tokens] if isinstance(tokens, list) else [tokens_tensor.size(1)] * tokens_tensor.size(0),
            "log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "loss_masks": loss_masks or [torch.ones(rl, device=self.device) for rl in response_lengths],
        }

        # Add rollout_log_probs if using off-policy
        if self.args.use_rollout_logprobs and "rollout_log_probs" in rollout_data:
            batch["rollout_log_probs"] = rollout_data["rollout_log_probs"]

        # Compute advantages and returns (GRPO)
        compute_advantages_and_returns(self.args, self.parallel_state, batch)

        # Training step
        self.optimizer.zero_grad()

        # Forward pass for current policy
        outputs = self.model(input_ids=tokens_tensor)
        logits = outputs.logits.float()

        # Compute policy loss
        loss, metrics = policy_loss_function(
            args=self.args,
            parallel_state=self.parallel_state,
            batch=batch,
            logits=logits,
            sum_of_sample_mean=lambda x: x.mean(),
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if hasattr(self.args, "clip_grad"):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.clip_grad
            )
        else:
            grad_norm = 0.0

        # Optimizer step
        self.optimizer.step()
        self.global_step += 1

        # Prepare return metrics
        return_metrics = {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "train/global_step": self.global_step,
        }
        for k, v in metrics.items():
            return_metrics[f"train/{k}"] = v.item() if torch.is_tensor(v) else v

        logger.info(f"[Rank {self.rank}] Loss: {loss.item():.4f}")
        return return_metrics

    def update_weights(self) -> None:
        """
        Synchronize actor weights to rollout engines.

        This is called after training to update SGLang engines
        with the new model weights.
        """
        if not self.rollout_engines:
            logger.warning("No rollout engines to update")
            return

        logger.info(f"[Rank {self.rank}] Updating weights to rollout engines")

        # Gather full state dict (only on rank 0)
        # For FSDP, we use state_dict with full precision
        with FSDP.state_dict_type(
            self.model,
            state_dict_type=torch.distributed.fsdp.StateDictType.FULL_STATE_DICT,
        ):
            if self.rank == 0:
                state_dict = self.model.state_dict()

                # Update each rollout engine
                for engine in self.rollout_engines:
                    try:
                        engine.update_weights(state_dict)
                    except Exception as e:
                        logger.error(f"Failed to update engine: {e}")

        # Sync all ranks
        dist.barrier()
        logger.info(f"[Rank {self.rank}] Weights updated")

    def register_rollout_engines(self, engines: list) -> None:
        """Register rollout engines for weight updates."""
        self.rollout_engines = engines

    def sleep(self) -> None:
        """Offload model to CPU (for colocate mode)."""
        if getattr(self.args, "offload_train", False):
            self.model.cpu()
            torch.cuda.empty_cache()

    def wake_up(self) -> None:
        """Move model back to GPU."""
        if getattr(self.args, "offload_train", False):
            self.model.cuda()
