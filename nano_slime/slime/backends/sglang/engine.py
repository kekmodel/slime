"""
SGLang Engine Wrapper for nano_slime

This file wraps SGLang for inference/rollout generation.
Based on slime/backends/sglang_utils/sglang_engine.py.

Key features:
1. Initialize SGLang server
2. Generate rollouts (sampling)
3. Compute log_probs for generated sequences
4. Update weights from FSDP actor

Usage:
    engine = SGLangEngine(args)
    engine.init()
    samples = engine.generate(prompts)
    engine.update_weights(state_dict)
"""

import logging
import os
from argparse import Namespace
from typing import Any

try:
    import sglang as sgl
    from sglang import RuntimeEndpoint
    HAS_SGLANG = True
except ImportError:
    sgl = None
    RuntimeEndpoint = None
    HAS_SGLANG = False

import torch

from ...utils.types import Sample

logger = logging.getLogger(__name__)


class SGLangEngine:
    """
    SGLang engine for rollout generation.

    This wrapper provides:
    1. Text generation with sampling
    2. Log probability computation
    3. Weight synchronization from training
    """

    def __init__(self, args: Namespace, gpu_id: int = 0):
        """
        Initialize SGLang engine.

        Args:
            args: Configuration including:
                - hf_checkpoint: Model path
                - rollout_temperature: Sampling temperature
                - rollout_top_p: Top-p sampling
                - max_new_tokens: Maximum generation length
            gpu_id: GPU device ID
        """
        self.args = args
        self.gpu_id = gpu_id
        self.runtime = None
        self.weight_version = 0

    def init(self) -> None:
        """
        Initialize SGLang runtime.

        Starts a local SGLang server on the specified GPU.
        """
        if not HAS_SGLANG:
            raise ImportError(
                "SGLang is not installed. Install with: pip install sglang"
            )

        logger.info(f"[GPU {self.gpu_id}] Initializing SGLang engine")

        # Start SGLang runtime
        self.runtime = sgl.Runtime(
            model_path=self.args.hf_checkpoint,
            tp_size=getattr(self.args, "rollout_tp_size", 1),
            trust_remote_code=True,
            mem_fraction_static=getattr(self.args, "mem_fraction_static", 0.8),
        )

        logger.info(f"[GPU {self.gpu_id}] SGLang engine initialized")

    def generate(
        self,
        prompts: list[str],
        n_samples: int = 1,
    ) -> list[Sample]:
        """
        Generate responses for given prompts.

        Args:
            prompts: List of prompt strings
            n_samples: Number of samples per prompt (for GRPO)

        Returns:
            List of Sample objects containing:
                - tokens: Generated token ids
                - response_length: Length of response
                - rollout_log_probs: Log probs from generation
                - reward: To be filled by reward model
        """
        if self.runtime is None:
            raise RuntimeError("Engine not initialized. Call init() first.")

        logger.info(f"Generating {len(prompts)} prompts x {n_samples} samples")

        samples = []
        temperature = getattr(self.args, "rollout_temperature", 1.0)
        top_p = getattr(self.args, "rollout_top_p", 1.0)
        max_new_tokens = getattr(self.args, "max_new_tokens", 512)

        for prompt_idx, prompt in enumerate(prompts):
            # Generate n_samples responses for each prompt
            for sample_idx in range(n_samples):
                # Generate with SGLang
                output = self.runtime.generate(
                    prompt,
                    sampling_params={
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "return_logprob": True,
                        "logprob_start_len": 0,
                    },
                )

                # Extract tokens and log probs
                prompt_tokens = output.get("prompt_token_ids", [])
                response_tokens = output.get("token_ids", [])
                log_probs = output.get("token_logprobs", [])

                # Create Sample
                all_tokens = list(prompt_tokens) + list(response_tokens)
                sample = Sample(
                    tokens=torch.tensor(all_tokens, dtype=torch.long),
                    response_length=len(response_tokens),
                    total_length=len(all_tokens),
                    rollout_log_probs=torch.tensor(log_probs, dtype=torch.float32) if log_probs else None,
                    prompt_text=prompt,
                    response_text=output.get("text", ""),
                    group_index=prompt_idx,  # For GRPO grouping
                    sample_index=sample_idx,
                )
                samples.append(sample)

        logger.info(f"Generated {len(samples)} samples")
        return samples

    def generate_batch(
        self,
        prompts: list[str],
        n_samples: int = 1,
    ) -> list[Sample]:
        """
        Generate responses in batch mode.

        More efficient than generate() for large batches.

        Args:
            prompts: List of prompt strings
            n_samples: Number of samples per prompt

        Returns:
            List of Sample objects
        """
        if self.runtime is None:
            raise RuntimeError("Engine not initialized. Call init() first.")

        # Expand prompts for n_samples
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * n_samples)

        temperature = getattr(self.args, "rollout_temperature", 1.0)
        top_p = getattr(self.args, "rollout_top_p", 1.0)
        max_new_tokens = getattr(self.args, "max_new_tokens", 512)

        # Batch generate
        outputs = self.runtime.generate(
            expanded_prompts,
            sampling_params={
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )

        # Convert to Sample objects
        samples = []
        for i, output in enumerate(outputs):
            prompt_idx = i // n_samples
            sample_idx = i % n_samples

            prompt_tokens = output.get("prompt_token_ids", [])
            response_tokens = output.get("token_ids", [])
            log_probs = output.get("token_logprobs", [])

            all_tokens = list(prompt_tokens) + list(response_tokens)
            sample = Sample(
                tokens=torch.tensor(all_tokens, dtype=torch.long),
                response_length=len(response_tokens),
                total_length=len(all_tokens),
                rollout_log_probs=torch.tensor(log_probs, dtype=torch.float32) if log_probs else None,
                prompt_text=expanded_prompts[i],
                response_text=output.get("text", ""),
                group_index=prompt_idx,
                sample_index=sample_idx,
            )
            samples.append(sample)

        return samples

    def compute_log_probs(
        self,
        tokens: list[torch.Tensor],
        response_lengths: list[int],
    ) -> list[torch.Tensor]:
        """
        Compute log probabilities for given token sequences.

        Used for computing ref_log_probs or re-computing actor log_probs.

        Args:
            tokens: List of token tensors
            response_lengths: Response length for each sequence

        Returns:
            List of log_prob tensors
        """
        if self.runtime is None:
            raise RuntimeError("Engine not initialized. Call init() first.")

        log_probs_list = []

        for token_seq, response_len in zip(tokens, response_lengths):
            # Convert to list for SGLang
            token_list = token_seq.tolist()

            # Get log probs from SGLang
            output = self.runtime.get_logprobs(
                token_ids=token_list,
                logprob_start_len=len(token_list) - response_len - 1,
            )

            log_probs = torch.tensor(
                output.get("token_logprobs", []),
                dtype=torch.float32
            )
            log_probs_list.append(log_probs)

        return log_probs_list

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Update model weights from state dict.

        Called after training to sync weights from FSDP actor.

        Args:
            state_dict: Model state dict from training
        """
        if self.runtime is None:
            raise RuntimeError("Engine not initialized. Call init() first.")

        logger.info(f"[GPU {self.gpu_id}] Updating weights (version {self.weight_version + 1})")

        # SGLang weight update API
        self.runtime.update_weights(state_dict)
        self.weight_version += 1

        logger.info(f"[GPU {self.gpu_id}] Weights updated to version {self.weight_version}")

    def get_weight_version(self) -> int:
        """Get current weight version."""
        return self.weight_version

    def release_memory_occupation(self) -> None:
        """Release GPU memory (for colocate mode)."""
        if self.runtime is not None:
            self.runtime.release_memory()

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        """Resume GPU memory occupation."""
        if self.runtime is not None:
            self.runtime.resume_memory(tags=tags)

    def shutdown(self) -> None:
        """Shutdown the engine."""
        if self.runtime is not None:
            self.runtime.shutdown()
            self.runtime = None
            logger.info(f"[GPU {self.gpu_id}] SGLang engine shutdown")


class MockSGLangEngine:
    """
    Mock SGLang engine for testing without GPU.

    Generates random tokens and log probs.
    """

    def __init__(self, args: Namespace, gpu_id: int = 0):
        self.args = args
        self.gpu_id = gpu_id
        self.weight_version = 0
        self.vocab_size = getattr(args, "vocab_size", 32000)

    def init(self) -> None:
        logger.info(f"[Mock GPU {self.gpu_id}] Mock SGLang engine initialized")

    def generate(
        self,
        prompts: list[str],
        n_samples: int = 1,
    ) -> list[Sample]:
        """Generate mock samples."""
        import random

        samples = []
        max_new_tokens = getattr(self.args, "max_new_tokens", 128)

        for prompt_idx, prompt in enumerate(prompts):
            for sample_idx in range(n_samples):
                # Random response length
                response_len = random.randint(10, max_new_tokens)

                # Random tokens
                prompt_tokens = [random.randint(0, self.vocab_size - 1) for _ in range(50)]
                response_tokens = [random.randint(0, self.vocab_size - 1) for _ in range(response_len)]
                all_tokens = prompt_tokens + response_tokens

                # Random log probs
                log_probs = torch.randn(response_len) - 5.0  # Negative log probs

                sample = Sample(
                    tokens=torch.tensor(all_tokens, dtype=torch.long),
                    response_length=response_len,
                    total_length=len(all_tokens),
                    rollout_log_probs=log_probs,
                    prompt_text=prompt,
                    response_text=f"<mock response {sample_idx}>",
                    group_index=prompt_idx,
                    sample_index=sample_idx,
                )
                samples.append(sample)

        return samples

    def generate_batch(
        self,
        prompts: list[str],
        n_samples: int = 1,
    ) -> list[Sample]:
        return self.generate(prompts, n_samples)

    def compute_log_probs(
        self,
        tokens: list[torch.Tensor],
        response_lengths: list[int],
    ) -> list[torch.Tensor]:
        """Generate mock log probs."""
        return [torch.randn(rl) - 5.0 for rl in response_lengths]

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.weight_version += 1
        logger.info(f"[Mock GPU {self.gpu_id}] Mock weights updated to version {self.weight_version}")

    def get_weight_version(self) -> int:
        return self.weight_version

    def release_memory_occupation(self) -> None:
        pass

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        pass

    def shutdown(self) -> None:
        logger.info(f"[Mock GPU {self.gpu_id}] Mock engine shutdown")
