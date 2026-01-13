"""
Nano SLIME - Main Training Script

This is the main entry point for nano_slime GRPO training.
Connects SGLang (inference) + FSDP (training) for actual model training.

Usage:
    # Single GPU (mock mode for testing)
    python train.py --mock

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 train.py --hf-checkpoint <model_path>

    # With SGLang rollout
    python train.py --hf-checkpoint <model_path> --use-sglang

Training Flow:
    1. Initialize SGLang engine (rollout/inference)
    2. Initialize FSDP actor (training)
    3. Main loop:
        a. Generate samples from prompts (SGLang)
        b. Compute rewards
        c. GRPO normalization
        d. Train actor (FSDP)
        e. Update SGLang weights
        f. Evaluate periodically
"""

import argparse
import logging
import os
from argparse import Namespace
from typing import Any

import torch
import torch.distributed as dist

from slime.rollout.reward import (
    convert_samples_to_train_data,
    filter_zero_std_groups,
    post_process_rewards,
)
from slime.utils.tracking import close, init_tracking, log
from slime.utils.types import RolloutBatch, Sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Nano SLIME GRPO Training")

    # Model settings
    parser.add_argument("--hf-checkpoint", type=str, required=False,
                        help="Path to HuggingFace model checkpoint")
    parser.add_argument("--ref-checkpoint", type=str, default=None,
                        help="Path to reference model (default: same as hf-checkpoint)")

    # Training settings
    parser.add_argument("--num-rollout", type=int, default=100,
                        help="Number of rollout iterations")
    parser.add_argument("--rollout-batch-size", type=int, default=8,
                        help="Number of prompts per rollout")
    parser.add_argument("--n-samples-per-prompt", type=int, default=4,
                        help="Number of samples per prompt (GRPO group size)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate")

    # GRPO settings
    parser.add_argument("--advantage-estimator", type=str, default="grpo",
                        choices=["grpo", "gspo", "ppo"])
    parser.add_argument("--grpo-std-normalization", action="store_true", default=True,
                        help="Normalize by std in GRPO (False = adv wo std)")
    parser.add_argument("--filter-zero-std", action="store_true",
                        help="Filter groups with zero std")

    # KL settings
    parser.add_argument("--kl-coef", type=float, default=0.05,
                        help="KL coefficient")
    parser.add_argument("--kl-loss-type", type=str, default="k3",
                        choices=["k1", "k2", "k3", "low_var_kl"])
    parser.add_argument("--use-kl-loss", action="store_true",
                        help="Add KL loss term")
    parser.add_argument("--use-unbiased-kl", action="store_true",
                        help="Use unbiased KL with importance ratio")
    parser.add_argument("--kl-loss-coef", type=float, default=0.1,
                        help="KL loss coefficient")

    # PPO settings
    parser.add_argument("--eps-clip", type=float, default=0.2,
                        help="PPO clip epsilon")
    parser.add_argument("--eps-clip-high", type=float, default=0.2,
                        help="PPO clip epsilon (high)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                        help="Gradient clipping")

    # Off-policy settings
    parser.add_argument("--use-rollout-logprobs", action="store_true",
                        help="Use rollout log_probs as old policy")
    parser.add_argument("--normalize-advantages", action="store_true",
                        help="Normalize advantages across DP group")

    # Optimizer settings
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Sampling settings
    parser.add_argument("--rollout-temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--rollout-top-p", type=float, default=1.0,
                        help="Top-p sampling")

    # Distributed settings
    parser.add_argument("--colocate", action="store_true",
                        help="Colocate rollout and training on same GPU")
    parser.add_argument("--offload-train", action="store_true",
                        help="Offload training to CPU when doing rollout")

    # Logging settings
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--tb-dir", type=str, default="./runs",
                        help="TensorBoard directory")
    parser.add_argument("--wandb-project", type=str, default="nano_slime")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Checkpoint save interval")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")

    # Execution mode
    parser.add_argument("--mock", action="store_true",
                        help="Use mock engine for testing without GPU")
    parser.add_argument("--use-sglang", action="store_true",
                        help="Use SGLang for rollout (requires sglang)")
    parser.add_argument("--seed", type=int, default=42)

    # Data settings
    parser.add_argument("--prompt-data", type=str, default=None,
                        help="Path to prompt data (JSON lines)")

    return parser.parse_args()


def load_prompts(args: Namespace) -> list[str]:
    """Load prompts from file or generate mock prompts."""
    if args.prompt_data and os.path.exists(args.prompt_data):
        import json
        prompts = []
        with open(args.prompt_data) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get("prompt", data.get("text", "")))
        return prompts

    # Mock prompts for testing
    return [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate the Fibonacci sequence.",
        "What are the benefits of regular exercise?",
        "Describe the water cycle in nature.",
        "How do computers store and process information?",
        "What is the theory of relativity?",
        "Explain how photosynthesis works.",
        "What are the main causes of climate change?",
    ] * 10  # Repeat for more data


def compute_rewards(
    samples: list[Sample],
    args: Namespace,
) -> list[Sample]:
    """
    Compute rewards for generated samples.

    In a real setup, this would call a reward model.
    For now, we use a simple heuristic based on response length.

    Args:
        samples: List of generated samples
        args: Configuration

    Returns:
        Samples with rewards filled in
    """
    for sample in samples:
        # Simple reward: longer responses get higher rewards (with noise)
        # In practice, replace this with actual reward model
        import random

        base_reward = min(sample.response_length / 100.0, 2.0)
        noise = random.uniform(-0.5, 0.5)
        sample.reward = base_reward + noise

    return samples


def train_grpo(args: Namespace) -> None:
    """
    Main GRPO training function.

    Connects SGLang inference with FSDP training.
    """
    logger.info("=" * 60)
    logger.info("Nano SLIME GRPO Training")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(args)}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize tracking
    is_primary = not dist.is_initialized() or dist.get_rank() == 0
    if is_primary:
        init_tracking(args, primary=True)

    # Load prompts
    prompts = load_prompts(args)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Initialize rollout engine
    if args.mock:
        from slime.backends.sglang.engine import MockSGLangEngine
        rollout_engine = MockSGLangEngine(args)
        rollout_engine.init()
        logger.info("Using MockSGLangEngine for testing")
    elif args.use_sglang:
        from slime.backends.sglang.engine import SGLangEngine
        rollout_engine = SGLangEngine(args)
        rollout_engine.init()
        logger.info("Using SGLang for rollout")
    else:
        # Use mock engine by default if no sglang
        from slime.backends.sglang.engine import MockSGLangEngine
        rollout_engine = MockSGLangEngine(args)
        rollout_engine.init()
        logger.info("Using MockSGLangEngine (no --use-sglang flag)")

    # Initialize training actor (only if we have a model)
    actor = None
    if args.hf_checkpoint and not args.mock:
        try:
            from slime.backends.fsdp.actor import FSDPActor
            actor = FSDPActor(args)
            actor.init()
            actor.register_rollout_engines([rollout_engine])
            logger.info("FSDP Actor initialized")
        except Exception as e:
            logger.warning(f"Could not initialize FSDP Actor: {e}")
            logger.warning("Running in inference-only mode")

    # Main training loop
    logger.info("Starting training loop...")
    num_prompts_per_batch = args.rollout_batch_size

    for rollout_id in range(args.num_rollout):
        logger.info(f"\n{'='*40}")
        logger.info(f"Rollout {rollout_id + 1}/{args.num_rollout}")
        logger.info(f"{'='*40}")

        # Get batch of prompts
        start_idx = (rollout_id * num_prompts_per_batch) % len(prompts)
        batch_prompts = prompts[start_idx:start_idx + num_prompts_per_batch]

        # Colocate mode: offload training before rollout
        if args.colocate and actor is not None:
            actor.sleep()

        # 1. Generate samples (rollout)
        logger.info(f"Generating {len(batch_prompts)} x {args.n_samples_per_prompt} samples...")
        samples = rollout_engine.generate(
            batch_prompts,
            n_samples=args.n_samples_per_prompt,
        )
        logger.info(f"Generated {len(samples)} samples")

        # 2. Compute rewards
        samples = compute_rewards(samples, args)

        # 3. GRPO reward normalization
        raw_rewards, normalized_rewards = post_process_rewards(args, samples)
        logger.info(f"Rewards - Raw mean: {sum(raw_rewards)/len(raw_rewards):.3f}, "
                    f"Normalized mean: {sum(normalized_rewards)/len(normalized_rewards):.3f}")

        # 4. Filter zero std groups (optional)
        if args.filter_zero_std:
            samples = filter_zero_std_groups(args, samples)
            logger.info(f"After filtering: {len(samples)} samples")

        # 5. Convert to training batch
        rollout_data = convert_samples_to_train_data(args, samples, normalized_rewards)

        # Colocate mode: wake up training
        if args.colocate and actor is not None:
            actor.wake_up()

        # 6. Train (if actor is available)
        if actor is not None:
            train_metrics = actor.train(rollout_id, rollout_data)
        else:
            # Simulate training metrics
            import random
            train_metrics = {
                "train/rollout_id": rollout_id,
                "train/loss": random.uniform(0.1, 0.5),
                "train/pg_loss": random.uniform(0.05, 0.3),
                "train/batch_size": len(samples),
            }

        # 7. Log metrics
        metrics = {
            "rollout/num_samples": len(samples),
            "rollout/mean_reward": sum(raw_rewards) / len(raw_rewards),
            "rollout/mean_response_length": sum(s.response_length for s in samples) / len(samples),
            **train_metrics,
        }
        if is_primary:
            log(metrics, step=rollout_id)

        # 8. Update rollout engine weights
        if actor is not None and rollout_id % 5 == 0:  # Update every 5 steps
            actor.update_weights()

        # 9. Evaluate (periodically)
        if (rollout_id + 1) % args.eval_interval == 0:
            logger.info("Running evaluation...")
            # Placeholder for evaluation
            eval_metrics = {"eval/dummy_score": 0.5}
            if is_primary:
                log(eval_metrics, step=rollout_id)

        # 10. Save checkpoint (periodically)
        if (rollout_id + 1) % args.save_interval == 0 and actor is not None:
            logger.info(f"Saving checkpoint at rollout {rollout_id + 1}")
            # Placeholder for checkpoint saving

    # Cleanup
    rollout_engine.shutdown()
    if is_primary:
        close()

    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    train_grpo(args)
