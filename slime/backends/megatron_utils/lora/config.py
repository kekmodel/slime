"""LoRA configuration for Megatron backend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

ALL_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "expert"]
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 0
    alpha: float = 0.0
    target_modules: tuple[str, ...] = ()
    dropout: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @property
    def scaling(self) -> float:
        if self.rank == 0:
            return 0.0
        return self.alpha / self.rank

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LoRAConfig:
        rank = getattr(args, "lora_rank", 0)
        alpha = getattr(args, "lora_alpha", None)
        if alpha is None and rank > 0:
            alpha = 2 * rank
        elif alpha is None:
            alpha = 0.0
        target_modules = tuple(getattr(args, "lora_target_modules", []))
        dropout = getattr(args, "lora_dropout", 0.0)

        if "expert" in target_modules and not getattr(args, "num_experts", None):
            raise ValueError("'expert' in lora_target_modules requires --num-experts > 0")

        return cls(rank=rank, alpha=alpha, target_modules=target_modules, dropout=dropout)


def add_lora_args(parser: argparse._ActionsContainer) -> None:
    """Register LoRA-specific CLI arguments."""
    group = parser.add_argument_group(title="LoRA", description="Low-Rank Adaptation arguments")
    group.add_argument("--lora-rank", type=int, default=0, help="LoRA rank. 0 disables LoRA.")
    group.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha for scaling. Default: 2 * rank.")
    group.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        choices=ALL_TARGET_MODULES,
        help="Target modules for LoRA. Default: all except expert.",
    )
    group.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout rate.")
    group.add_argument("--adapter-load", type=str, default=None, help="Path to load LoRA adapter checkpoint.")
    group.add_argument(
        "--save-adapter-only",
        action="store_true",
        default=None,
        dest="save_adapter_only",
        help="Save only adapter weights in checkpoint. Default: True when --lora-rank > 0.",
    )
    group.add_argument(
        "--no-save-adapter-only",
        action="store_false",
        dest="save_adapter_only",
        help="Save full checkpoint even when LoRA is enabled.",
    )
