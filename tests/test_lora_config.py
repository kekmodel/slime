"""Tests for LoRA configuration."""
import argparse

import pytest


def test_lora_config_from_args():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=64,
        lora_alpha=128,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )
    config = LoRAConfig.from_args(args)
    assert config.rank == 64
    assert config.alpha == 128
    assert config.scaling == 128 / 64
    assert "q_proj" in config.target_modules
    assert config.dropout == 0.05


def test_lora_config_disabled():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=0,
        lora_alpha=0,
        lora_target_modules=[],
        lora_dropout=0.0,
    )
    config = LoRAConfig.from_args(args)
    assert not config.enabled


def test_lora_config_default_alpha():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=32,
        lora_alpha=None,
        lora_target_modules=["q_proj"],
        lora_dropout=0.0,
    )
    config = LoRAConfig.from_args(args)
    assert config.alpha == 64  # default: 2 * rank


def test_lora_config_expert_requires_num_experts():
    from slime.backends.megatron_utils.lora.config import LoRAConfig

    args = argparse.Namespace(
        lora_rank=64,
        lora_alpha=128,
        lora_target_modules=["expert"],
        lora_dropout=0.0,
        num_experts=None,
    )
    with pytest.raises(ValueError, match="expert"):
        LoRAConfig.from_args(args)
