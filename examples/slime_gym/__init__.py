"""
slime_gym - NeMo Gym style environment abstraction for SLIME.

This module provides a clean, extensible interface for building
agentic tool-calling environments for RL training.

Features:
- SLIME standard compatible (Sample-based)
- Gym-style high-level interface (run_episode, generate)
- Per-sample environment and tool customization
- Dynamic tool loading from metadata

Usage:
    python -m slime.train \
        --custom-generate-function-path examples.slime_gym.generate_with_gym.generate \
        --custom-rm-path examples.slime_gym.generate_with_gym.reward_func \
        ...
"""

from .base import BaseEnvironment, EpisodeResult, parse_tool_calls, tool
from .dynamic_env import DynamicServiceEnvironment
from .gym_types import ToolCall, ToolResult
from .retail_env import RetailServiceEnvironment
from .tool_registry import DynamicToolMixin, ToolDefinition, ToolRegistry, get_registry, register_tool

__all__ = [
    # Types
    "ToolCall",
    "ToolResult",
    "ToolDefinition",
    "EpisodeResult",
    # Base
    "BaseEnvironment",
    "tool",
    "parse_tool_calls",
    # Dynamic tools
    "ToolRegistry",
    "DynamicToolMixin",
    "get_registry",
    "register_tool",
    # Environments
    "RetailServiceEnvironment",
    "DynamicServiceEnvironment",
]
