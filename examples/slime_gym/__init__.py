"""
slime_gym - Gym-style environment abstraction for SLIME RL training.

Usage:
    python -m slime.train \
        --custom-generate-function-path examples.slime_gym.generate_with_gym.generate \
        --custom-rm-path examples.slime_gym.generate_with_gym.reward_func \
        ...
"""

# Base environment
from .base import BaseEnvironment, parse_tool_calls, tool

# Configuration
from .config import DYNAMIC_MAX_TURNS, MAX_TURNS, MAX_TURNS_BUFFER, resolve_max_turns
from .dynamic_env import DynamicServiceEnvironment

# Environment registry
from .env_registry import EnvironmentRegistry, resolve_env_name

# Formatters
from .formatters import ChatMLFormatter, get_formatter

# Environments (import to trigger registration)
from .retail_env import RetailServiceEnvironment

# Types
from .types import ExecutionState, ToolCall, ToolDefinition, ToolResult, append_to_sample, init_sample_for_generation

# Dynamic tools (advanced - for DynamicServiceEnvironment example)
# Import directly from tool_registry if needed:
#   from examples.slime_gym.tool_registry import ToolRegistry, DynamicToolMixin


# Generate helpers (optional - requires SLIME dependencies)
try:
    from .generate_with_gym import GenerateContext, setup_generate

    _HAS_GENERATE = True
except ImportError:
    GenerateContext = None  # type: ignore
    setup_generate = None  # type: ignore
    _HAS_GENERATE = False

__all__ = [
    # Types
    "ToolCall",
    "ToolResult",
    "ToolDefinition",
    "ExecutionState",
    "append_to_sample",
    "init_sample_for_generation",
    # Base
    "BaseEnvironment",
    "tool",
    "parse_tool_calls",
    # Config
    "MAX_TURNS",
    "MAX_TURNS_BUFFER",
    "DYNAMIC_MAX_TURNS",
    "resolve_max_turns",
    # Registry
    "EnvironmentRegistry",
    "resolve_env_name",
    # Formatters
    "ChatMLFormatter",
    "get_formatter",
    # Environments
    "RetailServiceEnvironment",
    "DynamicServiceEnvironment",
    # Generate helpers
    "GenerateContext",
    "setup_generate",
]
