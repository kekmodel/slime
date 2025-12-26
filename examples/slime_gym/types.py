"""
Core type definitions for slime_gym.

Consolidates all dataclasses and type definitions in one place.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from slime.utils.types import Sample


# ==================== Sample Helpers ====================


def init_sample_for_generation(
    sample: Sample,
    prompt_text: str,
    prompt_token_ids: list[int],
) -> None:
    """Initialize Sample fields for generation."""
    sample.prompt = prompt_text
    sample.tokens = prompt_token_ids.copy()
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []


def append_to_sample(
    sample: Sample,
    text: str,
    token_ids: list[int],
    log_probs: list[float],
    trainable: bool = True,
) -> None:
    """
    Append tokens to sample with proper loss_mask handling.

    Args:
        sample: Sample to append to
        text: Text to append to response
        token_ids: Token IDs to append
        log_probs: Log probabilities for each token
        trainable: If True, loss_mask=1 (model output). If False, loss_mask=0 (tool output).
    """
    sample.tokens.extend(token_ids)
    sample.response += text
    sample.response_length += len(token_ids)
    sample.loss_mask.extend([1 if trainable else 0] * len(token_ids))
    sample.rollout_log_probs.extend(log_probs)


# ==================== Data Classes ====================


@dataclass
class ToolCall:
    """Parsed tool call from model response."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of tool execution."""

    output: str
    success: bool = True


@dataclass
class ToolDefinition:
    """Definition of a dynamically loadable tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: Callable[..., Awaitable[str]]
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    def to_schema(self) -> dict:
        """Convert to OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ExecutionState:
    """
    Generic state tracking for environment execution.

    Automatically tracks which tools were executed and their results.
    Environments can extend this with domain-specific fields.
    """

    executed_tools: set[str] = field(default_factory=set)
    tool_results: dict[str, Any] = field(default_factory=dict)
    submitted_result: Any = None

    def record_execution(self, tool_name: str, result: Any = None) -> None:
        """Record a tool execution."""
        self.executed_tools.add(tool_name)
        if result is not None:
            self.tool_results[tool_name] = result

    def has_executed(self, tool_name: str) -> bool:
        """Check if a tool was executed."""
        return tool_name in self.executed_tools

    def has_executed_all(self, tool_names: set[str]) -> bool:
        """Check if all specified tools were executed."""
        return tool_names.issubset(self.executed_tools)
