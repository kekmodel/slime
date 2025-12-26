"""
Base environment class - SLIME standard compatible with Gym-style interface.

Provides two levels of abstraction:
1. Low-level: execute_tool() for individual tool calls
2. High-level: run_episode() for complete agent-environment interaction

Supports per-sample environment and tool customization:
- Environment selection via sample.metadata["env_name"]
- Dynamic tool filtering via sample.metadata["enabled_tools"]
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from slime.utils.types import Sample

from .gym_types import ToolCall, ToolResult


@dataclass
class EpisodeResult:
    """
    Result of a complete episode (Gym-style).
    Uses Sample.Status for SLIME standard compatibility.
    """

    response: str  # Full response text
    reward: float  # Final reward
    status: Sample.Status = Sample.Status.COMPLETED  # Reuse SLIME standard
    info: dict = field(default_factory=dict)  # Additional info

    # Training-specific fields (aligned with Sample)
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)  # Same as Sample
    num_turns: int = 0

    def to_sample(
        self,
        prompt: str | list[dict] = "",
        prompt_token_ids: list[int] | None = None,
    ) -> Sample:
        """
        Convert to SLIME Sample for training pipeline integration.

        Args:
            prompt: Original prompt (string or message list)
            prompt_token_ids: Tokenized prompt. If provided, will be prepended to tokens.

        Returns:
            Sample compatible with SLIME training pipeline
        """
        # Combine prompt tokens + response tokens (SLIME standard)
        if prompt_token_ids is not None:
            all_tokens = prompt_token_ids + self.tokens
        else:
            all_tokens = self.tokens

        return Sample(
            prompt=prompt,
            tokens=all_tokens,
            response=self.response,
            response_length=len(self.tokens),  # Only response part
            reward=self.reward,
            loss_mask=self.loss_mask,  # Only for response tokens
            rollout_log_probs=self.rollout_log_probs,
            status=self.status,
            metadata=self.info,
        )


def tool(
    description: str,
    parameters: dict[str, Any] | None = None,
    name: str | None = None,
) -> Callable:
    """
    Decorator to register a method as a tool.

    Usage:
        @tool(
            description="Calculate a math expression",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        )
        async def calculate(self, expression: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        func._tool_schema = {
            "type": "function",
            "function": {
                "name": name or func.__name__,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            },
        }

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._tool_schema = func._tool_schema
        return wrapper

    return decorator


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from response text"""
    tool_calls = []
    # Match content between tags, then parse as JSON
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            # Parse JSON directly without modifying the string
            data = json.loads(match.strip())
            tool_calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue

    return tool_calls


class BaseEnvironment(ABC):
    """
    Base class for all environments.
    SLIME standard: verify() takes Sample, not Trajectory.

    Subclasses should:
    1. Define tools using @tool decorator
    2. Implement verify() method for reward calculation
    3. Optionally override seed() for per-sample initialization

    Per-sample customization:
    - Override seed() to enable/disable tools based on metadata
    - Use self._enabled_tools to filter available tools
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: dict[str, dict] = {}  # Changed: name -> schema mapping
        self._enabled_tools: set[str] | None = None  # None = all enabled
        self._register_tools()

    def _register_tools(self) -> None:
        """Auto-register methods decorated with @tool"""
        for name in dir(self):
            if name.startswith("_"):
                continue
            method = getattr(self, name)
            if hasattr(method, "_tool_schema"):
                tool_name = method._tool_schema["function"]["name"]
                self._tools[tool_name] = method
                self._tool_schemas[tool_name] = method._tool_schema

    def get_tools(self) -> list[dict]:
        """
        Return tool schemas in OpenAI format (for prompt formatting).

        Respects self._enabled_tools filtering set in seed().
        If _enabled_tools is None, returns all tools.
        """
        if self._enabled_tools is None:
            return list(self._tool_schemas.values())

        return [schema for name, schema in self._tool_schemas.items() if name in self._enabled_tools]

    def get_all_tool_names(self) -> list[str]:
        """Return all registered tool names (regardless of enabled status)"""
        return list(self._tools.keys())

    def get_enabled_tool_names(self) -> list[str]:
        """Return currently enabled tool names"""
        if self._enabled_tools is None:
            return list(self._tools.keys())
        return list(self._enabled_tools)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name (respects enabled_tools filtering)"""
        # Check if tool exists
        if name not in self._tools:
            return ToolResult(
                output=f"Error: Unknown tool '{name}'. Available: {self.get_enabled_tool_names()}",
                success=False,
            )

        # Check if tool is enabled
        if self._enabled_tools is not None and name not in self._enabled_tools:
            return ToolResult(
                output=f"Error: Tool '{name}' is not available for this task. Available: {self.get_enabled_tool_names()}",
                success=False,
            )

        try:
            result = await self._tools[name](**arguments)
            return ToolResult(output=str(result), success=True)
        except Exception as e:
            return ToolResult(output=f"Error: {e}", success=False)

    def seed(self, metadata: dict) -> None:
        """
        Initialize environment for a specific sample.
        Override this to set up per-sample state and tool filtering.

        Example:
            def seed(self, metadata: dict) -> None:
                super().seed(metadata)  # Reset enabled_tools
                # Filter tools based on task type
                if metadata.get("task_type") == "read_only":
                    self._enabled_tools = {"get_info", "list_items"}
                # Or based on explicit list
                if "enabled_tools" in metadata:
                    self._enabled_tools = set(metadata["enabled_tools"])
        """
        # Reset to all tools enabled by default
        self._enabled_tools = None

        # Support explicit enabled_tools in metadata
        if "enabled_tools" in metadata:
            requested = set(metadata["enabled_tools"])
            available = set(self._tools.keys())
            self._enabled_tools = requested & available  # Intersection

    def reset(self) -> None:
        """Reset environment state. Override if needed."""
        self._enabled_tools = None

    @abstractmethod
    async def verify(self, sample: Sample) -> float:
        """
        Calculate reward for the sample.
        SLIME standard: parse tool calls from sample.response.

        Args:
            sample: The Sample with response text

        Returns:
            float: 1.0 for success, 0.0 for failure
        """

    # ==================== Gym-style Interface ====================

    async def run_episode(
        self,
        model_fn: Callable,
        initial_prompt: str | list[dict],
        max_turns: int = 10,
        tokenizer: Callable[[str], list[int]] | None = None,
    ) -> EpisodeResult:
        """
        Run a complete episode (Gym-style high-level interface).

        This is the main entry point for agent-environment interaction.
        Runs the full tool-calling loop until completion or max_turns.

        Args:
            model_fn: Async function that takes prompt and returns (text, token_ids, log_probs)
            initial_prompt: Starting prompt (string or message list)
            max_turns: Maximum number of tool-calling turns
            tokenizer: Optional tokenizer function for tool outputs (text -> token_ids).
                       Required for accurate token tracking during training.

        Returns:
            EpisodeResult containing response, reward, and training data
        """
        result = EpisodeResult(response="", reward=0.0, status=Sample.Status.PENDING)

        for _turn in range(max_turns):
            # Get model response
            cur_text, cur_tokens, cur_logprobs = await model_fn(result.response)

            # Record model output (trainable)
            result.response += cur_text
            result.tokens.extend(cur_tokens)
            result.loss_mask.extend([1] * len(cur_tokens))
            result.rollout_log_probs.extend(cur_logprobs)

            # Parse tool calls
            tool_calls = parse_tool_calls(cur_text)

            # No tool calls -> episode complete
            if not tool_calls:
                result.status = Sample.Status.COMPLETED
                break

            # Execute tools
            for tc in tool_calls:
                tool_result = await self.execute_tool(tc.name, tc.arguments)

                # Format tool output with tool name for clarity (not trainable)
                tool_text = f'\n<tool_result name="{tc.name}">\n{tool_result.output}\n</tool_result>\n'
                result.response += tool_text
                result.num_turns += 1

                # Track tool tokens if tokenizer provided
                if tokenizer is not None:
                    tool_token_ids = tokenizer(tool_text)
                    result.tokens.extend(tool_token_ids)
                    result.loss_mask.extend([0] * len(tool_token_ids))
                    result.rollout_log_probs.extend([0.0] * len(tool_token_ids))

        # Set TRUNCATED if max_turns reached without completion
        if result.status == Sample.Status.PENDING:
            result.status = Sample.Status.TRUNCATED

        # Create a temporary Sample for verification
        temp_sample = Sample(response=result.response)
        result.reward = await self.verify(temp_sample)
        result.info = {
            "num_turns": result.num_turns,
            "max_turns_reached": result.status == Sample.Status.TRUNCATED,
        }

        return result
