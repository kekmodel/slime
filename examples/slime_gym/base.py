"""
Base environment class for slime_gym.

Provides:
1. BaseEnvironment: Abstract base class with automatic tool tracking
2. @tool decorator: Register methods as callable tools
3. parse_tool_calls: Extract tool calls from response text
"""

import json
import re
from collections.abc import Callable
from functools import wraps
from typing import Any

from .types import ExecutionState, ToolCall, ToolResult


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
    """Extract tool calls from response text."""
    tool_calls = []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match.strip())
            tool_calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            continue

    return tool_calls


class BaseEnvironment:
    """
    Base class for all environments.

    Features:
    - Automatic tool registration via @tool decorator
    - Automatic execution tracking via ExecutionState
    - Per-sample tool filtering via enabled_tools

    Subclasses should:
    1. Define tools using @tool decorator
    2. Implement verify() method for reward calculation
    3. Optionally override seed() for per-sample initialization
    4. Optionally extend state with domain-specific fields
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: dict[str, dict] = {}
        self._enabled_tools: set[str] | None = None
        self.state: ExecutionState = ExecutionState()
        self.expected_actions: set[str] = set()
        self._register_tools()

    def _register_tools(self) -> None:
        """Auto-register methods decorated with @tool."""
        for name in dir(self):
            if name.startswith("_"):
                continue
            method = getattr(self, name)
            if hasattr(method, "_tool_schema"):
                tool_name = method._tool_schema["function"]["name"]
                self._tools[tool_name] = method
                self._tool_schemas[tool_name] = method._tool_schema

    def get_tools(self) -> list[dict]:
        """Return tool schemas in OpenAI format."""
        if self._enabled_tools is None:
            return list(self._tool_schemas.values())
        return [schema for name, schema in self._tool_schemas.items() if name in self._enabled_tools]

    def get_all_tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def get_enabled_tool_names(self) -> list[str]:
        """Return currently enabled tool names."""
        if self._enabled_tools is None:
            return list(self._tools.keys())
        return list(self._enabled_tools)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with automatic state tracking."""
        # Check if tool exists
        if name not in self._tools:
            return ToolResult(
                output=f"Error: Unknown tool '{name}'. Available: {self.get_enabled_tool_names()}",
                success=False,
            )

        # Check if tool is enabled
        if self._enabled_tools is not None and name not in self._enabled_tools:
            return ToolResult(
                output=f"Error: Tool '{name}' is not available. Available: {self.get_enabled_tool_names()}",
                success=False,
            )

        try:
            result = await self._tools[name](**arguments)
            # Automatic state tracking
            self.state.record_execution(name, result)
            return ToolResult(output=str(result), success=True)
        except Exception as e:
            return ToolResult(output=f"Error: {e}", success=False)

    def setup(self, metadata: dict) -> None:
        """
        Initialize environment for a specific sample.

        Override this to set up per-sample state and tool filtering.
        Always call super().setup(metadata) first.
        """
        # Reset state
        self.state = ExecutionState()
        self._enabled_tools = None
        self.expected_actions = set(metadata.get("expected_actions", []))

        # Support explicit enabled_tools in metadata
        if "enabled_tools" in metadata:
            requested = set(metadata["enabled_tools"])
            available = set(self._tools.keys())
            self._enabled_tools = requested & available

    def reset(self) -> None:
        """Reset environment state."""
        self.state = ExecutionState()
        self._enabled_tools = None
        self.expected_actions = set()

    def verify(self) -> float:
        """
        Calculate reward based on execution state.

        Default: 1.0 if all expected_actions executed, 0.0 otherwise.
        Override for custom verification logic.
        """
        return 1.0 if self.state.has_executed_all(self.expected_actions) else 0.0
