"""
Core data types for slime_gym environment abstraction.
Aligned with SLIME standard (Sample-based).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """Tool call information"""

    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Tool execution result"""

    output: str
    success: bool = True
