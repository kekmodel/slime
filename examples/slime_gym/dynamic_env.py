"""
Dynamic Environment - Environment with runtime tool loading.

Demonstrates:
1. Loading tool implementations from metadata
2. Mixing static and dynamic tools
3. Using tool providers for modular organization
"""

from dataclasses import dataclass, field
from typing import Any

from .base import BaseEnvironment, tool
from .env_registry import EnvironmentRegistry
from .tool_registry import DynamicToolMixin, ToolDefinition, ToolRegistry, get_registry
from .types import ExecutionState, ToolResult

# ==================== Tool Providers ====================


class PaymentToolProvider:
    """Tool provider for payment-related tools."""

    @staticmethod
    def get_tools() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="process_payment",
                description="Process a payment transaction",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "Payment amount"},
                        "method": {"type": "string", "description": "Payment method"},
                    },
                    "required": ["amount", "method"],
                },
                implementation=PaymentToolProvider._process_payment,
                tags=["payment"],
            ),
            ToolDefinition(
                name="refund_payment",
                description="Process a refund",
                parameters={
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string", "description": "Transaction ID"},
                        "amount": {"type": "number", "description": "Refund amount"},
                    },
                    "required": ["transaction_id"],
                },
                implementation=PaymentToolProvider._refund_payment,
                tags=["payment", "refund"],
            ),
        ]

    @staticmethod
    async def _process_payment(amount: float, method: str) -> str:
        return f"Payment of ${amount:.2f} via {method} processed. Transaction ID: TXN-12345"

    @staticmethod
    async def _refund_payment(transaction_id: str, amount: float | None = None) -> str:
        amount_str = f"${amount:.2f}" if amount else "full amount"
        return f"Refund of {amount_str} for transaction {transaction_id} processed."


class AnalyticsToolProvider:
    """Tool provider for analytics tools."""

    @staticmethod
    def get_tools() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_metrics",
                description="Get analytics metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string", "description": "Metric name"},
                        "time_range": {"type": "string", "description": "Time range"},
                    },
                    "required": ["metric_name"],
                },
                implementation=AnalyticsToolProvider._get_metrics,
                tags=["analytics"],
            ),
        ]

    @staticmethod
    async def _get_metrics(metric_name: str, time_range: str = "7d") -> str:
        return f"Metrics for {metric_name} over {time_range}: value=42, trend=+5%"


def _register_providers() -> None:
    """Register providers in global registry."""
    registry = get_registry()
    registry.register_provider("payment_tools", PaymentToolProvider)
    registry.register_provider("analytics_tools", AnalyticsToolProvider)


_register_providers()


# ==================== Dynamic Environment ====================


@dataclass
class DynamicState(ExecutionState):
    """State with context storage."""

    context: dict[str, Any] = field(default_factory=dict)
    task_completed: bool = False


@EnvironmentRegistry.register("dynamic_service")
class DynamicServiceEnvironment(DynamicToolMixin, BaseEnvironment):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    Environment with dynamic tool loading from metadata.

    Example metadata:
        {
            "tool_implementations": {"search": "search_advanced"},
            "tool_providers": ["payment_tools"],
            "enabled_tools": ["search", "process_payment"],
            "expected_actions": ["search", "process_payment"]
        }
    """

    state: DynamicState  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        super().__init__(registry=registry)
        self.state = DynamicState()

    def setup(self, metadata: dict[str, Any]) -> None:
        """Initialize with static filtering and dynamic tool loading."""
        # Reset state
        self.state = DynamicState(context=metadata.get("context", {}))
        self._enabled_tools = None
        self.expected_actions = set(metadata.get("expected_actions", []))

        # Handle enabled_tools from metadata
        if "enabled_tools" in metadata:
            requested = set(metadata["enabled_tools"])
            available = set(self._tools.keys())
            self._enabled_tools = requested & available

        # Load dynamic tools
        self.load_dynamic_tools(metadata)

    def reset(self) -> None:
        super().reset()
        self.state = DynamicState()  # pyright: ignore[reportIncompatibleVariableOverride]
        self._dynamic_tools.clear()
        self._dynamic_tool_schemas.clear()

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute tool with state tracking."""
        result = await super().execute_tool(name, arguments)
        # State tracking is handled in parent classes
        return result

    # ==================== Static Tools ====================

    @tool(
        description="Get context information",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Context key"}},
            "required": ["key"],
        },
    )
    async def get_context(self, key: str) -> str:
        if key in self.state.context:
            return f"{key}: {self.state.context[key]}"
        return f"No context found for key: {key}"

    @tool(
        description="Set context information",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Context key"},
                "value": {"type": "string", "description": "Value to set"},
            },
            "required": ["key", "value"],
        },
    )
    async def set_context(self, key: str, value: str) -> str:
        self.state.context[key] = value
        return f"Set {key} = {value}"

    @tool(
        description="Mark task as completed",
        parameters={
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Completion summary"}},
            "required": ["summary"],
        },
    )
    async def complete_task(self, summary: str) -> str:
        self.state.task_completed = True
        return f"Task completed: {summary}"

    # ==================== Verification ====================

    def verify(self) -> float:
        """1.0 if all expected_actions executed and task completed, 0.0 otherwise."""
        if not self.state.has_executed_all(self.expected_actions):
            return 0.0
        if "complete_task" in self.expected_actions and not self.state.task_completed:
            return 0.0
        return 1.0
