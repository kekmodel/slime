"""
Dynamic Environment - Example of environment with runtime tool loading.

Demonstrates how to:
1. Load tool implementations from metadata
2. Mix static and dynamic tools
3. Use tool providers for modular tool organization

Example metadata:
    {
        "env_name": "dynamic_service",
        "tool_implementations": {
            "search": "search_advanced",
            "calculate": "calculate_safe"
        },
        "tool_providers": ["payment_tools"],
        "expected_actions": ["search", "calculate"]
    }
"""

from dataclasses import dataclass, field

from slime.utils.types import Sample

from .base import BaseEnvironment, tool
from .gym_types import ToolResult
from .tool_registry import DynamicToolMixin, ToolDefinition, get_registry

# ==================== Custom Tool Providers ====================


class PaymentToolProvider:
    """Example tool provider for payment-related tools."""

    @staticmethod
    def get_tools() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="process_payment",
                description="Process a payment transaction",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Payment amount",
                        },
                        "method": {
                            "type": "string",
                            "description": "Payment method",
                        },
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
                        "transaction_id": {
                            "type": "string",
                            "description": "Original transaction ID",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Refund amount",
                        },
                    },
                    "required": ["transaction_id"],
                },
                implementation=PaymentToolProvider._refund_payment,
                tags=["payment", "refund"],
            ),
        ]

    @staticmethod
    async def _process_payment(amount: float, method: str) -> str:
        return f"Payment of ${amount:.2f} via {method} processed successfully. Transaction ID: TXN-12345"

    @staticmethod
    async def _refund_payment(transaction_id: str, amount: float | None = None) -> str:
        amount_str = f"${amount:.2f}" if amount else "full amount"
        return f"Refund of {amount_str} for transaction {transaction_id} processed."


class AnalyticsToolProvider:
    """Example tool provider for analytics tools."""

    @staticmethod
    def get_tools() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_metrics",
                description="Get analytics metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "Name of the metric",
                        },
                        "time_range": {
                            "type": "string",
                            "description": "Time range (e.g., '7d', '30d')",
                        },
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


# Register providers in global registry
def _register_providers():
    registry = get_registry()
    registry.register_provider("payment_tools", PaymentToolProvider)
    registry.register_provider("analytics_tools", AnalyticsToolProvider)


_register_providers()


# ==================== Dynamic Environment State ====================


@dataclass
class DynamicState:
    """State for dynamic environment."""

    context: dict = field(default_factory=dict)
    tool_calls_made: list[str] = field(default_factory=list)
    task_completed: bool = False


# ==================== Dynamic Environment ====================


class DynamicServiceEnvironment(DynamicToolMixin, BaseEnvironment):
    """
    Environment with dynamic tool loading from metadata.

    Supports:
    - Static tools defined with @tool decorator
    - Dynamic tools loaded from registry via tool_implementations
    - Tool providers for modular tool organization
    - Per-sample tool filtering

    Example metadata:
        {
            "tool_implementations": {"search": "search_advanced"},
            "tool_providers": ["payment_tools"],
            "enabled_tools": ["search", "process_payment"],
            "expected_actions": ["search", "process_payment"]
        }
    """

    def __init__(self):
        super().__init__()
        self.state: DynamicState | None = None
        self.expected_actions: set[str] = set()

    def seed(self, metadata: dict) -> None:
        """
        Initialize environment with static filtering and dynamic tool loading.
        """
        # 1. Call BaseEnvironment.seed() for enabled_tools filtering
        super().seed(metadata)

        # 2. Load dynamic tools from metadata
        self.load_dynamic_tools(metadata)

        # 3. Initialize state
        self.state = DynamicState(
            context=metadata.get("context", {}),
        )
        self.expected_actions = set(metadata.get("expected_actions", []))

    def reset(self) -> None:
        super().reset()
        self.state = None
        self.expected_actions = set()
        self._dynamic_tools.clear()
        self._dynamic_tool_schemas.clear()

    # ==================== Tool Execution ====================

    async def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute tool and track in state."""
        result = await super().execute_tool(name, arguments)
        # Track successful tool calls in state
        if result.success and self.state:
            self.state.tool_calls_made.append(name)
        return result

    # ==================== Static Tools ====================

    @tool(
        description="Get information about the current context",
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Context key to retrieve",
                }
            },
            "required": ["key"],
        },
    )
    async def get_context(self, key: str) -> str:
        """Static tool: Get context information."""
        if self.state and key in self.state.context:
            return f"{key}: {self.state.context[key]}"
        return f"No context found for key: {key}"

    @tool(
        description="Update the context with new information",
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
        """Static tool: Set context information."""
        if self.state:
            self.state.context[key] = value
            return f"Set {key} = {value}"
        return "Error: State not initialized"

    @tool(
        description="Mark the current task as completed",
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Task completion summary",
                }
            },
            "required": ["summary"],
        },
    )
    async def complete_task(self, summary: str) -> str:
        """Static tool: Mark task as complete."""
        if self.state:
            self.state.task_completed = True
            return f"Task completed: {summary}"
        return "Error: State not initialized"

    # ==================== Verification ====================

    async def verify(self, sample: Sample) -> float:
        """
        State-based verification.

        Checks:
        1. All expected_actions were executed (tracked in state.tool_calls_made)
        2. Task was marked as completed (if required)

        Note: Called from generate() while env.state is still valid.
        """
        if not self.state:
            return 0.0

        # Check all expected actions were executed
        executed = set(self.state.tool_calls_made)
        if not self.expected_actions.issubset(executed):
            return 0.0

        # Check task completion if expected
        if "complete_task" in self.expected_actions:
            if not self.state.task_completed:
                return 0.0

        return 1.0


# ==================== Usage Examples ====================

"""
Example 1: Basic dynamic tools

metadata = {
    "tool_implementations": {
        "search": "search_advanced",      # Use advanced search implementation
        "calculate": "calculate_safe",    # Use safe calculator
    },
    "expected_actions": ["search", "complete_task"],
}


Example 2: With tool providers

metadata = {
    "tool_providers": ["payment_tools"],  # Load all payment tools
    "expected_actions": ["process_payment", "complete_task"],
}


Example 3: Mixed static, dynamic, and filtered

metadata = {
    "tool_implementations": {"search": "search_basic"},
    "tool_providers": ["analytics_tools"],
    "enabled_tools": ["get_context", "search", "get_metrics"],  # Filter tools
    "expected_actions": ["search", "get_metrics"],
}


Example 4: Context-aware task

metadata = {
    "context": {
        "user_id": "USER-123",
        "session_type": "support",
    },
    "tool_providers": ["payment_tools"],
    "expected_actions": ["get_context", "process_payment", "complete_task"],
}
"""
