"""
Retail Service Environment - Agentic tool chain task for customer service.

Example task: Customer requests refund
Required tool chain:
1. get_customer_info -> get customer details
2. get_order_details -> get order information
3. check_refund_policy -> verify eligibility
4. process_refund -> execute refund
5. send_notification -> notify customer
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .base import BaseEnvironment, tool
from .env_registry import EnvironmentRegistry
from .types import ExecutionState


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    membership_tier: str  # bronze, silver, gold, platinum


@dataclass
class Order:
    order_id: str
    customer_id: str
    product_name: str
    total_amount: float
    status: str  # pending, shipped, delivered, cancelled, refunded
    ordered_at: datetime
    delivered_at: datetime | None = None


@dataclass
class RetailState(ExecutionState):
    """Extended state with retail-specific data."""

    customers: dict[str, Customer] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)


@EnvironmentRegistry.register("retail_service")
class RetailServiceEnvironment(BaseEnvironment):
    """
    Retail customer service environment.
    Agent must chain multiple tools in correct order.
    """

    state: RetailState
    expected_result: dict | None

    def __init__(self):
        super().__init__()
        self.state = RetailState()
        self.expected_result = None

    def setup(self, metadata: dict) -> None:
        """Initialize with task-specific data."""
        # Reset base state and expected_actions
        self.state = RetailState()
        self._enabled_tools = None
        self.expected_actions = set(metadata.get("expected_actions", []))
        self.expected_result = metadata.get("expected_result")

        # Handle enabled_tools from metadata
        if "enabled_tools" in metadata:
            requested = set(metadata["enabled_tools"])
            available = set(self._tools.keys())
            self._enabled_tools = requested & available

        # Load mock data
        self._load_mock_data(metadata)

        # Apply task_type based tool filtering (if not already set)
        if self._enabled_tools is None and "task_type" in metadata:
            task_type = metadata["task_type"]
            if task_type == "read_only":
                self._enabled_tools = {"get_customer_info", "get_order_details", "submit_result"}
            elif task_type == "refund":
                self._enabled_tools = {
                    "get_customer_info",
                    "get_order_details",
                    "check_refund_policy",
                    "process_refund",
                    "send_notification",
                    "submit_result",
                }
            elif task_type == "cancel":
                self._enabled_tools = {
                    "get_customer_info",
                    "get_order_details",
                    "cancel_order",
                    "send_notification",
                    "submit_result",
                }

    def reset(self) -> None:
        super().reset()
        self.state = RetailState()
        self.expected_result = None

    def _load_mock_data(self, metadata: dict) -> None:
        """Load mock database from metadata."""
        customer_data = metadata.get("customer", {})
        customer = Customer(
            customer_id=customer_data.get("id", "CUST-001"),
            name=customer_data.get("name", "John Doe"),
            email=customer_data.get("email", "john@example.com"),
            membership_tier=customer_data.get("tier", "gold"),
        )
        self.state.customers[customer.customer_id] = customer

        order_data = metadata.get("order", {})
        days_ago = order_data.get("days_ago", 5)
        order = Order(
            order_id=order_data.get("id", "ORD-12345"),
            customer_id=customer.customer_id,
            product_name=order_data.get("product_name", "Wireless Earbuds"),
            total_amount=order_data.get("price", 89.99),
            status=order_data.get("status", "delivered"),
            ordered_at=datetime.now() - timedelta(days=days_ago),
            delivered_at=datetime.now() - timedelta(days=max(0, days_ago - 2)),
        )
        self.state.orders[order.order_id] = order

    # ==================== Tools ====================

    @tool(
        description="Look up customer information by customer ID.",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "string", "description": "Customer ID"}},
            "required": ["customer_id"],
        },
    )
    async def get_customer_info(self, customer_id: str) -> str:
        customer = self.state.customers.get(customer_id)
        if not customer:
            return f"Error: Customer {customer_id} not found"
        return f"Customer: {customer.name}, Email: {customer.email}, Tier: {customer.membership_tier}"

    @tool(
        description="Look up order details by order ID.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order ID"}},
            "required": ["order_id"],
        },
    )
    async def get_order_details(self, order_id: str) -> str:
        order = self.state.orders.get(order_id)
        if not order:
            return f"Error: Order {order_id} not found"
        return f"Order {order.order_id}: {order.product_name}, ${order.total_amount:.2f}, Status: {order.status}"

    @tool(
        description="Check refund policy for an order.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order ID"}},
            "required": ["order_id"],
        },
    )
    async def check_refund_policy(self, order_id: str) -> str:
        order = self.state.orders.get(order_id)
        if not order:
            return f"Error: Order {order_id} not found"

        if order.status == "refunded":
            return "This order has already been refunded."

        days_since = (datetime.now() - order.delivered_at).days if order.delivered_at else 0
        if days_since > 30:
            return f"Refund not eligible: {days_since} days since delivery (max 30 days)"

        customer = self.state.customers.get(order.customer_id)
        method = "original payment" if customer.membership_tier in ["gold", "platinum"] else "store credit"
        return f"Refund eligible: ${order.total_amount:.2f} via {method}"

    @tool(
        description="Process a refund for an order.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"},
                "reason": {"type": "string", "description": "Reason for refund"},
            },
            "required": ["order_id", "reason"],
        },
    )
    async def process_refund(self, order_id: str, reason: str) -> str:
        order = self.state.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        if order.status == "refunded":
            raise ValueError("Already refunded")

        order.status = "refunded"
        return f"Refund processed: ${order.total_amount:.2f} for '{reason}'"

    @tool(
        description="Send notification to customer.",
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer ID"},
                "message": {"type": "string", "description": "Message content"},
            },
            "required": ["customer_id", "message"],
        },
    )
    async def send_notification(self, customer_id: str, message: str) -> str:
        customer = self.state.customers.get(customer_id)
        if not customer:
            return f"Error: Customer {customer_id} not found"
        return f"Notification sent to {customer.email}"

    @tool(
        description="Cancel an order. Only works for pending orders.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"},
                "reason": {"type": "string", "description": "Cancellation reason"},
            },
            "required": ["order_id", "reason"],
        },
    )
    async def cancel_order(self, order_id: str, reason: str) -> str:
        order = self.state.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        if order.status != "pending":
            raise ValueError(f"Cannot cancel {order.status} orders")
        order.status = "cancelled"
        return f"Order {order_id} cancelled: {reason}"

    @tool(
        description="Submit the final result to complete the task.",
        parameters={
            "type": "object",
            "properties": {"result": {"type": "object", "description": "The final result as a structured object"}},
            "required": ["result"],
        },
    )
    async def submit_result(self, result: dict) -> str:
        self.state.submitted_result = result
        return "Result submitted successfully"

    # ==================== Verification ====================

    def verify(self) -> float:
        """1.0 if all expected_actions executed and result matches, 0.0 otherwise."""
        if not self.state.has_executed_all(self.expected_actions):
            return 0.0
        if self.expected_result is not None:
            if self.state.submitted_result != self.expected_result:
                return 0.0
        return 1.0
