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

from slime.utils.types import Sample

from .base import BaseEnvironment, tool


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
class EnvironmentState:
    """Per-session state"""

    customers: dict[str, Customer] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    # Track all tool executions for state-based verification
    customer_info_retrieved: bool = False
    order_details_retrieved: bool = False
    policy_checked: bool = False
    refund_processed: bool = False
    order_cancelled: bool = False
    notification_sent: bool = False
    # Final result submission
    submitted_result: dict | None = None


class RetailServiceEnvironment(BaseEnvironment):
    """
    Retail customer service environment.
    Agent must chain multiple tools in correct order.
    """

    def __init__(self):
        super().__init__()
        self.state: EnvironmentState | None = None
        self.expected_actions: set[str] = set()
        self.expected_result: dict | None = None

    def seed(self, metadata: dict) -> None:
        """
        Initialize with task-specific data.

        Supports per-sample tool filtering via:
        - metadata["enabled_tools"]: Explicit list of enabled tools
        - metadata["task_type"]: Predefined tool sets ("read_only", "refund", "cancel")

        Example metadata:
            {"task_type": "read_only"}  # Only get_* tools
            {"enabled_tools": ["get_customer_info", "get_order_details"]}  # Explicit
        """
        # Call parent to handle enabled_tools from metadata
        super().seed(metadata)

        # Initialize state
        self.state = EnvironmentState()
        self._load_mock_data(metadata)
        self.expected_actions = set(metadata.get("expected_actions", []))
        self.expected_result = metadata.get("expected_result")  # For submit_result verification

        # Apply task_type based tool filtering (if not already set by enabled_tools)
        if self._enabled_tools is None and "task_type" in metadata:
            task_type = metadata["task_type"]
            if task_type == "read_only":
                # Only information retrieval tools
                self._enabled_tools = {"get_customer_info", "get_order_details", "submit_result"}
            elif task_type == "refund":
                # Full refund workflow tools
                self._enabled_tools = {
                    "get_customer_info",
                    "get_order_details",
                    "check_refund_policy",
                    "process_refund",
                    "send_notification",
                    "submit_result",
                }
            elif task_type == "cancel":
                # Cancellation workflow tools
                self._enabled_tools = {
                    "get_customer_info",
                    "get_order_details",
                    "cancel_order",
                    "send_notification",
                    "submit_result",
                }

    def reset(self) -> None:
        super().reset()  # Reset enabled_tools
        self.state = None
        self.expected_actions = set()
        self.expected_result = None

    def _load_mock_data(self, metadata: dict) -> None:
        """Load mock database from metadata"""
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
        self.state.customer_info_retrieved = True
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
        self.state.order_details_retrieved = True
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
        self.state.policy_checked = True
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
        self.state.refund_processed = True
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
        self.state.notification_sent = True
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
        self.state.order_cancelled = True
        return f"Order {order_id} cancelled: {reason}"

    @tool(
        description="Submit the final result to complete the task. Call this after gathering all required information.",
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

    async def verify(self, sample: Sample) -> float:
        """
        Binary reward: 1.0 if all conditions met, 0.0 otherwise.

        Verifies:
        1. All expected_actions were completed (state-based)
        2. submitted_result matches expected_result (exact match)

        Note: Called from generate() while env.state is still valid.
        """
        if not self.state:
            return 0.0

        # Map action names to state flags
        action_to_state = {
            "get_customer_info": self.state.customer_info_retrieved,
            "get_order_details": self.state.order_details_retrieved,
            "check_refund_policy": self.state.policy_checked,
            "process_refund": self.state.refund_processed,
            "cancel_order": self.state.order_cancelled,
            "send_notification": self.state.notification_sent,
            "submit_result": self.state.submitted_result is not None,
        }

        # 1. Verify all expected actions were completed
        for action in self.expected_actions:
            if action in action_to_state and not action_to_state[action]:
                return 0.0

        # 2. Verify submitted result matches expected (if expected_result is defined)
        if self.expected_result is not None:
            if self.state.submitted_result != self.expected_result:
                return 0.0

        return 1.0
