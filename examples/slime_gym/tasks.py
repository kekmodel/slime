"""
Example tasks for retail service environment.

Each task is a dict that can be used as sample metadata.
The task defines:
- customer/order data for the environment
- expected_actions: required tool calls for success
- env_name (optional): environment type for per-sample selection
- task_type (optional): predefined tool set ("read_only", "refund", "cancel")
- enabled_tools (optional): explicit list of available tools

Customization examples:
    # Per-sample environment selection
    {"env_name": "retail_service", ...}
    {"env_name": "banking_service", ...}

    # Task-type based tool filtering
    {"task_type": "read_only", ...}  # Only get_* tools
    {"task_type": "refund", ...}     # Refund workflow tools

    # Explicit tool filtering
    {"enabled_tools": ["get_customer_info", "get_order_details"], ...}

    # Result verification with submit_result
    {"expected_actions": [..., "submit_result"], "expected_result": {...}}

Note: Verification is state-based (tool execution tracking).
      Use expected_result for exact match verification of final answers.
"""

# ==================== Refund Tasks with Result Verification ====================

REFUND_TASK_WITH_RESULT = {
    "task_id": "refund_with_result_001",
    "env_name": "retail_service",
    "task_type": "refund",
    "customer": {
        "id": "CUST-001",
        "name": "Alice Kim",
        "email": "alice@example.com",
        "tier": "gold",
    },
    "order": {
        "id": "ORD-12345",
        "product_name": "Wireless Earbuds",
        "price": 89.99,
        "status": "delivered",
        "days_ago": 5,
    },
    # Required tool calls including submit_result
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
        "check_refund_policy",
        "process_refund",
        "send_notification",
        "submit_result",
    ],
    # Expected final result (exact match verification)
    "expected_result": {
        "customer_name": "Alice Kim",
        "order_id": "ORD-12345",
        "refund_amount": 89.99,
        "status": "refunded",
    },
}

# ==================== Refund Tasks (without result verification) ====================

REFUND_TASK_SIMPLE = {
    "task_id": "refund_001",
    "env_name": "retail_service",  # Per-sample environment selection
    "task_type": "refund",  # Enables only refund-related tools
    "customer": {
        "id": "CUST-001",
        "name": "Alice Kim",
        "email": "alice@example.com",
        "tier": "gold",
    },
    "order": {
        "id": "ORD-12345",
        "product_name": "Wireless Earbuds",
        "price": 89.99,
        "status": "delivered",
        "days_ago": 5,
    },
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
        "check_refund_policy",
        "process_refund",
        "send_notification",
    ],
}

REFUND_TASK_BRONZE_TIER = {
    "task_id": "refund_002",
    "customer": {
        "id": "CUST-002",
        "name": "Bob Lee",
        "email": "bob@example.com",
        "tier": "bronze",  # Gets store credit instead of refund
    },
    "order": {
        "id": "ORD-67890",
        "product_name": "Bluetooth Speaker",
        "price": 149.99,
        "status": "delivered",
        "days_ago": 10,
    },
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
        "check_refund_policy",
        "process_refund",
        "send_notification",
    ],
}

REFUND_TASK_EXPIRED = {
    "task_id": "refund_003",
    "customer": {
        "id": "CUST-003",
        "name": "Charlie Park",
        "email": "charlie@example.com",
        "tier": "silver",
    },
    "order": {
        "id": "ORD-11111",
        "product_name": "Smart Watch",
        "price": 299.99,
        "status": "delivered",
        "days_ago": 45,  # Beyond 30-day policy
    },
    # For expired refunds, agent should check policy and inform customer
    # No refund should be processed
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
        "check_refund_policy",
        "send_notification",  # Notify about policy
    ],
}

# ==================== Cancellation Tasks ====================

CANCEL_TASK_PENDING = {
    "task_id": "cancel_001",
    "task_type": "cancel",  # Enables only cancellation-related tools
    "customer": {
        "id": "CUST-004",
        "name": "Diana Choi",
        "email": "diana@example.com",
        "tier": "platinum",
    },
    "order": {
        "id": "ORD-22222",
        "product_name": "Gaming Keyboard",
        "price": 179.99,
        "status": "pending",  # Can be cancelled
        "days_ago": 1,
    },
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
        "cancel_order",
        "send_notification",
    ],
}

CANCEL_TASK_SHIPPED = {
    "task_id": "cancel_002",
    "task_type": "cancel",
    "customer": {
        "id": "CUST-005",
        "name": "Eric Jung",
        "email": "eric@example.com",
        "tier": "gold",
    },
    "order": {
        "id": "ORD-33333",
        "product_name": "Laptop Stand",
        "price": 59.99,
        "status": "shipped",  # Cannot cancel, needs refund after delivery
        "days_ago": 3,
    },
    # Agent should recognize shipped orders can't be cancelled
    # and inform the customer
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
        "send_notification",
    ],
}

# ==================== Read-Only Tasks ====================

READ_ONLY_TASK = {
    "task_id": "info_001",
    "task_type": "read_only",  # Only get_customer_info, get_order_details available
    "customer": {
        "id": "CUST-006",
        "name": "Frank Lee",
        "email": "frank@example.com",
        "tier": "silver",
    },
    "order": {
        "id": "ORD-44444",
        "product_name": "USB-C Hub",
        "price": 49.99,
        "status": "shipped",
        "days_ago": 2,
    },
    # Agent only needs to look up information
    "expected_actions": [
        "get_customer_info",
        "get_order_details",
    ],
}

# Example with explicit tool filtering
EXPLICIT_TOOLS_TASK = {
    "task_id": "explicit_001",
    # Explicit list of enabled tools (overrides task_type)
    "enabled_tools": ["get_customer_info", "send_notification"],
    "customer": {
        "id": "CUST-007",
        "name": "Grace Kim",
        "email": "grace@example.com",
        "tier": "gold",
    },
    "order": {
        "id": "ORD-55555",
        "product_name": "Wireless Mouse",
        "price": 29.99,
        "status": "delivered",
        "days_ago": 1,
    },
    "expected_actions": [
        "get_customer_info",
        "send_notification",
    ],
}

# ==================== Sample Prompts ====================


def get_prompt_for_task(task: dict) -> list[dict]:
    """Generate a user prompt for the given task."""
    customer = task["customer"]
    order = task["order"]
    task_type = task.get("task_type", "")
    task_id = task["task_id"]

    if "refund" in task_id or task_type == "refund":
        content = (
            f"Hi, I'm {customer['name']} (customer ID: {customer['id']}). "
            f"I'd like to request a refund for my order {order['id']}. "
            f"The {order['product_name']} I received isn't working properly."
        )
    elif "cancel" in task_id or task_type == "cancel":
        content = (
            f"Hi, I'm {customer['name']} (customer ID: {customer['id']}). "
            f"I need to cancel my order {order['id']} for the {order['product_name']}. "
            f"I changed my mind about the purchase."
        )
    elif "info" in task_id or task_type == "read_only":
        content = (
            f"Hi, I'm {customer['name']} (customer ID: {customer['id']}). "
            f"Can you tell me the status of my order {order['id']}?"
        )
    else:
        content = (
            f"Hi, I'm {customer['name']} (customer ID: {customer['id']}). "
            f"I have a question about my order {order['id']}."
        )

    return [{"role": "user", "content": content}]


# ==================== All Tasks ====================

ALL_TASKS = [
    # Refund task with result verification (submit_result + expected_result)
    REFUND_TASK_WITH_RESULT,
    # Refund tasks (task_type="refund")
    REFUND_TASK_SIMPLE,
    REFUND_TASK_BRONZE_TIER,
    REFUND_TASK_EXPIRED,
    # Cancellation tasks (task_type="cancel")
    CANCEL_TASK_PENDING,
    CANCEL_TASK_SHIPPED,
    # Read-only task (task_type="read_only")
    READ_ONLY_TASK,
    # Explicit tool filtering example
    EXPLICIT_TOOLS_TASK,
]


def generate_training_samples() -> list[dict]:
    """
    Generate training samples in SLIME format.

    Returns list of dicts with:
    - prompt: list of messages
    - metadata: task configuration
    """
    samples = []
    for task in ALL_TASKS:
        samples.append(
            {
                "prompt": get_prompt_for_task(task),
                "metadata": task,
            }
        )
    return samples


if __name__ == "__main__":
    import json

    # Generate and print samples
    samples = generate_training_samples()
    for sample in samples:
        print(json.dumps(sample, indent=2))
        print("-" * 50)
