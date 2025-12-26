"""
Standalone test script for slime_gym functionality.
Mocks SLIME dependencies to run without torch.

Run:
    cd slime_repo
    python examples/slime_gym/test_gym_standalone.py
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)

# ==================== Mock SLIME types ====================


class MockStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class MockSample:
    """Mock of slime.utils.types.Sample"""

    prompt: str | list = ""
    response: str = ""
    tokens: list = field(default_factory=list)
    response_length: int = 0
    loss_mask: list = field(default_factory=list)
    rollout_log_probs: list = field(default_factory=list)
    reward: Any = None
    metadata: dict = field(default_factory=dict)
    status: MockStatus = MockStatus.PENDING

    class Status:
        PENDING = MockStatus.PENDING
        COMPLETED = MockStatus.COMPLETED
        TRUNCATED = MockStatus.TRUNCATED
        ABORTED = MockStatus.ABORTED


# Mock the slime module before importing slime_gym
sys.modules["slime"] = type(sys)("slime")
sys.modules["slime.utils"] = type(sys)("slime.utils")
sys.modules["slime.utils.types"] = type(sys)("slime.utils.types")
sys.modules["slime.utils.types"].Sample = MockSample


# ==================== Test utilities ====================


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"       {details}")


# ==================== Tests ====================


async def test_basic_environment():
    """Test basic RetailServiceEnvironment functionality."""
    print_header("Test 1: Basic Environment")

    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()

    # Test tool registration
    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    expected_tools = [
        "get_customer_info",
        "get_order_details",
        "check_refund_policy",
        "process_refund",
        "send_notification",
        "cancel_order",
    ]

    print_result(
        "Tool registration",
        all(t in tool_names for t in expected_tools),
        f"Found: {tool_names}",
    )

    # Test seed with metadata
    metadata = {
        "customer": {
            "id": "CUST-001",
            "name": "Test User",
            "email": "test@example.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test Product",
            "price": 99.99,
            "status": "delivered",
            "days_ago": 5,
        },
        "expected_actions": ["get_customer_info", "get_order_details"],
    }
    env.seed(metadata)

    print_result(
        "Environment seeding",
        env.state is not None and env.expected_actions == {"get_customer_info", "get_order_details"},
        f"State initialized, expected_actions: {env.expected_actions}",
    )

    # Test tool execution
    result = await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    print_result(
        "Tool execution",
        result.success and "Test User" in result.output,
        f"Output: {result.output[:50]}...",
    )

    # Test unknown tool
    result = await env.execute_tool("unknown_tool", {})
    print_result(
        "Unknown tool handling",
        not result.success and "Unknown tool" in result.output,
        f"Output: {result.output[:50]}...",
    )


async def test_tool_filtering():
    """Test per-sample tool filtering."""
    print_header("Test 2: Tool Filtering")

    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()

    # Test 1: enabled_tools filtering
    metadata = {
        "customer": {
            "id": "CUST-001",
            "name": "Test",
            "email": "t@t.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test",
            "price": 10,
            "status": "delivered",
            "days_ago": 1,
        },
        "enabled_tools": ["get_customer_info", "get_order_details"],
    }
    env.seed(metadata)

    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print_result(
        "enabled_tools filtering",
        set(tool_names) == {"get_customer_info", "get_order_details"},
        f"Active tools: {tool_names}",
    )

    # Test executing disabled tool
    result = await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "test"})
    print_result(
        "Disabled tool blocked",
        not result.success and "not available" in result.output,
        f"Output: {result.output[:60]}...",
    )

    # Test 2: task_type filtering - reset first
    env.reset()
    metadata2 = {
        "customer": {
            "id": "CUST-001",
            "name": "Test",
            "email": "t@t.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test",
            "price": 10,
            "status": "delivered",
            "days_ago": 1,
        },
        "task_type": "read_only",
    }
    env.seed(metadata2)

    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print_result(
        "task_type='read_only' filtering",
        set(tool_names) == {"get_customer_info", "get_order_details", "submit_result"},
        f"Active tools: {tool_names}",
    )

    # Test 3: task_type="refund"
    env.reset()
    metadata3 = {
        "customer": {
            "id": "CUST-001",
            "name": "Test",
            "email": "t@t.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test",
            "price": 10,
            "status": "delivered",
            "days_ago": 1,
        },
        "task_type": "refund",
    }
    env.seed(metadata3)

    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    expected = {
        "get_customer_info",
        "get_order_details",
        "check_refund_policy",
        "process_refund",
        "send_notification",
        "submit_result",
    }
    print_result(
        "task_type='refund' filtering",
        set(tool_names) == expected,
        f"Active tools: {tool_names}",
    )


async def test_dynamic_tools():
    """Test dynamic tool loading from metadata."""
    print_header("Test 3: Dynamic Tool Loading")

    from examples.slime_gym.dynamic_env import DynamicServiceEnvironment
    from examples.slime_gym.tool_registry import get_registry

    # Check registry has pre-registered tools
    registry = get_registry()
    registered = registry.list_tools()
    print_result(
        "Tool registry populated",
        len(registered) > 0,
        f"Registered tools: {registered}",
    )

    env = DynamicServiceEnvironment()

    # Test with tool_implementations
    metadata = {
        "tool_implementations": {
            "search": "search_advanced",
            "calculate": "calculate_safe",
        },
        "expected_actions": ["search", "calculate"],
    }
    env.seed(metadata)

    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print_result(
        "Dynamic tool loading (tool_implementations)",
        "search" in tool_names and "calculate" in tool_names,
        f"Loaded tools: {tool_names}",
    )

    # Test executing dynamic tool
    result = await env.execute_tool("search", {"query": "Python tutorials"})
    print_result(
        "Dynamic tool execution",
        result.success and "Advanced" in result.output,
        f"Output: {result.output}",
    )

    # Test with tool_providers
    env.reset()
    metadata2 = {
        "tool_providers": ["payment_tools"],
        "expected_actions": ["process_payment"],
    }
    env.seed(metadata2)

    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print_result(
        "Tool provider loading",
        "process_payment" in tool_names,
        f"Loaded tools: {tool_names}",
    )

    # Test executing provider tool
    result = await env.execute_tool("process_payment", {"amount": 99.99, "method": "credit_card"})
    print_result(
        "Provider tool execution",
        result.success and "99.99" in result.output,
        f"Output: {result.output}",
    )

    # Test state-based verification for dynamic env
    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Dynamic env state-based verification",
        reward == 1.0,
        f"Reward: {reward}, tool_calls_made: {env.state.tool_calls_made}",
    )

    # Test incomplete execution
    env.reset()
    metadata3 = {
        "tool_implementations": {"search": "search_advanced"},
        "expected_actions": ["search", "complete_task"],
    }
    env.seed(metadata3)
    await env.execute_tool("search", {"query": "test"})
    # Don't call complete_task

    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Dynamic env incomplete verification",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0, missing complete_task)",
    )


async def test_verification():
    """Test state-based verification logic."""
    print_header("Test 4: Verification Logic (State-Based)")

    from examples.slime_gym.base import parse_tool_calls
    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()
    metadata = {
        "customer": {
            "id": "CUST-001",
            "name": "Test",
            "email": "t@t.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test",
            "price": 10,
            "status": "delivered",
            "days_ago": 1,
        },
        "expected_actions": ["get_customer_info", "get_order_details"],
    }
    env.seed(metadata)

    # Execute all expected tools to set state
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("get_order_details", {"order_id": "ORD-001"})

    # Verify with state (response content doesn't matter for state-based verification)
    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Verification - all actions executed",
        reward == 1.0,
        f"Reward: {reward}, state: customer_info={env.state.customer_info_retrieved}, order_details={env.state.order_details_retrieved}",
    )

    # Test with missing action - reset and only execute one tool
    env.reset()
    env.seed(metadata)
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    # Don't execute get_order_details

    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Verification - missing action",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0, order_details_retrieved={env.state.order_details_retrieved})",
    )

    # Test parse_tool_calls (still useful for response parsing)
    response = """
<tool_call>{"name": "get_customer_info", "arguments": {"customer_id": "CUST-001"}}</tool_call>
<tool_call>{"name": "get_order_details", "arguments": {"order_id": "ORD-001"}}</tool_call>
"""
    tool_calls = parse_tool_calls(response)
    print_result(
        "parse_tool_calls",
        len(tool_calls) == 2 and tool_calls[0].name == "get_customer_info",
        f"Parsed {len(tool_calls)} tool calls: {[tc.name for tc in tool_calls]}",
    )


async def test_refund_workflow():
    """Test complete refund workflow."""
    print_header("Test 5: Complete Refund Workflow")

    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()
    metadata = {
        "customer": {
            "id": "CUST-001",
            "name": "Alice",
            "email": "alice@example.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-123",
            "product_name": "Headphones",
            "price": 99.99,
            "status": "delivered",
            "days_ago": 5,
        },
        "task_type": "refund",
        "expected_actions": [
            "get_customer_info",
            "get_order_details",
            "check_refund_policy",
            "process_refund",
            "send_notification",
        ],
    }
    env.seed(metadata)

    # Execute workflow
    steps = []

    # Step 1: Get customer info
    result = await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    steps.append(("get_customer_info", result.success))

    # Step 2: Get order details
    result = await env.execute_tool("get_order_details", {"order_id": "ORD-123"})
    steps.append(("get_order_details", result.success))

    # Step 3: Check refund policy
    result = await env.execute_tool("check_refund_policy", {"order_id": "ORD-123"})
    steps.append(("check_refund_policy", result.success))
    policy_checked = env.state.policy_checked

    # Step 4: Process refund
    result = await env.execute_tool("process_refund", {"order_id": "ORD-123", "reason": "Defective product"})
    steps.append(("process_refund", result.success))
    refund_processed = env.state.refund_processed

    # Step 5: Send notification
    result = await env.execute_tool(
        "send_notification",
        {"customer_id": "CUST-001", "message": "Refund processed"},
    )
    steps.append(("send_notification", result.success))

    all_success = all(s[1] for s in steps)
    print_result(
        "Workflow execution",
        all_success,
        f"Steps: {[(s[0], '✓' if s[1] else '✗') for s in steps]}",
    )

    print_result(
        "State tracking",
        policy_checked and refund_processed,
        f"policy_checked={policy_checked}, refund_processed={refund_processed}",
    )

    # Verify with correct response
    response = """
<tool_call>{"name": "get_customer_info", "arguments": {"customer_id": "CUST-001"}}</tool_call>
<tool_call>{"name": "get_order_details", "arguments": {"order_id": "ORD-123"}}</tool_call>
<tool_call>{"name": "check_refund_policy", "arguments": {"order_id": "ORD-123"}}</tool_call>
<tool_call>{"name": "process_refund", "arguments": {"order_id": "ORD-123", "reason": "Defective"}}</tool_call>
<tool_call>{"name": "send_notification", "arguments": {"customer_id": "CUST-001", "message": "Done"}}</tool_call>
"""
    sample = MockSample(response=response)
    reward = await env.verify(sample)
    print_result("Full workflow verification", reward == 1.0, f"Reward: {reward}")


async def test_order_independent():
    """Test that tool execution order doesn't matter - only completion counts."""
    print_header("Test 6: Order-Independent Execution")

    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()
    metadata = {
        "customer": {
            "id": "CUST-001",
            "name": "Test",
            "email": "t@t.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test",
            "price": 10,
            "status": "delivered",
            "days_ago": 1,
        },
        "expected_actions": ["check_refund_policy", "process_refund"],
    }
    env.seed(metadata)

    # Execute in "wrong" order - should still work
    result = await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "test"})
    print_result(
        "process_refund first (any order works)",
        result.success and "Refund processed" in result.output,
        f"Output: {result.output}",
    )

    await env.execute_tool("check_refund_policy", {"order_id": "ORD-001"})

    # Verify: all expected actions completed
    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Verification - all actions completed (any order)",
        reward == 1.0,
        f"Reward: {reward}, policy_checked={env.state.policy_checked}, refund_processed={env.state.refund_processed}",
    )

    # Test incomplete execution
    env.reset()
    env.seed(metadata)
    await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "test"})
    # Don't call check_refund_policy

    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Verification - incomplete (missing check_refund_policy)",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0, policy_checked={env.state.policy_checked})",
    )


async def test_submit_result():
    """Test submit_result for final answer verification."""
    print_header("Test 7: Submit Result Verification")

    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()

    # Define expected result
    expected_result = {
        "customer_name": "Alice",
        "order_id": "ORD-001",
        "refund_amount": 99.99,
        "status": "refunded",
    }

    metadata = {
        "customer": {
            "id": "CUST-001",
            "name": "Alice",
            "email": "alice@example.com",
            "tier": "gold",
        },
        "order": {
            "id": "ORD-001",
            "product_name": "Test",
            "price": 99.99,
            "status": "delivered",
            "days_ago": 1,
        },
        "expected_actions": [
            "get_customer_info",
            "process_refund",
            "submit_result",
        ],
        "expected_result": expected_result,
    }
    env.seed(metadata)

    # Execute tools
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "defective"})

    # Submit correct result
    await env.execute_tool("submit_result", {"result": expected_result})

    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Correct result submission",
        reward == 1.0,
        f"Reward: {reward}, submitted={env.state.submitted_result}",
    )

    # Test with wrong result
    env.reset()
    env.seed(metadata)

    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "defective"})

    # Submit wrong result
    wrong_result = {
        "customer_name": "Bob",
        "order_id": "ORD-001",
        "refund_amount": 50.0,
        "status": "refunded",
    }
    await env.execute_tool("submit_result", {"result": wrong_result})

    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Wrong result submission",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0, submitted != expected)",
    )

    # Test without submit_result call
    env.reset()
    env.seed(metadata)

    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "defective"})
    # Don't call submit_result

    sample = MockSample(response="")
    reward = await env.verify(sample)
    print_result(
        "Missing submit_result",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0, submit_result not called)",
    )


async def main():
    print("\n" + "=" * 60)
    print("  slime_gym Standalone Test Suite")
    print("=" * 60)

    try:
        await test_basic_environment()
        await test_tool_filtering()
        await test_dynamic_tools()
        await test_verification()
        await test_refund_workflow()
        await test_order_independent()
        await test_submit_result()

        print("\n" + "=" * 60)
        print("  ✅ All tests completed successfully!")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
