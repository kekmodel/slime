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
    status = "PASS" if passed else "FAIL"
    symbol = "+" if passed else "x"
    print(f"[{symbol}] {status}: {test_name}")
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

    # Test setup with metadata
    metadata = {
        "customer": {"id": "CUST-001", "name": "Test User", "email": "test@example.com", "tier": "gold"},
        "order": {
            "id": "ORD-001",
            "product_name": "Test Product",
            "price": 99.99,
            "status": "delivered",
            "days_ago": 5,
        },
        "expected_actions": ["get_customer_info", "get_order_details"],
    }
    env.setup(metadata)

    print_result(
        "Environment setup",
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

    # Test automatic state tracking
    print_result(
        "Automatic state tracking",
        env.state.has_executed("get_customer_info"),
        f"executed_tools: {env.state.executed_tools}",
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
        "customer": {"id": "CUST-001", "name": "Test", "email": "t@t.com", "tier": "gold"},
        "order": {"id": "ORD-001", "product_name": "Test", "price": 10, "status": "delivered", "days_ago": 1},
        "enabled_tools": ["get_customer_info", "get_order_details"],
    }
    env.setup(metadata)

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

    # Test 2: task_type filtering
    env.reset()
    metadata2 = {
        "customer": {"id": "CUST-001", "name": "Test", "email": "t@t.com", "tier": "gold"},
        "order": {"id": "ORD-001", "product_name": "Test", "price": 10, "status": "delivered", "days_ago": 1},
        "task_type": "read_only",
    }
    env.setup(metadata2)

    tools = env.get_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print_result(
        "task_type='read_only' filtering",
        set(tool_names) == {"get_customer_info", "get_order_details", "submit_result"},
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
        "tool_implementations": {"search": "search_advanced", "calculate": "calculate_safe"},
        "expected_actions": ["search", "calculate"],
    }
    env.setup(metadata)

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
    env.setup(metadata2)

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


async def test_verification():
    """Test state-based verification logic."""
    print_header("Test 4: Verification Logic (State-Based)")

    from examples.slime_gym.base import parse_tool_calls
    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()
    metadata = {
        "customer": {"id": "CUST-001", "name": "Test", "email": "t@t.com", "tier": "gold"},
        "order": {"id": "ORD-001", "product_name": "Test", "price": 10, "status": "delivered", "days_ago": 1},
        "expected_actions": ["get_customer_info", "get_order_details"],
    }
    env.setup(metadata)

    # Execute all expected tools
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("get_order_details", {"order_id": "ORD-001"})

    # Verify with state
    reward = env.verify()
    print_result(
        "Verification - all actions executed",
        reward == 1.0,
        f"Reward: {reward}, executed: {env.state.executed_tools}",
    )

    # Test with missing action
    env.reset()
    env.setup(metadata)
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})

    reward = env.verify()
    print_result(
        "Verification - missing action",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0)",
    )

    # Test parse_tool_calls
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
        "customer": {"id": "CUST-001", "name": "Alice", "email": "alice@example.com", "tier": "gold"},
        "order": {"id": "ORD-123", "product_name": "Headphones", "price": 99.99, "status": "delivered", "days_ago": 5},
        "task_type": "refund",
        "expected_actions": [
            "get_customer_info",
            "get_order_details",
            "check_refund_policy",
            "process_refund",
            "send_notification",
        ],
    }
    env.setup(metadata)

    # Execute workflow
    steps = []
    steps.append(
        ("get_customer_info", (await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})).success)
    )
    steps.append(("get_order_details", (await env.execute_tool("get_order_details", {"order_id": "ORD-123"})).success))
    steps.append(
        ("check_refund_policy", (await env.execute_tool("check_refund_policy", {"order_id": "ORD-123"})).success)
    )
    steps.append(
        (
            "process_refund",
            (await env.execute_tool("process_refund", {"order_id": "ORD-123", "reason": "Defective"})).success,
        )
    )
    steps.append(
        (
            "send_notification",
            (
                await env.execute_tool("send_notification", {"customer_id": "CUST-001", "message": "Refund processed"})
            ).success,
        )
    )

    all_success = all(s[1] for s in steps)
    print_result(
        "Workflow execution",
        all_success,
        f"Steps: {[(s[0], '+' if s[1] else 'x') for s in steps]}",
    )

    # Verify
    reward = env.verify()
    print_result(
        "Full workflow verification",
        reward == 1.0,
        f"Reward: {reward}",
    )


async def test_submit_result():
    """Test submit_result for final answer verification."""
    print_header("Test 6: Submit Result Verification")

    from examples.slime_gym.retail_env import RetailServiceEnvironment

    env = RetailServiceEnvironment()

    expected_result = {
        "customer_name": "Alice",
        "order_id": "ORD-001",
        "refund_amount": 99.99,
        "status": "refunded",
    }

    metadata = {
        "customer": {"id": "CUST-001", "name": "Alice", "email": "alice@example.com", "tier": "gold"},
        "order": {"id": "ORD-001", "product_name": "Test", "price": 99.99, "status": "delivered", "days_ago": 1},
        "expected_actions": ["get_customer_info", "process_refund", "submit_result"],
        "expected_result": expected_result,
    }
    env.setup(metadata)

    # Execute tools and submit correct result
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "defective"})
    await env.execute_tool("submit_result", {"result": expected_result})

    reward = env.verify()
    print_result(
        "Correct result submission",
        reward == 1.0,
        f"Reward: {reward}",
    )

    # Test with wrong result
    env.reset()
    env.setup(metadata)
    await env.execute_tool("get_customer_info", {"customer_id": "CUST-001"})
    await env.execute_tool("process_refund", {"order_id": "ORD-001", "reason": "defective"})
    await env.execute_tool("submit_result", {"result": {"wrong": "result"}})

    reward = env.verify()
    print_result(
        "Wrong result submission",
        reward == 0.0,
        f"Reward: {reward} (expected 0.0)",
    )


async def test_safe_math_evaluator():
    """Test SafeMathEvaluator."""
    print_header("Test 7: Safe Math Evaluator")

    from examples.slime_gym.tool_registry import SafeMathEvaluator

    # Test basic operations
    test_cases = [
        ("2 + 3", 5.0),
        ("10 - 4", 6.0),
        ("3 * 4", 12.0),
        ("15 / 3", 5.0),
        ("2 ** 3", 8.0),
        ("-5", -5.0),
        ("2 + 3 * 4", 14.0),
        ("(2 + 3) * 4", 20.0),
    ]

    all_passed = True
    for expr, expected in test_cases:
        try:
            result = SafeMathEvaluator.evaluate(expr)
            if result != expected:
                all_passed = False
                print(f"       FAIL: {expr} = {result}, expected {expected}")
        except Exception as e:
            all_passed = False
            print(f"       FAIL: {expr} raised {e}")

    print_result("Basic arithmetic", all_passed)

    # Test rejection of unsafe expressions
    unsafe_cases = [
        "__import__('os')",
        "open('/etc/passwd')",
        "eval('1+1')",
        "x + 1",  # Variable
    ]

    all_rejected = True
    for expr in unsafe_cases:
        try:
            SafeMathEvaluator.evaluate(expr)
            all_rejected = False
            print(f"       FAIL: '{expr}' should have been rejected")
        except ValueError:
            pass  # Expected

    print_result("Unsafe expressions rejected", all_rejected)


async def test_environment_registry():
    """Test environment registry."""
    print_header("Test 8: Environment Registry")

    from examples.slime_gym.env_registry import EnvironmentRegistry

    # Test listing environments
    envs = EnvironmentRegistry.list_environments()
    print_result(
        "Environment listing",
        "retail_service" in envs and "dynamic_service" in envs,
        f"Registered: {envs}",
    )

    # Test getting environment
    env = EnvironmentRegistry.get("retail_service")
    print_result(
        "Get environment by name",
        env is not None and hasattr(env, "get_tools"),
        f"Got: {type(env).__name__}",
    )

    # Test unknown environment
    try:
        EnvironmentRegistry.get("unknown_env")
        print_result("Unknown environment raises error", False)
    except ValueError as e:
        print_result("Unknown environment raises error", True, str(e)[:50])


async def test_config():
    """Test configuration management."""
    print_header("Test 9: Configuration")

    from examples.slime_gym import config as cfg

    # Test per-sample override
    metadata = {"max_turns": 5}
    result = cfg.resolve_max_turns(metadata)
    print_result("Per-sample max_turns override", result == 5, f"max_turns: {result}")

    # Test dynamic mode (default: DYNAMIC_MAX_TURNS=True, MAX_TURNS_BUFFER=0)
    metadata = {"expected_actions": ["a", "b", "c"]}
    result = cfg.resolve_max_turns(metadata)
    print_result(
        "Dynamic max_turns (len + buffer)",
        result == 3,  # 3 + 0 (default buffer)
        f"max_turns: {result}",
    )

    # Test fallback (default: MAX_TURNS=10)
    metadata = {}
    result = cfg.resolve_max_turns(metadata)
    print_result("Fallback to config default", result == cfg.MAX_TURNS, f"max_turns: {result}")


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
        await test_submit_result()
        await test_safe_math_evaluator()
        await test_environment_registry()
        await test_config()

        print("\n" + "=" * 60)
        print("  All tests completed successfully!")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
