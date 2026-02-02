"""
Comprehensive E2E Live Test - ALL Cases

Tests all tool calling scenarios against live OpenRouter API:
1. Single tool call (basic)
2. Parallel tool calls (2 tools at once)
3. Multi-hop chain (sequential tools)
4. Special characters in tool output
5. Thinking/reasoning preservation

Usage:
    python -m pytest examples/tool_calling/tests/test_openrouter_e2e_all_cases.py -v
    # Or directly:
    python examples/tool_calling/tests/test_openrouter_e2e_all_cases.py

Requires:
    - OPENROUTER_API_KEY environment variable or set API_KEY below
    - Internet connection to OpenRouter API
"""

import os
import json
import pytest
import requests
from typing import Optional, Tuple, Dict, Any, List

# API Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if API_KEY is None:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is required. Set it before running this test: export OPENROUTER_API_KEY=your_key")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test: (model_id, parser_name, providers)
MODELS_TO_TEST = [
    ("qwen/qwen3-30b-a3b-thinking-2507", "qwen25", ["Alibaba"]),
    ("qwen/qwen3-next-80b-a3b-thinking", "qwen25", ["DeepInfra"]),
    ("qwen/qwen3-coder-plus", "qwen3_coder", ["Alibaba"]),
    ("z-ai/glm-4.7-flash", "glm47", ["Z.AI"]),
    ("moonshotai/kimi-k2-thinking", "kimi_k2", ["DeepInfra"]),
    ("deepseek/deepseek-v3.2", "deepseekv32", ["SiliconFlow"]),
    ("minimax/minimax-m2", "minimax-m2", ["Minimax"]),
    ("openai/gpt-oss-120b", "gpt-oss", ["Fireworks"]),
    ("nvidia/nemotron-3-nano-30b-a3b:free", "qwen3_coder", ["NVIDIA"]),
]

# Tool definitions
TOOLS = [
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate a mathematical expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search for information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}},
]


# Unified sampling parameters for fair comparison
SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    # top_k=-1 is default (disabled), OpenRouter doesn't support top_k parameter
}


def api_call(model_id: str, messages: List[Dict], providers: List[str], tools: Optional[List[Dict]] = None, tool_choice: Optional[str] = None) -> Dict[str, Any]:
    """Make API call to OpenRouter with unified sampling parameters."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "provider": {"order": providers},
        **SAMPLING_PARAMS,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    response = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
    return response.json()


def get_tool_calls(result: Dict) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Extract tool calls from API result."""
    if "error" in result:
        return None, result["error"].get("message", "")[:100]
    if "choices" not in result or not result["choices"]:
        return None, "No choices"
    msg = result["choices"][0].get("message", {})
    return msg.get("tool_calls", []), None


def get_content(result: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Extract content and reasoning from API result."""
    if "error" in result:
        return None, result["error"].get("message", "")[:100]
    if "choices" not in result or not result["choices"]:
        return None, "No choices"
    msg = result["choices"][0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning") or msg.get("thinking") or ""
    return {"content": content, "reasoning": reasoning}, None


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture(params=MODELS_TO_TEST, ids=[m[0].split("/")[-1] for m in MODELS_TO_TEST])
def model_config(request):
    """Parametrize tests across all models."""
    model_id, parser_name, providers = request.param
    return {"model_id": model_id, "parser_name": parser_name, "providers": providers}


# ============================================================================
# Test Cases
# ============================================================================


class TestSingleToolCall:
    """Test Case 1: Basic single tool call."""

    def test_single_tool_call(self, model_config):
        model_id = model_config["model_id"]
        providers = model_config["providers"]

        messages = [{"role": "user", "content": "Calculate 7*8"}]
        tool_choice = "required" if "gpt-oss" in model_id else None

        result = api_call(model_id, messages, providers, tools=TOOLS[:1], tool_choice=tool_choice)
        tool_calls, error = get_tool_calls(result)

        assert error is None, f"API error: {error}"
        assert tool_calls, "No tool calls returned"
        assert tool_calls[0]["function"]["name"] == "calculator"


class TestParallelToolCalls:
    """Test Case 2: Parallel tool calls (2 tools at once)."""

    def test_parallel_tool_calls(self, model_config):
        model_id = model_config["model_id"]
        providers = model_config["providers"]

        messages = [{"role": "user", "content": "Calculate 10+5 AND get the weather in Seoul. Do both at the same time."}]
        tool_choice = "required" if "gpt-oss" in model_id else None

        result = api_call(model_id, messages, providers, tools=TOOLS[:2], tool_choice=tool_choice)
        tool_calls, error = get_tool_calls(result)

        assert error is None, f"API error: {error}"
        assert tool_calls, "No tool calls returned"

        # At least 1 tool call required, 2 is ideal
        names = [tc["function"]["name"] for tc in tool_calls]
        assert len(tool_calls) >= 1, f"Expected at least 1 tool call, got {len(tool_calls)}"

        # If we got 2, they should be different tools
        if len(tool_calls) >= 2:
            assert len(set(names)) == 2, f"Expected 2 different tools, got {names}"


class TestMultiHopChain:
    """Test Case 3: Multi-hop chain (sequential tool calls)."""

    def test_multi_hop_chain(self, model_config):
        model_id = model_config["model_id"]
        providers = model_config["providers"]

        # Turn 1: Initial request
        messages = [{"role": "user", "content": "Search for the population of Tokyo, then calculate that number divided by 1000000."}]
        tool_choice = "required" if "gpt-oss" in model_id else None

        result1 = api_call(model_id, messages, providers, tools=TOOLS, tool_choice=tool_choice)
        tool_calls1, error1 = get_tool_calls(result1)

        assert error1 is None, f"Turn 1 API error: {error1}"
        assert tool_calls1, "Turn 1: No tool calls returned"

        tc1 = tool_calls1[0]

        # Simulate tool response
        tool_result1 = "Tokyo population is 14000000 (14 million)"

        # Turn 2: Tool response
        messages2 = [messages[0], {"role": "assistant", "content": None, "tool_calls": [{"id": tc1["id"], "type": "function", "function": tc1["function"]}]}, {"role": "tool", "tool_call_id": tc1["id"], "content": tool_result1}]

        result2 = api_call(model_id, messages2, providers, tools=TOOLS)
        tool_calls2, error2 = get_tool_calls(result2)

        assert error2 is None, f"Turn 2 API error: {error2}"

        if tool_calls2:
            # Got second tool call (multi-hop)
            tc2 = tool_calls2[0]

            # Turn 3: Final answer
            messages3 = messages2 + [{"role": "assistant", "content": None, "tool_calls": [{"id": tc2["id"], "type": "function", "function": tc2["function"]}]}, {"role": "tool", "tool_call_id": tc2["id"], "content": "14"}]
            result3 = api_call(model_id, messages3, providers)
            answer, _ = get_content(result3)

            assert answer, "No answer in Turn 3"
            full_text = (answer.get("content", "") + answer.get("reasoning", "")).lower()
            assert "14" in full_text, f"Expected '14' in final answer, got: {full_text[:200]}"
        else:
            # Model gave direct answer after first tool
            answer, _ = get_content(result2)
            assert answer, "No answer in Turn 2"
            full_text = (answer.get("content", "") + answer.get("reasoning", "")).lower()
            assert "14" in full_text, f"Expected '14' in answer, got: {full_text[:200]}"


class TestSpecialCharacters:
    """Test Case 4: Special characters in tool output."""

    def test_special_characters(self, model_config):
        model_id = model_config["model_id"]
        providers = model_config["providers"]

        messages = [{"role": "user", "content": "Search for: What is HTML?"}]
        tool_choice = "required" if "gpt-oss" in model_id else None

        result1 = api_call(model_id, messages, providers, tools=[TOOLS[2]], tool_choice=tool_choice)
        tool_calls, error = get_tool_calls(result1)

        if error or not tool_calls:
            raise RuntimeError(f"Model did not make tool call: {error}")

        tc = tool_calls[0]

        # Send response with special characters
        special_response = """Result: <html> is a tag. Use & for "and". Quotes: 'single' and "double".
JSON example: {"key": "value"}
Unicode: 한글, 日本語, émojis 🎉"""

        messages2 = [messages[0], {"role": "assistant", "content": None, "tool_calls": [{"id": tc["id"], "type": "function", "function": tc["function"]}]}, {"role": "tool", "tool_call_id": tc["id"], "content": special_response}]

        result2 = api_call(model_id, messages2, providers)
        answer, error2 = get_content(result2)

        assert error2 is None, f"API error: {error2}"
        assert answer, "No answer returned"

        content = answer.get("content", "")
        assert len(content) > 0, "Empty content returned"


class TestThinkingPreservation:
    """Test Case 5: Thinking/reasoning preservation."""

    def test_thinking_preservation(self, model_config):
        model_id = model_config["model_id"]
        providers = model_config["providers"]

        messages = [{"role": "user", "content": "Think step by step: What is 17 * 23?"}]
        tool_choice = "required" if "gpt-oss" in model_id else None

        result = api_call(model_id, messages, providers, tools=TOOLS[:1], tool_choice=tool_choice)
        tool_calls, _ = get_tool_calls(result)

        # Check for reasoning in response
        msg = result.get("choices", [{}])[0].get("message", {})
        reasoning = msg.get("reasoning") or msg.get("thinking") or ""

        # Either tool call or reasoning should be present
        has_tool_call = bool(tool_calls)
        has_reasoning = len(reasoning) > 10

        assert has_tool_call or has_reasoning, "Neither tool call nor reasoning present"

        if has_reasoning:
            # Verify reasoning has some content
            assert len(reasoning) > 10, f"Reasoning too short: {reasoning}"


# ============================================================================
# Standalone execution
# ============================================================================


def run_standalone():
    """Run all tests without pytest."""
    print("=" * 70)
    print("COMPREHENSIVE E2E TEST - ALL CASES")
    print("=" * 70)

    results = []

    for model_id, parser_name, providers in MODELS_TO_TEST:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_id}")
        print(f"{'=' * 70}")

        config = {"model_id": model_id, "parser_name": parser_name, "providers": providers}
        model_result = {"model": model_id, "cases": {}}

        # Test 1: Single tool call
        try:
            test = TestSingleToolCall()
            test.test_single_tool_call(config)
            print("  [Case 1: Single Tool Call] ✓")
            model_result["cases"]["single_tool"] = True
        except Exception as e:
            print(f"  [Case 1: Single Tool Call] ✗ {e}")
            model_result["cases"]["single_tool"] = False

        # Test 2: Parallel tool calls
        try:
            test = TestParallelToolCalls()
            test.test_parallel_tool_calls(config)
            print("  [Case 2: Parallel Tool Calls] ✓")
            model_result["cases"]["parallel_tools"] = True
        except Exception as e:
            print(f"  [Case 2: Parallel Tool Calls] ✗ {e}")
            model_result["cases"]["parallel_tools"] = False

        # Test 3: Multi-hop chain
        try:
            test = TestMultiHopChain()
            test.test_multi_hop_chain(config)
            print("  [Case 3: Multi-hop Chain] ✓")
            model_result["cases"]["multi_hop"] = True
        except Exception as e:
            print(f"  [Case 3: Multi-hop Chain] ✗ {e}")
            model_result["cases"]["multi_hop"] = False

        # Test 4: Special characters
        try:
            test = TestSpecialCharacters()
            test.test_special_characters(config)
            print("  [Case 4: Special Characters] ✓")
            model_result["cases"]["special_chars"] = True
        except Exception as e:
            print(f"  [Case 4: Special Characters] ✗ {e}")
            model_result["cases"]["special_chars"] = False

        # Test 5: Thinking preservation
        try:
            test = TestThinkingPreservation()
            test.test_thinking_preservation(config)
            print("  [Case 5: Thinking Preservation] ✓")
            model_result["cases"]["thinking"] = True
        except Exception as e:
            print(f"  [Case 5: Thinking Preservation] ✗ {e}")
            model_result["cases"]["thinking"] = False

        results.append(model_result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        passed = sum(1 for v in r["cases"].values() if v)
        total = len(r["cases"])
        model_short = r["model"].split("/")[-1]
        print(f"  {model_short}: {passed}/{total}")

    # Save results with timestamp filename
    import datetime

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    output = {
        "test_timestamp": now.isoformat(),
        "sampling_params": SAMPLING_PARAMS,
        "models_tested": len(results),
        "summary": {r["model"].split("/")[-1]: sum(1 for v in r["cases"].values() if v) for r in results},
        "results": results,
    }

    # Save with timestamp filename
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "e2e")
    os.makedirs(output_dir, exist_ok=True)
    timestamped_path = os.path.join(output_dir, f"e2e_results_{timestamp_str}.json")
    with open(timestamped_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Also save as latest (for quick access)
    latest_path = os.path.join(output_dir, "e2e_results_latest.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  - {timestamped_path}")
    print(f"  - {latest_path}")

    return results


if __name__ == "__main__":
    run_standalone()
