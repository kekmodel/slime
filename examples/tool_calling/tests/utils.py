"""Shared test utilities for tool calling tests."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("transformers")


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def get_tokenizer(tokenizer_id: str):
    """Load tokenizer, fail test if not available.

    Note: Prefer using get_tokenizer_for_parser() from conftest.py which
    provides caching and maps parser names to HF model IDs.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception as e:
        pytest.fail(
            f"Could not load tokenizer {tokenizer_id}: {e}\n"
            f"Try: huggingface-cli download {tokenizer_id}"
        )


def create_mock_tool_functions():
    """Create mock tool implementations."""

    def get_weather(city: str) -> dict:
        """Mock weather function."""
        weather_data = {
            "New York": {"weather": "Sunny", "temperature": "25°C"},
            "London": {"weather": "Cloudy", "temperature": "22°C"},
            "Paris": {"weather": "Rainy", "temperature": "18°C"},
        }
        return weather_data.get(city, {"weather": "Unknown", "temperature": "N/A"})

    def calculator(expression: str) -> str:
        """Safe calculator."""
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def web_search(query: str) -> dict:
        """Mock search function."""
        return {"results": [f"Result for: {query}"], "count": 1}

    return {
        "get_weather": get_weather,
        "calculator": calculator,
        "web_search": web_search,
    }


# ============================================================================
# Mock Response Classes
# ============================================================================


@dataclass
class MockGenerateResponse:
    """Mock /generate response matching SGLang format.

    The response contains output_token_logprobs which is the source of truth
    for RL training data. Format: [(logprob, token_id), ...]
    """

    text: str
    token_ids: list[int]
    log_probs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"

    def __post_init__(self):
        if not self.log_probs:
            self.log_probs = [-0.5] * len(self.token_ids)

    def to_dict(self) -> dict:
        """Convert to SGLang /generate response format."""
        output_token_logprobs = [[lp, tid] for lp, tid in zip(self.log_probs, self.token_ids)]
        return {
            "meta_info": {
                "output_token_logprobs": output_token_logprobs,
                "finish_reason": {"type": self.finish_reason},
            }
        }


def create_generate_response(tokenizer, text: str, finish_reason: str = "stop"):
    """Factory for creating mock generate responses."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return MockGenerateResponse(text=text, token_ids=token_ids, finish_reason=finish_reason)


# ============================================================================
# Tool Definitions for Testing
# ============================================================================


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information. Call this tool when the user needs to get weather information",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                }
            },
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Only basic arithmetic is supported.",
        "parameters": {
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 3 * 4'",
                }
            },
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
        },
    },
}


# ============================================================================
# Debug Helpers
# ============================================================================


def format_diff(expected: str, actual: str, context_chars: int = 50) -> str:
    """Show exact character differences with positions.

    Args:
        expected: Expected string
        actual: Actual string
        context_chars: Characters of context to show around diff

    Returns:
        Human-readable diff with position markers
    """
    if expected == actual:
        return "Strings are identical"

    # Find first difference
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            start = max(0, i - context_chars)
            end_exp = min(len(expected), i + context_chars)
            end_act = min(len(actual), i + context_chars)

            return (
                f"First difference at position {i}:\n"
                f"  Expected char: {repr(e)} (ord={ord(e)})\n"
                f"  Actual char:   {repr(a)} (ord={ord(a)})\n"
                f"  Expected context: ...{repr(expected[start:end_exp])}...\n"
                f"  Actual context:   ...{repr(actual[start:end_act])}..."
            )

    # Length difference
    if len(expected) != len(actual):
        return (
            f"Length mismatch: expected {len(expected)}, got {len(actual)}\n"
            f"  Expected ends with: {repr(expected[-context_chars:])}\n"
            f"  Actual ends with:   {repr(actual[-context_chars:])}"
        )

    return "Unknown difference"


def save_debug_info(
    parser_name: str,
    test_name: str,
    data: dict,
    output_dir: Path | None = None,
) -> Path:
    """Save full context to JSON for debugging failed tests.

    Args:
        parser_name: Name of the parser being tested
        test_name: Name of the test that failed
        data: Debug data to save (expected, actual, tokens, etc.)
        output_dir: Directory to save to (default: tests/outputs/debug/)

    Returns:
        Path to the saved debug file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{parser_name}_{test_name}_{timestamp}.json"
    filepath = output_dir / filename

    debug_data = {
        "parser_name": parser_name,
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        **data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)

    return filepath


