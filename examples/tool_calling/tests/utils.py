"""Shared test utilities for tool calling tests."""

from dataclasses import dataclass, field

import pytest

pytest.importorskip("transformers")


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def get_tokenizer(tokenizer_id: str):
    """Load tokenizer, skip test if not available."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer {tokenizer_id}: {e}")


def create_mock_tool_functions() -> dict[str, callable]:
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


