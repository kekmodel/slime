"""Ground truth tests: Our formatters vs HuggingFace apply_chat_template.

The definitive test: our formatter output must match EXACTLY what HuggingFace
produces when building the same conversation with apply_chat_template.

Key design:
- NO pytest.skip() - use pytest.fail() with debug info or pytest.xfail() for known issues
- Debug helpers save full context on failure
- Parametrized across parsers that have real HF templates
"""

import json

import pytest

pytest.importorskip("transformers")

from examples.tool_calling.tests.conftest import (
    PARSER_TO_HF_MODEL,
    get_tokenizer_for_parser,
)
from examples.tool_calling.tests.utils import (
    CALCULATOR_TOOL,
    WEATHER_TOOL,
    create_mock_tool_functions,
    format_diff,
    save_debug_info,
)
from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS


# Parsers that have real HF templates (not fallbacks to Qwen tokenizer)
# Tests will FAIL if our formatter doesn't match HF template - that's intended!
# Failures indicate the formatter needs fixing, not that the test should be skipped.
PARSERS_WITH_HF_TEMPLATES = ["qwen", "qwen25", "qwen3_coder"]


class TestToolResponseGroundTruth:
    """Test that our formatter output matches HF template exactly."""

    @pytest.fixture(params=PARSERS_WITH_HF_TEMPLATES)
    def parser_name(self, request) -> str:
        return request.param

    def test_single_tool_response_matches_hf(self, parser_name: str):
        """Our formatter output must match HF apply_chat_template exactly."""
        tokenizer = get_tokenizer_for_parser(parser_name, use_small=False)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]
        tool_map = create_mock_tool_functions()

        tool_result = json.dumps(tool_map["get_weather"]("New York"))
        tool_call_id = "call_0"

        # Build HF conversation
        messages_before = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
                    }
                ],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "tool_call_id": tool_call_id, "content": tool_result}]

        tools = [WEATHER_TOOL]

        # Get HF output
        before_text = tokenizer.apply_chat_template(
            messages_before, tools=tools, tokenize=False, add_generation_prompt=False
        )
        after_text = tokenizer.apply_chat_template(
            messages_after, tools=tools, tokenize=False, add_generation_prompt=True
        )
        hf_tool_response = after_text[len(before_text) :]

        # Get our formatter output
        kwargs = {"content": tool_result, "add_generation_prompt": True}
        if parser_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = tool_call_id
        if parser_name == "gpt-oss":
            kwargs["tool_name"] = "get_weather"

        our_tool_response = formatter(**kwargs)

        # Compare
        if our_tool_response != hf_tool_response:
            debug_path = save_debug_info(
                parser_name=parser_name,
                test_name="single_tool_response",
                data={
                    "expected": hf_tool_response,
                    "actual": our_tool_response,
                    "diff": format_diff(hf_tool_response, our_tool_response),
                    "messages_before": messages_before,
                    "messages_after": messages_after,
                },
            )
            pytest.fail(
                f"Formatter output doesn't match HF template!\n"
                f"{format_diff(hf_tool_response, our_tool_response)}\n"
                f"Debug info saved to: {debug_path}"
            )

    def test_token_level_match(self, parser_name: str):
        """Token-level verification: our tokens == HF tokens."""
        tokenizer = get_tokenizer_for_parser(parser_name, use_small=False)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        content = '{"result": 42}'
        tool_call_id = "call_0"

        # Build HF conversation
        messages_before = [
            {"role": "user", "content": "Calculate 6*7"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'},
                    }
                ],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "tool_call_id": tool_call_id, "content": content}]

        tools = [CALCULATOR_TOOL]

        before_text = tokenizer.apply_chat_template(
            messages_before, tools=tools, tokenize=False, add_generation_prompt=False
        )
        after_text = tokenizer.apply_chat_template(
            messages_after, tools=tools, tokenize=False, add_generation_prompt=True
        )

        hf_tool_response = after_text[len(before_text) :]
        hf_tokens = tokenizer.encode(hf_tool_response, add_special_tokens=False)

        # Our output
        kwargs = {"content": content, "add_generation_prompt": True}
        if parser_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = tool_call_id
        if parser_name == "gpt-oss":
            kwargs["tool_name"] = "calculator"

        our_tool_response = formatter(**kwargs)
        our_tokens = tokenizer.encode(our_tool_response, add_special_tokens=False)

        if hf_tokens != our_tokens:
            debug_path = save_debug_info(
                parser_name=parser_name,
                test_name="token_level_match",
                data={
                    "hf_tokens": hf_tokens,
                    "our_tokens": our_tokens,
                    "hf_text": hf_tool_response,
                    "our_text": our_tool_response,
                    "diff": format_diff(hf_tool_response, our_tool_response),
                },
            )
            pytest.fail(
                f"Token mismatch!\n"
                f"  HF:   {len(hf_tokens)} tokens\n"
                f"  Ours: {len(our_tokens)} tokens\n"
                f"  Debug info: {debug_path}"
            )


class TestMultiTurnGroundTruth:
    """Test multi-turn tool calling accumulation."""

    @pytest.fixture(params=PARSERS_WITH_HF_TEMPLATES)
    def parser_name(self, request) -> str:
        return request.param

    def test_two_turn_tool_calling(self, parser_name: str):
        """Multi-hop: model calls tool twice, then answers."""
        tokenizer = get_tokenizer_for_parser(parser_name, use_small=False)
        tool_map = create_mock_tool_functions()

        messages = [
            {"role": "user", "content": "Weather in NY and London?"},
            {
                "role": "assistant",
                "content": "Checking NY first.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(tool_map["get_weather"]("New York"))},
            {
                "role": "assistant",
                "content": "Now London.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(tool_map["get_weather"]("London"))},
            {"role": "assistant", "content": "NY: Sunny 25°C, London: Cloudy 22°C."},
        ]

        tools = [WEATHER_TOOL]

        # Ground truth
        ground_truth = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)

        # Incremental accumulation (simulating our training flow)
        accumulated_tokens: list[int] = []
        current_text_len = 0

        for i in range(len(messages)):
            partial = messages[: i + 1]
            add_gen = i < len(messages) - 1 and messages[i]["role"] == "tool"

            partial_text = tokenizer.apply_chat_template(
                partial, tools=tools, tokenize=False, add_generation_prompt=add_gen
            )

            new_text = partial_text[current_text_len:]
            if new_text:
                new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
                accumulated_tokens.extend(new_tokens)
                current_text_len = len(partial_text)

        if accumulated_tokens != ground_truth_tokens:
            debug_path = save_debug_info(
                parser_name=parser_name,
                test_name="two_turn_tool_calling",
                data={
                    "ground_truth_len": len(ground_truth_tokens),
                    "accumulated_len": len(accumulated_tokens),
                    "messages": messages,
                },
            )
            pytest.fail(
                f"Multi-turn accumulation mismatch!\n"
                f"  Ground truth: {len(ground_truth_tokens)} tokens\n"
                f"  Accumulated:  {len(accumulated_tokens)} tokens\n"
                f"  Debug info: {debug_path}"
            )


class TestEdgeCaseGroundTruth:
    """Test edge cases in ground truth comparison."""

    @pytest.fixture(params=PARSERS_WITH_HF_TEMPLATES)
    def parser_name(self, request) -> str:
        return request.param

    def test_special_characters_in_content(self, parser_name: str):
        """Tool results with special characters must round-trip correctly."""
        tokenizer = get_tokenizer_for_parser(parser_name, use_small=False)

        # Content with various special characters
        special_content = '{"html": "<div>test</div>", "quote": "He said \\"hi\\"", "unicode": "日本語"}'
        tool_call_id = "call_special"

        messages = [
            {"role": "user", "content": "Test special chars"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_call_id, "content": special_content},
        ]

        tools = [CALCULATOR_TOOL]

        # Should not raise and should contain the content
        result = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        assert "日本語" in result, f"{parser_name}: Unicode content lost in template"
        assert "div" in result, f"{parser_name}: HTML content lost in template"
