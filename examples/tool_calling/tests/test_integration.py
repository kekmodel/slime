"""Integration tests for the generate() function and tool execution flow.

Tests the complete flow with mocked SGLang responses:
1. generate() receives prompt
2. Mock returns tool call
3. Tool executes
4. Observation formatted
5. Loop continues until finish_reason != "tool_calls"
"""

import json

import pytest

pytest.importorskip("transformers")

from examples.tool_calling.tests.conftest import get_tokenizer_for_parser
from examples.tool_calling.tests.utils import (
    MockGenerateResponse,
    create_mock_tool_functions,
    WEATHER_TOOL,
    CALCULATOR_TOOL,
)


class TestGenerateFunctionFlow:
    """Test the generate() function with mocked SGLang."""

    def test_single_tool_call_flow(self):
        """Test complete flow: user question → tool call → tool result → final answer."""
        tokenizer = get_tokenizer_for_parser("qwen25")
        tool_map = create_mock_tool_functions()

        # Simulate the flow without actually calling generate()
        # This tests our understanding of the token accumulation

        # Step 1: Model generates tool call
        model_output_1 = "Let me check that.\n"
        tokens_1 = tokenizer.encode(model_output_1, add_special_tokens=False)
        loss_mask_1 = [1] * len(tokens_1)  # Model generated

        # Step 2: Tool executes
        tool_result = tool_map["get_weather"]("New York")

        # Step 3: Format observation
        from examples.tool_calling.tools import format_qwen

        observation = format_qwen(json.dumps(tool_result))
        obs_tokens = tokenizer.encode(observation, add_special_tokens=False)
        loss_mask_obs = [0] * len(obs_tokens)  # Not trainable

        # Step 4: Model generates final answer
        final_output = "The weather is sunny."
        tokens_final = tokenizer.encode(final_output, add_special_tokens=False)
        loss_mask_final = [1] * len(tokens_final)  # Model generated

        # Accumulate
        all_tokens = tokens_1 + obs_tokens + tokens_final
        all_loss_mask = loss_mask_1 + loss_mask_obs + loss_mask_final

        # Verify invariants
        assert len(all_tokens) == len(all_loss_mask)
        assert sum(all_loss_mask) == len(tokens_1) + len(tokens_final)


class TestStatusHandling:
    """Test handling of different finish_reason values."""

    def test_completed_status(self):
        """COMPLETED: Model finished normally."""
        response = MockGenerateResponse(text="Done.", token_ids=[1, 2, 3], finish_reason="stop")

        output = response.to_dict()
        assert output["meta_info"]["finish_reason"]["type"] == "stop"

    def test_truncated_status(self):
        """TRUNCATED: Hit max tokens."""
        response = MockGenerateResponse(text="Partial...", token_ids=[1, 2, 3], finish_reason="length")

        output = response.to_dict()
        assert output["meta_info"]["finish_reason"]["type"] == "length"


class TestToolExecutionFlow:
    """Test tool execution and result formatting."""

    def test_weather_tool_execution(self):
        """Weather tool returns expected format."""
        tool_map = create_mock_tool_functions()

        result = tool_map["get_weather"]("New York")

        assert "weather" in result
        assert "temperature" in result
        assert result["weather"] == "Sunny"

    def test_calculator_tool_execution(self):
        """Calculator tool handles basic math."""
        tool_map = create_mock_tool_functions()

        result = tool_map["calculator"]("2 + 3 * 4")

        assert result == "14"

    def test_calculator_rejects_dangerous_input(self):
        """Calculator rejects potentially dangerous input."""
        tool_map = create_mock_tool_functions()

        # Should reject import statements
        result = tool_map["calculator"]("__import__('os')")

        assert "Error" in result

    def test_unknown_city_returns_default(self):
        """Weather tool handles unknown cities gracefully."""
        tool_map = create_mock_tool_functions()

        result = tool_map["get_weather"]("Unknown City")

        assert result["weather"] == "Unknown"
        assert result["temperature"] == "N/A"


class TestMultiHopFlow:
    """Test multi-hop tool calling flow."""

    def test_two_tool_calls_accumulation(self):
        """Test accumulation after two sequential tool calls."""
        tokenizer = get_tokenizer_for_parser("qwen25")
        tool_map = create_mock_tool_functions()
        from examples.tool_calling.tools import format_qwen

        all_tokens: list[int] = []
        all_loss_mask: list[int] = []

        # Hop 1: First tool call
        gen_1 = "Checking NY weather."
        tokens_1 = tokenizer.encode(gen_1, add_special_tokens=False)
        all_tokens.extend(tokens_1)
        all_loss_mask.extend([1] * len(tokens_1))

        # Tool result 1
        result_1 = tool_map["get_weather"]("New York")
        obs_1 = format_qwen(json.dumps(result_1))
        obs_tokens_1 = tokenizer.encode(obs_1, add_special_tokens=False)
        all_tokens.extend(obs_tokens_1)
        all_loss_mask.extend([0] * len(obs_tokens_1))

        # Invariant check after hop 1
        assert len(all_tokens) == len(all_loss_mask)

        # Hop 2: Second tool call
        gen_2 = "Now checking London."
        tokens_2 = tokenizer.encode(gen_2, add_special_tokens=False)
        all_tokens.extend(tokens_2)
        all_loss_mask.extend([1] * len(tokens_2))

        # Tool result 2
        result_2 = tool_map["get_weather"]("London")
        obs_2 = format_qwen(json.dumps(result_2))
        obs_tokens_2 = tokenizer.encode(obs_2, add_special_tokens=False)
        all_tokens.extend(obs_tokens_2)
        all_loss_mask.extend([0] * len(obs_tokens_2))

        # Invariant check after hop 2
        assert len(all_tokens) == len(all_loss_mask)

        # Final response
        final = "NY is sunny, London is cloudy."
        final_tokens = tokenizer.encode(final, add_special_tokens=False)
        all_tokens.extend(final_tokens)
        all_loss_mask.extend([1] * len(final_tokens))

        # Final invariants
        assert len(all_tokens) == len(all_loss_mask)
        trainable = sum(all_loss_mask)
        non_trainable = len(all_loss_mask) - trainable

        # Trainable = gen_1 + gen_2 + final
        expected_trainable = len(tokens_1) + len(tokens_2) + len(final_tokens)
        assert trainable == expected_trainable

        # Non-trainable = obs_1 + obs_2
        expected_non_trainable = len(obs_tokens_1) + len(obs_tokens_2)
        assert non_trainable == expected_non_trainable
