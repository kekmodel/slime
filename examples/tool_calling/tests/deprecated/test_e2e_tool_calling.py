"""
E2E tests for tool calling with realistic scenarios.

Based on real-world examples from:
- https://docs.z.ai/guides/capabilities/function-calling
- https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/docs/tool_call_guidance.md
- SGLang FunctionCallParser and ReasoningParser source code

Tests the complete token flow for RL training:
    token_ids (from logprobs) → decode → parse tool calls
    → execute tools → format observation → encode → observation_token_ids

Critical invariants:
- len(token_ids) == len(loss_mask) == len(log_probs)
- loss_mask = 1 for model-generated tokens (including reasoning), 0 for observations
- Token round-trip: decode(encode(text)) preserves meaning

================================================================================
REASONING/THINKING TOKEN HANDLING SUMMARY
================================================================================
KEY INSIGHT: behavior differs between TOOL CALL context and REGULAR CHAT!

1. INPUT SIDE (HuggingFace Chat Templates)
   Each model's template decides how to render reasoning_content in history:

   | Model            | Tool Call Context | Regular Chat    | Gen Prompt   |
   |------------------|-------------------|-----------------|--------------|
   | Qwen2.5          | No support        | No support      | No thinking  |
   | Qwen3/Qwen3-Next | ALL preserved     | Last only       | No thinking  |
   | GLM-4.7          | ALL preserved     | Last only       | <think>      |
   | Kimi-K2-Thinking | ALL preserved     | Empty <think>   | No thinking  |
   | DeepSeek-R1      | Stripped          | Stripped        | <think>\\n   |
   | DeepSeek-V3.2    | SGLang handles    | Stripped        | No thinking  |

2. OUTPUT SIDE (SGLang ReasoningParser)
   Parses model generation to extract reasoning vs content:
   - "deepseek-r1": DeepSeekR1Detector (<think>...</think>, force=True)
   - "qwen3": Qwen3Detector (<think>...</think>, force=False)
   - "kimi_k2": Qwen3Detector (<think>...</think>, force=False)
   - "kimi": KimiDetector (◁think▷...◁/think▷, force=False)
   - "gpt-oss": GptOssDetector (<|channel|>analysis...<|end|>, force=True)

3. RL TRAINING IMPLICATIONS
   For multi-hop tool calling:
   - Qwen3/Kimi-K2: Reasoning accumulates in context (model sees history)
   - DeepSeek: Each hop starts fresh (no history reasoning)
   - loss_mask=1 for ALL model-generated tokens (including reasoning)
   - loss_mask=0 for observations (tool results)

TOOL RESPONSE FORMATS (verified against HuggingFace templates):
- Qwen/Qwen3:     <|im_start|>user\\n<tool_response>\\n{content}\\n</tool_response><|im_end|>\\n<|im_start|>assistant\\n
- GLM-4:          <|observation|>\\n{content}<|assistant|>
- GLM-4.7:        <|observation|><tool_response>{content}</tool_response><|assistant|><think>
- DeepSeek-V3/R1: <｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{content}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>
- Kimi-K2:        <|im_system|>tool<|im_middle|>## Return of {id}\\n{content}<|im_end|><|im_assistant|>assistant<|im_middle|>
================================================================================
"""

import asyncio
import json
import pytest
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

pytest.importorskip("transformers")


# ============================================================================
# Realistic Tool Definitions (from documentation)
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
# Test Fixtures and Helpers
# ============================================================================


def get_tokenizer(tokenizer_id: str):
    """Load tokenizer, skip test if not available."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer {tokenizer_id}: {e}")


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


@dataclass
class MockGenerateResponse:
    """
    Mock /generate response matching SGLang format.

    The response contains output_token_logprobs which is the source of truth
    for RL training data. Format: [(logprob, token_id), ...]
    """

    text: str
    token_ids: List[int]
    log_probs: List[float] = field(default_factory=list)
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


def create_generate_response(text: str, tokenizer, finish_reason: str = "stop") -> dict:
    """Create a mock /generate response with logprobs."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return MockGenerateResponse(text=text, token_ids=token_ids, finish_reason=finish_reason).to_dict()


# ============================================================================
# Test Classes
# ============================================================================


class TestRealisticWeatherQuery:
    """
    Test realistic weather query scenario from Kimi K2 documentation.

    Flow:
    1. User asks "What's the weather like in New York?"
    2. Model generates tool call
    3. Tool executes, returns {"weather": "Sunny"}
    4. Model generates final response

    This tests the exact flow described in:
    https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/docs/tool_call_guidance.md
    """

    @pytest.fixture
    def tools(self):
        return [WEATHER_TOOL]

    @pytest.fixture
    def tool_map(self):
        return create_mock_tool_functions()

    def test_single_turn_weather_query_token_flow(self, tools, tool_map):
        """Test token flow for single-turn weather query."""
        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

        # Simulate the flow
        prompt = "What's the weather like in New York today?"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        # Model generates tool call (simulated output)
        model_output = "Let me check the weather for you.\n"
        model_tokens = tokenizer.encode(model_output, add_special_tokens=False)

        # Execute tool
        tool_result = tool_map["get_weather"]("New York")
        tool_result_json = json.dumps(tool_result, ensure_ascii=False)

        # Format tool response (Qwen format)
        from examples.tool_calling.tools import format_qwen

        observation = format_qwen(tool_result_json)
        observation_tokens = tokenizer.encode(observation, add_special_tokens=False)

        # Model continues with final response
        final_response = "The weather in New York is Sunny with temperature 25°C."
        final_tokens = tokenizer.encode(final_response, add_special_tokens=False)

        # Build training data
        all_response_tokens = model_tokens + observation_tokens + final_tokens
        loss_mask = (
            [1] * len(model_tokens)  # Generated - trainable
            + [0] * len(observation_tokens)  # Observation - not trainable
            + [1] * len(final_tokens)  # Generated - trainable
        )

        # Verify invariants
        assert len(all_response_tokens) == len(loss_mask)
        assert sum(loss_mask) == len(model_tokens) + len(final_tokens)

        # Verify round-trip
        decoded = tokenizer.decode(all_response_tokens, skip_special_tokens=False)
        assert "weather" in decoded.lower()
        assert "Sunny" in decoded or "sunny" in decoded.lower()

    def test_multi_turn_weather_loop(self, tools, tool_map):
        """
        Test multi-turn loop until finish_reason != "tool_calls".

        This matches the pattern from documentation:
        while finish_reason is None or finish_reason == "tool_calls":
            ...
        """
        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        from examples.tool_calling.tools import format_qwen

        # Simulate conversation state
        messages = [{"role": "user", "content": "What's the weather in New York and London?"}]

        all_response_tokens = []
        all_loss_mask = []
        all_log_probs = []

        # Turn 1: Model requests New York weather
        turn1_output = "I'll check both cities. First, New York."
        turn1_tokens = tokenizer.encode(turn1_output, add_special_tokens=False)
        turn1_logprobs = [-0.3] * len(turn1_tokens)

        all_response_tokens.extend(turn1_tokens)
        all_loss_mask.extend([1] * len(turn1_tokens))
        all_log_probs.extend(turn1_logprobs)

        # Tool response 1
        result1 = tool_map["get_weather"]("New York")
        obs1 = format_qwen(json.dumps(result1))
        obs1_tokens = tokenizer.encode(obs1, add_special_tokens=False)

        all_response_tokens.extend(obs1_tokens)
        all_loss_mask.extend([0] * len(obs1_tokens))
        all_log_probs.extend([0.0] * len(obs1_tokens))  # Dummy for observations

        # Turn 2: Model requests London weather
        turn2_output = "Now checking London."
        turn2_tokens = tokenizer.encode(turn2_output, add_special_tokens=False)
        turn2_logprobs = [-0.4] * len(turn2_tokens)

        all_response_tokens.extend(turn2_tokens)
        all_loss_mask.extend([1] * len(turn2_tokens))
        all_log_probs.extend(turn2_logprobs)

        # Tool response 2
        result2 = tool_map["get_weather"]("London")
        obs2 = format_qwen(json.dumps(result2))
        obs2_tokens = tokenizer.encode(obs2, add_special_tokens=False)

        all_response_tokens.extend(obs2_tokens)
        all_loss_mask.extend([0] * len(obs2_tokens))
        all_log_probs.extend([0.0] * len(obs2_tokens))

        # Turn 3: Final answer (finish_reason = "stop")
        turn3_output = "New York: Sunny 25°C. London: Cloudy 22°C."
        turn3_tokens = tokenizer.encode(turn3_output, add_special_tokens=False)
        turn3_logprobs = [-0.2] * len(turn3_tokens)

        all_response_tokens.extend(turn3_tokens)
        all_loss_mask.extend([1] * len(turn3_tokens))
        all_log_probs.extend(turn3_logprobs)

        # Verify all invariants
        assert len(all_response_tokens) == len(all_loss_mask) == len(all_log_probs)

        # Count trainable vs non-trainable
        trainable_count = sum(all_loss_mask)
        non_trainable_count = len(all_loss_mask) - trainable_count

        expected_trainable = len(turn1_tokens) + len(turn2_tokens) + len(turn3_tokens)
        expected_non_trainable = len(obs1_tokens) + len(obs2_tokens)

        assert trainable_count == expected_trainable
        assert non_trainable_count == expected_non_trainable


class TestKimiK2ToolCallIdFormat:
    """
    Test Kimi K2's specific tool_call_id format requirements.

    From documentation:
    - ID format: functions.{func_name}:{idx}
    - idx is a global counter starting at 0
    - Incorrect ID causes tool-call crash
    """

    def test_correct_tool_call_id_format(self):
        """Test correct tool_call_id format generation."""
        from examples.tool_calling.tools import format_kimi_k2

        # First call: functions.get_weather:0
        result = format_kimi_k2(
            content='{"weather": "Sunny"}',
            tool_call_id="functions.get_weather:0",
            add_generation_prompt=True,
        )

        assert "functions.get_weather:0" in result
        # Role is always 'tool', not the function name
        assert "<|im_system|>tool<|im_middle|>" in result
        assert "## Return of functions.get_weather:0" in result
        # Generation prompt suffix
        assert "<|im_assistant|>assistant<|im_middle|>" in result

    def test_multiple_tool_calls_incrementing_idx(self):
        """Test that tool_call_id idx increments correctly."""
        from examples.tool_calling.tools import format_kimi_k2

        tool_call_ids = [
            "functions.get_weather:0",
            "functions.calculator:1",
            "functions.get_weather:2",  # Same function, different idx
        ]

        for expected_id in tool_call_ids:
            result = format_kimi_k2(
                content="test",
                tool_call_id=expected_id,
            )
            assert expected_id in result
            assert f"## Return of {expected_id}" in result
            assert "<|im_system|>tool<|im_middle|>" in result

    def test_tool_call_id_parsing_regex(self):
        """Test regex for parsing tool_call_id from model output."""
        import re

        # Regex from Kimi K2 documentation
        tool_call_id_regex = re.compile(r"^(?:functions\.)?(?P<name>[\w\.]+):(?P<index>\d+)$")

        test_cases = [
            ("functions.get_weather:0", "get_weather", 0),
            ("functions.calculator:1", "calculator", 1),
            ("functions.web_search:10", "web_search", 10),
            ("get_weather:0", "get_weather", 0),  # Without functions. prefix
        ]

        for tool_call_id, expected_name, expected_idx in test_cases:
            match = tool_call_id_regex.match(tool_call_id)
            assert match is not None, f"Failed to match: {tool_call_id}"
            assert match.group("name") == expected_name
            assert int(match.group("index")) == expected_idx


class TestRealisticToolCallIdFormats:
    """
    Test realistic tool_call_id formats from live OpenRouter API responses.

    Real API responses use various ID formats:
    - Qwen: call_703c3d5ebf084ba3a56b31 (hash-based)
    - GLM-4.7: call_b312886f6d874a899ab61aaf (hash-based)
    - Kimi-K2: functions.calculator:0 (indexed)
    - DeepSeek: 019c16d80729bba4c164d9aa0488f381 (UUID-like)
    - MiniMax: call_function_cnfru7c4vcaw_1 (custom)
    - GPT-OSS: functions.calculator_aad4 (with suffix)

    These IDs are passed through our formatters and must be preserved exactly.
    """

    def test_qwen_realistic_id_format(self):
        """Test Qwen hash-based ID format."""
        from examples.tool_calling.tools import format_qwen

        realistic_id = "call_703c3d5ebf084ba3a56b31"
        result = format_qwen(content='{"result": 42}')

        # Qwen formatter doesn't include tool_call_id in output
        assert "<tool_response>" in result
        assert "42" in result

    def test_kimi_k2_realistic_id_format(self):
        """Test Kimi-K2 indexed ID format from live API."""
        from examples.tool_calling.tools import format_kimi_k2

        # Real format from OpenRouter API
        realistic_ids = [
            "functions.calculator:0",
            "functions.get_weather:1",
            " functions.calculator:0 ",  # With spaces (seen in some responses)
        ]

        for tool_call_id in realistic_ids:
            result = format_kimi_k2(
                content='{"result": 42}',
                tool_call_id=tool_call_id.strip(),
            )
            assert f"## Return of {tool_call_id.strip()}" in result

    def test_deepseek_realistic_id_format(self):
        """Test DeepSeek UUID-like ID format."""
        from examples.tool_calling.tools import format_deepseek_v3

        realistic_id = "019c16d80729bba4c164d9aa0488f381"
        result = format_deepseek_v3(content='{"result": 42}')

        # DeepSeek formatter doesn't include tool_call_id in output
        assert "<｜tool▁output▁begin｜>" in result
        assert "42" in result

    def test_gpt_oss_realistic_id_format(self):
        """Test GPT-OSS ID format with suffix."""
        from examples.tool_calling.tools import format_gpt_oss

        result = format_gpt_oss(
            content="42",
            tool_name="calculator",
        )

        # GPT-OSS uses tool_name, not tool_call_id
        assert "functions.calculator" in result
        assert "42" in result

    def test_minimax_realistic_id_format(self):
        """Test MiniMax custom ID format."""
        from examples.tool_calling.tools import format_minimax

        result = format_minimax(content='{"result": 42}')

        # MiniMax formatter doesn't include tool_call_id
        assert "<response>" in result
        assert "42" in result

    def test_all_formatters_preserve_content_exactly(self):
        """Test that all formatters preserve content without modification."""
        from examples.tool_calling.tools import (
            format_qwen,
            format_glm47,
            format_kimi_k2,
            format_deepseek_v3,
            format_deepseek_v32,
            format_minimax,
            format_gpt_oss,
        )

        # Content with special characters
        content = '{"html": "<div>test</div>", "unicode": "한글日本語", "quotes": "\\"nested\\""}'

        # All formatters should preserve content
        assert content in format_qwen(content)
        assert content in format_glm47(content)
        assert content in format_kimi_k2(content, tool_call_id="functions.test:0")
        assert content in format_deepseek_v3(content)
        assert content in format_deepseek_v32(content)
        assert content in format_minimax(content)
        # GPT-OSS JSON-encodes content
        assert "html" in format_gpt_oss(content, tool_name="test")


class TestTokenFlowAlignment:
    """
    Test critical token flow alignment for RL training.

    The key invariant:
        len(token_ids) == len(loss_mask) == len(log_probs)

    This must hold at every step of multi-hop tool calling.
    """

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    def test_single_hop_alignment(self, tokenizer):
        """Test alignment after single tool call hop."""
        from examples.tool_calling.tools import format_qwen

        # Generation
        gen_text = "Calculating 2+3..."
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
        gen_logprobs = [-0.5] * len(gen_tokens)
        gen_loss_mask = [1] * len(gen_tokens)

        # Observation
        obs_text = format_qwen("5")
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
        obs_logprobs = [0.0] * len(obs_tokens)  # Dummy
        obs_loss_mask = [0] * len(obs_tokens)

        # Combine
        all_tokens = gen_tokens + obs_tokens
        all_logprobs = gen_logprobs + obs_logprobs
        all_loss_mask = gen_loss_mask + obs_loss_mask

        # Verify alignment
        assert len(all_tokens) == len(all_logprobs) == len(all_loss_mask)

    def test_multi_hop_alignment_with_assertions(self, tokenizer):
        """Test alignment with explicit assertions at each step."""
        from examples.tool_calling.tools import format_qwen

        all_tokens: List[int] = []
        all_logprobs: List[float] = []
        all_loss_mask: List[int] = []

        for hop in range(3):
            # Generation phase
            gen_text = f"Step {hop + 1}..."
            gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)

            all_tokens.extend(gen_tokens)
            all_logprobs.extend([-0.5] * len(gen_tokens))
            all_loss_mask.extend([1] * len(gen_tokens))

            # Assert after generation
            assert len(all_tokens) == len(all_logprobs) == len(all_loss_mask), f"Hop {hop} post-gen: lengths don't match: tokens={len(all_tokens)}, logprobs={len(all_logprobs)}, loss_mask={len(all_loss_mask)}"

            # Observation phase (except last hop)
            if hop < 2:
                obs_text = format_qwen(f"Result {hop + 1}")
                obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)

                all_tokens.extend(obs_tokens)
                all_logprobs.extend([0.0] * len(obs_tokens))
                all_loss_mask.extend([0] * len(obs_tokens))

                # Assert after observation
                assert len(all_tokens) == len(all_logprobs) == len(all_loss_mask), f"Hop {hop} post-obs: lengths don't match: tokens={len(all_tokens)}, logprobs={len(all_logprobs)}, loss_mask={len(all_loss_mask)}"

        # Final verification
        assert all(m in (0, 1) for m in all_loss_mask), "Invalid loss_mask values"

    def test_token_decode_encode_roundtrip(self, tokenizer):
        """Test that decode(token_ids) produces parseable text."""
        from examples.tool_calling.tools import format_qwen

        # Create tokens from multiple sources
        gen_text = "Let me calculate that."
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)

        obs_text = format_qwen('{"result": 42}')
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)

        final_text = "The answer is 42."
        final_tokens = tokenizer.encode(final_text, add_special_tokens=False)

        # Combine and decode
        all_tokens = gen_tokens + obs_tokens + final_tokens
        decoded = tokenizer.decode(all_tokens, skip_special_tokens=False)

        # Verify key content preserved
        assert "calculate" in decoded.lower()
        assert "42" in decoded
        assert "answer" in decoded.lower()

        # Verify tool response markers present
        assert "<tool_response>" in decoded or "<|im_start|>" in decoded


class TestFormatterTokenAlignment:
    """
    Test that formatters produce output that tokenizes consistently.

    Key: Our formatter output must tokenize identically to HuggingFace
    apply_chat_template output for the same content.
    """

    def test_qwen_formatter_token_alignment(self):
        """Verify Qwen formatter matches HF template at token level."""
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

        tool_result = '{"weather": "Sunny", "temperature": "25°C"}'
        tool_call_id = "call_0"

        # Build HF conversation
        messages_before = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tool_call_id, "function": {"name": "get_weather", "arguments": {"city": "New York"}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "tool_call_id": tool_call_id, "content": tool_result}]

        tools = [WEATHER_TOOL]

        before_text = tokenizer.apply_chat_template(messages_before, tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=tools, tokenize=False, add_generation_prompt=True)

        # Extract the tool response section
        hf_tool_response = after_text[len(before_text) :]

        # Our formatter output (add_generation_prompt=True to match HF)
        our_tool_response = format_qwen(tool_result, add_generation_prompt=True)

        # String match
        assert our_tool_response == hf_tool_response, f"Format mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

        # Token match
        hf_tokens = tokenizer.encode(hf_tool_response, add_special_tokens=False)
        our_tokens = tokenizer.encode(our_tool_response, add_special_tokens=False)

        assert hf_tokens == our_tokens, f"Token mismatch:\n  HF:   {hf_tokens}\n  Ours: {our_tokens}"

    def test_glm_formatter_token_alignment(self):
        """Verify GLM formatter matches HF template at token level."""
        from examples.tool_calling.tools import format_glm

        tokenizer = get_tokenizer("/tmp/glm4-full")

        tool_result = "Sunny, 25°C"

        # Build HF conversation
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "observation", "content": tool_result},
        ]

        hf_output = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Extract observation section
        import re

        match = re.search(r"<\|observation\|>.*?<\|assistant\|>", hf_output, re.DOTALL)
        assert match, f"Could not find observation section in: {hf_output}"

        hf_observation = match.group()
        our_observation = format_glm(tool_result, add_generation_prompt=True)

        assert our_observation == hf_observation, f"Format mismatch:\n  HF:   {repr(hf_observation)}\n  Ours: {repr(our_observation)}"


class TestMessagesHistoryGroundTruth:
    """
    Ground truth tests using messages history + apply_chat_template.

    This is the definitive test: build the complete conversation as messages,
    use apply_chat_template to get the expected output, then verify our
    token accumulation produces the exact same result.

    Based on the pattern from Kimi K2 documentation:
    ```python
    messages = [{"role": "user", "content": "..."}]
    while finish_reason == "tool_calls":
        messages.append(choice.message)  # assistant with tool_calls
        for tool_call in tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call_name,
                "content": json.dumps(tool_result),
            })
    ```
    """

    def test_single_tool_call_ground_truth_qwen(self):
        """
        Test single tool call against HuggingFace ground truth.

        Scenario: User asks weather → Model calls tool → Tool responds → Model answers
        """
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        tools = [WEATHER_TOOL]
        tool_map = create_mock_tool_functions()

        # ====================================================================
        # Step 1: Build complete messages history (ground truth)
        # ====================================================================
        messages = [
            {"role": "user", "content": "What's the weather in New York?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "New York"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": json.dumps(tool_map["get_weather"]("New York")),
            },
            {
                "role": "assistant",
                "content": "The weather in New York is Sunny with temperature 25°C.",
            },
        ]

        # Get ground truth from HuggingFace
        ground_truth_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        ground_truth_tokens = tokenizer.encode(ground_truth_text, add_special_tokens=False)

        # ====================================================================
        # Step 2: Simulate our token accumulation flow
        # ====================================================================
        # Get prompt (user message only)
        prompt_messages = [messages[0]]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tools=tools, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Get assistant's first turn (with tool call)
        assistant_turn1_messages = messages[:2]
        assistant_turn1_text = tokenizer.apply_chat_template(assistant_turn1_messages, tools=tools, tokenize=False, add_generation_prompt=False)
        # Extract just the assistant part
        assistant_turn1_response = assistant_turn1_text[len(prompt_text) :]
        assistant_turn1_tokens = tokenizer.encode(assistant_turn1_response, add_special_tokens=False)

        # Tool response (our formatter with add_generation_prompt=True to match HF)
        tool_result = json.dumps(tool_map["get_weather"]("New York"))
        our_tool_response = format_qwen(tool_result, add_generation_prompt=True)
        our_tool_response_tokens = tokenizer.encode(our_tool_response, add_special_tokens=False)

        # Get HF's tool response for comparison
        with_tool_messages = messages[:3]
        with_tool_text = tokenizer.apply_chat_template(with_tool_messages, tools=tools, tokenize=False, add_generation_prompt=True)
        hf_tool_response = with_tool_text[len(assistant_turn1_text) :]
        hf_tool_response_tokens = tokenizer.encode(hf_tool_response, add_special_tokens=False)

        # Verify tool response matches
        assert our_tool_response == hf_tool_response, f"Tool response mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"
        assert our_tool_response_tokens == hf_tool_response_tokens

        # Final assistant response
        final_messages = messages[:4]
        final_text = tokenizer.apply_chat_template(final_messages, tools=tools, tokenize=False, add_generation_prompt=False)
        final_response = final_text[len(with_tool_text) :]
        final_response_tokens = tokenizer.encode(final_response, add_special_tokens=False)

        # ====================================================================
        # Step 3: Accumulate and verify
        # ====================================================================
        accumulated_tokens = prompt_tokens + assistant_turn1_tokens + our_tool_response_tokens + final_response_tokens

        # Ground truth tokens should match accumulated tokens
        assert accumulated_tokens == ground_truth_tokens, f"Token mismatch!\n  Ground truth length: {len(ground_truth_tokens)}\n  Accumulated length:  {len(accumulated_tokens)}\n  Diff at position: {next((i for i, (a, b) in enumerate(zip(accumulated_tokens, ground_truth_tokens)) if a != b), 'lengths differ')}"

        # Verify decoded text matches
        accumulated_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=False)
        assert accumulated_text == ground_truth_text

    def test_multi_turn_tool_calls_ground_truth_qwen(self):
        """
        Test multi-turn tool calling against HuggingFace ground truth.

        Scenario: Model calls tool twice before giving final answer.
        This simulates the `while finish_reason == "tool_calls"` loop.
        """
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        tools = [WEATHER_TOOL]
        tool_map = create_mock_tool_functions()

        # ====================================================================
        # Build complete messages history (2 tool calls)
        # ====================================================================
        messages = [
            {"role": "user", "content": "What's the weather in New York and London?"},
            # Turn 1: First tool call
            {
                "role": "assistant",
                "content": "I'll check both cities.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "New York"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": json.dumps(tool_map["get_weather"]("New York")),
            },
            # Turn 2: Second tool call
            {
                "role": "assistant",
                "content": "Now London.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "London"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(tool_map["get_weather"]("London")),
            },
            # Final answer
            {
                "role": "assistant",
                "content": "New York is Sunny 25°C. London is Cloudy 22°C.",
            },
        ]

        # Ground truth
        ground_truth_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        ground_truth_tokens = tokenizer.encode(ground_truth_text, add_special_tokens=False)

        # ====================================================================
        # Simulate incremental accumulation
        # ====================================================================
        accumulated_tokens = []
        accumulated_loss_mask = []
        current_text_position = 0

        for i in range(len(messages)):
            partial_messages = messages[: i + 1]
            add_gen = (i < len(messages) - 1) and (messages[i]["role"] in ["tool"])

            partial_text = tokenizer.apply_chat_template(
                partial_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_gen,
            )

            # Extract new text
            new_text = partial_text[current_text_position:]
            new_tokens = tokenizer.encode(new_text, add_special_tokens=False)

            if new_tokens:
                accumulated_tokens.extend(new_tokens)

                # Set loss_mask based on role
                role = messages[i]["role"]
                if role == "tool":
                    # Tool response - not trainable
                    accumulated_loss_mask.extend([0] * len(new_tokens))
                elif role == "assistant":
                    # Assistant generation - trainable
                    accumulated_loss_mask.extend([1] * len(new_tokens))
                else:
                    # User/system - prompt, not in response
                    accumulated_loss_mask.extend([0] * len(new_tokens))

                current_text_position = len(partial_text)

        # Verify token match
        assert accumulated_tokens == ground_truth_tokens, f"Multi-turn token mismatch!\n  Ground truth: {len(ground_truth_tokens)} tokens\n  Accumulated:  {len(accumulated_tokens)} tokens"

        # Verify loss_mask structure
        assert len(accumulated_loss_mask) == len(accumulated_tokens)
        trainable = sum(accumulated_loss_mask)
        non_trainable = len(accumulated_loss_mask) - trainable
        assert trainable > 0, "Should have trainable (assistant) tokens"
        assert non_trainable > 0, "Should have non-trainable (tool/user) tokens"

    def test_tool_response_extraction_matches_formatter(self):
        """
        Test that extracting tool response from HF template matches our formatter.

        This is the critical verification: our formatter output must be
        EXACTLY the same as what HF produces between assistant and next assistant.
        """
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        tools = [CALCULATOR_TOOL]

        # Various test contents including edge cases
        test_contents = [
            "42",
            '{"result": 42, "status": "ok"}',
            "Error: Division by zero",
            "Line1\nLine2\nLine3",
            'He said "hello" and left.',
        ]

        for content in test_contents:
            messages_before = [
                {"role": "user", "content": "Calculate something"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "{}"},
                        }
                    ],
                },
            ]
            messages_after = messages_before + [
                {"role": "tool", "tool_call_id": "call_0", "content": content},
            ]

            before_text = tokenizer.apply_chat_template(messages_before, tools=tools, tokenize=False, add_generation_prompt=False)
            after_text = tokenizer.apply_chat_template(messages_after, tools=tools, tokenize=False, add_generation_prompt=True)

            # Extract HF's tool response
            hf_tool_response = after_text[len(before_text) :]

            # Our formatter (add_generation_prompt=True to match HF)
            our_tool_response = format_qwen(content, add_generation_prompt=True)

            assert our_tool_response == hf_tool_response, f"Mismatch for content {repr(content)}:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

            # Token-level verification
            hf_tokens = tokenizer.encode(hf_tool_response, add_special_tokens=False)
            our_tokens = tokenizer.encode(our_tool_response, add_special_tokens=False)
            assert hf_tokens == our_tokens, f"Token mismatch for content {repr(content)}"

    def test_glm_messages_history_ground_truth(self):
        """
        Test GLM-4 observation extraction matches our formatter.

        GLM uses 'observation' role for tool responses.
        We verify that our formatter produces the exact same string
        as HF's chat template for the observation section.
        """
        from examples.tool_calling.tools import format_glm

        tokenizer = get_tokenizer("/tmp/glm4-full")

        # GLM uses 'observation' role for tool responses
        messages_before = [
            {"role": "user", "content": "What is 2+2?"},
        ]
        messages_with_obs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "observation", "content": "4"},
        ]

        before_text = tokenizer.apply_chat_template(messages_before, tokenize=False, add_generation_prompt=False)
        with_obs_text = tokenizer.apply_chat_template(messages_with_obs, tokenize=False, add_generation_prompt=True)

        # Extract HF's observation section
        hf_observation_section = with_obs_text[len(before_text) :]

        # Our formatter (add_generation_prompt=True to match HF)
        our_observation = format_glm("4", add_generation_prompt=True)

        # They should match exactly
        assert our_observation == hf_observation_section, f"GLM observation mismatch:\n  HF:   {repr(hf_observation_section)}\n  Ours: {repr(our_observation)}"

        # Token-level verification
        hf_tokens = tokenizer.encode(hf_observation_section, add_special_tokens=False)
        our_tokens = tokenizer.encode(our_observation, add_special_tokens=False)
        assert hf_tokens == our_tokens, f"GLM token mismatch:\n  HF:   {hf_tokens}\n  Ours: {our_tokens}"


class TestParallelToolCalls:
    """
    Test handling of parallel/multiple tool calls in one response.

    Some models support calling multiple tools in a single turn.
    """

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    def test_multiple_tool_results_token_accumulation(self, tokenizer):
        """Test token accumulation with multiple tool results."""
        from examples.tool_calling.tools import format_qwen

        tool_map = create_mock_tool_functions()

        # Generation with multiple tool calls
        gen_text = "I'll check the weather in multiple cities."
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)

        all_tokens = list(gen_tokens)
        all_loss_mask = [1] * len(gen_tokens)
        all_logprobs = [-0.5] * len(gen_tokens)

        # Execute multiple tools
        cities = ["New York", "London", "Paris"]
        for city in cities:
            result = tool_map["get_weather"](city)
            obs = format_qwen(json.dumps(result))
            obs_tokens = tokenizer.encode(obs, add_special_tokens=False)

            all_tokens.extend(obs_tokens)
            all_loss_mask.extend([0] * len(obs_tokens))
            all_logprobs.extend([0.0] * len(obs_tokens))

            # Verify alignment after each observation
            assert len(all_tokens) == len(all_loss_mask) == len(all_logprobs)

        # Final response
        final_text = "Here are the weather conditions for all cities."
        final_tokens = tokenizer.encode(final_text, add_special_tokens=False)

        all_tokens.extend(final_tokens)
        all_loss_mask.extend([1] * len(final_tokens))
        all_logprobs.extend([-0.3] * len(final_tokens))

        # Final verification
        assert len(all_tokens) == len(all_loss_mask) == len(all_logprobs)

        # Count
        trainable = sum(all_loss_mask)
        non_trainable = len(all_loss_mask) - trainable

        assert trainable == len(gen_tokens) + len(final_tokens)
        assert non_trainable > 0  # Should have observation tokens


class TestToolExecutionAndFormatting:
    """Test tool execution and result formatting."""

    @pytest.fixture
    def registry(self):
        """Create a test registry with multiple tools."""
        from examples.tool_calling.tools import ToolRegistry, ToolSpec

        registry = ToolRegistry()
        tool_map = create_mock_tool_functions()

        # Register weather tool
        registry.register(
            ToolSpec(
                name="get_weather",
                description="Get weather information",
                parameters=WEATHER_TOOL["function"]["parameters"],
                func=tool_map["get_weather"],
            )
        )

        # Register calculator tool
        registry.register(
            ToolSpec(
                name="calculator",
                description="Evaluate math expression",
                parameters=CALCULATOR_TOOL["function"]["parameters"],
                func=tool_map["calculator"],
            )
        )

        return registry

    def test_tool_execution_success(self, registry):
        """Test successful tool execution."""
        from examples.tool_calling.tools import ToolCall

        async def run():
            call = ToolCall(
                name="get_weather",
                arguments={"city": "New York"},
                call_id="functions.get_weather:0",
            )
            result = await registry.execute(call)

            assert result.ok
            assert "Sunny" in result.output
            assert result.call_id == "functions.get_weather:0"

        asyncio.run(run())

    def test_tool_execution_with_formatting(self, registry):
        """Test tool execution followed by formatting for different models."""
        from examples.tool_calling.tools import ToolCall, format_observation

        async def run():
            call = ToolCall(
                name="calculator",
                arguments={"expression": "2 + 3 * 4"},
                call_id="call_0",
            )
            result = await registry.execute(call)

            assert result.ok
            assert result.output == "14"

            # Format for different parsers
            parser_checks = {
                "qwen25": "<tool_response>",
                "glm": "<|observation|>",
                "deepseekv3": "<｜tool▁output▁begin｜>",
            }

            for parser, expected_marker in parser_checks.items():
                obs = format_observation(result, parser)
                assert expected_marker in obs, f"Parser {parser}: missing {expected_marker}"
                assert result.output in obs

        asyncio.run(run())

    def test_tool_not_found_error(self, registry):
        """Test error handling for non-existent tool."""
        from examples.tool_calling.tools import ToolCall

        async def run():
            call = ToolCall(
                name="nonexistent_tool",
                arguments={},
                call_id="call_0",
            )
            result = await registry.execute(call)

            assert not result.ok
            assert result.error_type == "ToolNotFoundError"
            assert "nonexistent_tool" in result.error_message

        asyncio.run(run())

    def test_tool_execution_error_handling(self):
        """Test error handling when tool raises exception."""
        from examples.tool_calling.tools import ToolRegistry, ToolSpec, ToolCall

        def failing_tool(**kwargs):
            raise ValueError("Intentional test failure")

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="failing",
                description="A tool that always fails",
                parameters={"type": "object", "properties": {}},
                func=failing_tool,
            )
        )

        async def run():
            call = ToolCall(name="failing", arguments={}, call_id="call_0")
            result = await registry.execute(call)

            assert not result.ok
            assert result.error_type == "ValueError"
            assert result.error_message is not None
            assert "Intentional" in result.error_message

        asyncio.run(run())


class TestGenerateFunctionIntegration:
    """
    Integration tests for the main generate() function.

    These tests mock the HTTP calls but use real tokenizers
    to verify the complete flow.
    """

    @pytest.fixture
    def mock_args(self):
        """Create mock args object."""
        args = MagicMock()
        args.sglang_router_ip = "127.0.0.1"
        args.sglang_router_port = 30000
        args.partial_rollout = False
        args.rollout_max_context_len = 4096
        args.sglang_tool_call_parser = "qwen25"
        args.tool_parser = None
        return args

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    @pytest.fixture
    def registry(self):
        from examples.tool_calling.tools import ToolRegistry, create_calculator_tool

        registry = ToolRegistry()
        registry.register(create_calculator_tool())
        return registry

    @pytest.mark.asyncio
    async def test_generate_with_tool_call(self, mock_args, tokenizer, registry):
        """Test generate() with a single tool call."""
        pytest.importorskip("sglang")
        from examples.tool_calling.generate import generate, ToolCallingConfig
        from slime.utils.types import Sample

        sample = Sample(prompt="What is 2 + 3?")

        # Mock responses
        tool_call_text = "I'll calculate that.\n"
        final_text = "The answer is 5."

        responses = [
            create_generate_response(tool_call_text, tokenizer, "tool_calls"),
            create_generate_response(final_text, tokenizer, "stop"),
        ]
        call_idx = [0]

        async def mock_post(url, payload):
            idx = min(call_idx[0], len(responses) - 1)
            call_idx[0] += 1
            return responses[idx]

        # Mock GenerateState
        mock_state = MagicMock()
        mock_state.tokenizer = MagicMock()
        mock_state.tokenizer.side_effect = lambda text, add_special_tokens=False: {"input_ids": tokenizer.encode(text, add_special_tokens=add_special_tokens)}
        mock_state.tokenizer.decode = tokenizer.decode

        with patch("examples.tool_calling.generate.post", mock_post):
            with patch("examples.tool_calling.generate.GenerateState", return_value=mock_state):
                result = await generate(
                    args=mock_args,
                    sample=sample,
                    sampling_params={"max_new_tokens": 256},
                    registry=registry,
                    config=ToolCallingConfig(tool_parser="qwen25"),
                )

        # Verify result structure
        assert result.tokens is not None
        assert result.loss_mask is not None
        assert result.rollout_log_probs is not None

        # Verify alignment
        assert result.response_length == len(result.loss_mask)
        assert result.response_length == len(result.rollout_log_probs)

    @pytest.mark.asyncio
    async def test_generate_multi_hop(self, mock_args, tokenizer, registry):
        """Test generate() with multiple tool calling hops."""
        pytest.importorskip("sglang")
        from examples.tool_calling.generate import generate, ToolCallingConfig
        from slime.utils.types import Sample

        sample = Sample(prompt="Calculate 2+3, then multiply by 4")

        # Multi-hop responses with VALID Qwen25 tool call format
        # Qwen25 format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        hop1_text = 'First, let me calculate 2+3:\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+3"}}\n</tool_call>'
        hop2_text = 'Now multiply 5 by 4:\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "5*4"}}\n</tool_call>'
        hop3_text = "The final answer is 20."

        responses = [
            create_generate_response(hop1_text, tokenizer, "tool_calls"),
            create_generate_response(hop2_text, tokenizer, "tool_calls"),
            create_generate_response(hop3_text, tokenizer, "stop"),
        ]
        call_idx = [0]

        async def mock_post(url, payload):
            idx = min(call_idx[0], len(responses) - 1)
            call_idx[0] += 1
            return responses[idx]

        mock_state = MagicMock()
        mock_state.tokenizer = MagicMock()
        mock_state.tokenizer.side_effect = lambda text, add_special_tokens=False: {"input_ids": tokenizer.encode(text, add_special_tokens=add_special_tokens)}
        mock_state.tokenizer.decode = tokenizer.decode

        with patch("examples.tool_calling.generate.post", mock_post):
            with patch("examples.tool_calling.generate.GenerateState", return_value=mock_state):
                result = await generate(
                    args=mock_args,
                    sample=sample,
                    sampling_params={"max_new_tokens": 256},
                    registry=registry,
                    config=ToolCallingConfig(tool_parser="qwen25", max_hops=5),
                )

        # Verify multi-hop execution
        assert call_idx[0] == 3, "Should have made 3 /generate calls"

        # Verify alignment
        assert result.loss_mask is not None, "loss_mask should not be None"
        assert result.rollout_log_probs is not None, "rollout_log_probs should not be None"
        assert len(result.loss_mask) == result.response_length
        assert len(result.rollout_log_probs) == result.response_length

        # Should have both trainable and non-trainable tokens
        assert 1 in result.loss_mask, "Should have trainable tokens"
        # Should have observation tokens (loss_mask = 0)
        assert 0 in result.loss_mask, "Should have observation tokens (non-trainable)"


class TestErrorCases:
    """Test error handling and edge cases."""

    def test_formatter_missing_required_params(self):
        """Test formatters raise errors when required params missing."""
        from examples.tool_calling.tools import (
            format_gpt_oss,
            format_mistral,
            format_kimi_k2,
        )

        with pytest.raises(ValueError, match="tool_name"):
            format_gpt_oss("content")

        with pytest.raises(ValueError, match="tool_call_id"):
            format_mistral("content")

        # Kimi K2 only requires tool_call_id (tool_name no longer needed)
        with pytest.raises(ValueError, match="tool_call_id"):
            format_kimi_k2("content")

    def test_json_content_escaping_in_llama3(self):
        """Test that Llama3 formatter properly escapes special characters."""
        from examples.tool_calling.tools import format_llama3

        # Content with special characters
        test_cases = [
            'He said "hello"',
            "Line1\nLine2",
            "Path: C:\\Users\\test",
            '{"nested": "json"}',
        ]

        for content in test_cases:
            result = format_llama3(content)

            # Extract JSON part and verify it's valid
            json_start = result.find('{"output":')
            json_end = result.find("}<|eot_id|>") + 1
            json_part = result[json_start:json_end]

            parsed = json.loads(json_part)
            assert parsed["output"] == content, f"Content not preserved for: {content}"

    def test_empty_logprobs_raises_error(self):
        """Test that missing logprobs raises appropriate error."""
        pytest.importorskip("sglang")
        from examples.tool_calling.generate import extract_tokens_from_logprobs

        # Empty response
        empty_response = {"meta_info": {"output_token_logprobs": []}}

        with pytest.raises(ValueError, match="No output_token_logprobs"):
            extract_tokens_from_logprobs(empty_response)

        # Missing key
        missing_response = {"meta_info": {}}

        with pytest.raises(ValueError, match="No output_token_logprobs"):
            extract_tokens_from_logprobs(missing_response)


class TestGLM47DifficultCases:
    """
    Comprehensive tests for GLM-4.7 with difficult cases from docs.z.ai.

    These tests cover:
    - Multi-function assistant (get_current_time, calculate, search_web)
    - Sequential tool calls
    - Parallel tool calls (multiple tool_calls in one turn)
    - Complex JSON responses with nested objects
    - Error responses
    - Edge cases with special characters

    Based on: https://docs.z.ai/guides/capabilities/function-calling
    """

    @pytest.fixture
    def tokenizer(self):
        """Load GLM-4.7 tokenizer."""
        return get_tokenizer("/tmp/glm47")

    @pytest.fixture
    def multi_function_tools(self):
        """Tools from the multi-function assistant example."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression, e.g.: 2+3*4",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search web information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search keywords",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def test_glm47_single_tool_response_format(self, tokenizer, multi_function_tools):
        """Test GLM-4.7 single tool response matches HF template."""
        from examples.tool_calling.tools import format_glm47

        tool_result = '{"current_time": "2024-01-15 14:30:00", "timezone": "Asia/London"}'

        # Build HF conversation
        messages_before = [
            {"role": "user", "content": "What time is it now?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_0", "function": {"name": "get_current_time", "arguments": {}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": tool_result, "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        # Extract HF tool response section
        hf_tool_response = after_text[len(before_text) :]

        # Our formatter
        our_tool_response = format_glm47(tool_result, add_generation_prompt=True)

        assert our_tool_response == hf_tool_response, f"GLM-4.7 format mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

        # Token-level verification
        hf_tokens = tokenizer.encode(hf_tool_response, add_special_tokens=False)
        our_tokens = tokenizer.encode(our_tool_response, add_special_tokens=False)
        assert hf_tokens == our_tokens

    def test_glm47_calculate_with_complex_expression(self, tokenizer, multi_function_tools):
        """Test calculation with complex mathematical expression."""
        from examples.tool_calling.tools import format_glm47

        # Complex calculation result from docs.z.ai
        tool_result = '{"expression": "15 * 23 + 7", "result": 352}'

        messages_before = [
            {"role": "user", "content": "Help me calculate 15 * 23 + 7"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {"name": "calculate", "arguments": {"expression": "15 * 23 + 7"}},
                    }
                ],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": tool_result, "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_glm47(tool_result, add_generation_prompt=True)

        assert our_tool_response == hf_tool_response

    def test_glm47_search_with_nested_json(self, tokenizer, multi_function_tools):
        """Test search results with nested JSON structure."""
        from examples.tool_calling.tools import format_glm47

        # Complex nested JSON from search_web
        tool_result = json.dumps(
            {
                "query": "latest developments in artificial intelligence",
                "results": [
                    {"title": "AI Breakthrough 2024", "url": "https://example1.com", "score": 0.95},
                    {"title": "Machine Learning Advances", "url": "https://example2.com", "score": 0.87},
                ],
                "metadata": {"total_results": 1000, "search_time_ms": 45},
            },
            ensure_ascii=False,
        )

        messages_before = [
            {"role": "user", "content": "Search for the latest developments in artificial intelligence"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {
                            "name": "search_web",
                            "arguments": {"query": "latest developments in artificial intelligence"},
                        },
                    }
                ],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": tool_result, "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_glm47(tool_result, add_generation_prompt=True)

        assert our_tool_response == hf_tool_response

    def test_glm47_parallel_tool_calls(self, tokenizer, multi_function_tools):
        """
        Test parallel tool calls (multiple tool_calls in same turn).

        This is one of the most difficult cases: multiple tool responses
        share ONE <|observation|> prefix.
        """
        from examples.tool_calling.tools import format_glm47

        # Multiple tool calls in a single assistant turn
        messages_before = [
            {"role": "user", "content": "What time is it and calculate 2+2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_0", "function": {"name": "get_current_time", "arguments": {}}},
                    {"id": "call_1", "function": {"name": "calculate", "arguments": {"expression": "2+2"}}},
                ],
            },
        ]

        tool_result_1 = '{"current_time": "2024-01-15 14:30:00", "timezone": "Asia/London"}'
        tool_result_2 = '{"expression": "2+2", "result": 4}'

        messages_after = messages_before + [
            {"role": "tool", "content": tool_result_1, "tool_call_id": "call_0"},
            {"role": "tool", "content": tool_result_2, "tool_call_id": "call_1"},
        ]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]

        # For parallel tool calls, we need special handling
        # Expected: <|observation|><tool_response>result1</tool_response><tool_response>result2</tool_response><|assistant|><think>
        expected = f"<|observation|><tool_response>{tool_result_1}</tool_response><tool_response>{tool_result_2}</tool_response><|assistant|><think>"

        assert hf_tool_response == expected, f"Parallel tool call format mismatch:\n  HF:       {repr(hf_tool_response)}\n  Expected: {repr(expected)}"

        # Verify token alignment
        hf_tokens = tokenizer.encode(hf_tool_response, add_special_tokens=False)
        expected_tokens = tokenizer.encode(expected, add_special_tokens=False)
        assert hf_tokens == expected_tokens

    def test_glm47_error_response(self, tokenizer, multi_function_tools):
        """Test error response handling."""
        from examples.tool_calling.tools import format_glm47

        # Error from calculate function
        tool_result = json.dumps(
            {"error": "Expression contains disallowed characters", "error_code": "INVALID_PARAM"},
            ensure_ascii=False,
        )

        messages_before = [
            {"role": "user", "content": "Calculate rm -rf /"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_0", "function": {"name": "calculate", "arguments": {"expression": "rm -rf /"}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": tool_result, "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_glm47(tool_result, add_generation_prompt=True)

        assert our_tool_response == hf_tool_response

    def test_glm47_multi_turn_conversation(self, tokenizer, multi_function_tools):
        """
        Test multi-turn conversation with sequential tool calls.

        This simulates: User → Tool Call 1 → Response 1 → Tool Call 2 → Response 2 → Final Answer
        """
        from examples.tool_calling.tools import format_glm47

        # Build complete multi-turn conversation
        messages = [
            {"role": "user", "content": "What time is it? Then calculate how many seconds until midnight."},
            # First tool call
            {
                "role": "assistant",
                "content": "Let me check the time first.",
                "tool_calls": [{"id": "call_0", "function": {"name": "get_current_time", "arguments": {}}}],
            },
            {
                "role": "tool",
                "content": '{"current_time": "2024-01-15 22:30:00", "timezone": "Asia/London"}',
                "tool_call_id": "call_0",
            },
            # Second tool call
            {
                "role": "assistant",
                "content": "Now let me calculate the seconds until midnight.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "calculate", "arguments": {"expression": "(24-22.5)*3600"}},
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"expression": "(24-22.5)*3600", "result": 5400}',
                "tool_call_id": "call_1",
            },
            # Final answer
            {
                "role": "assistant",
                "content": "It's 10:30 PM and there are 5400 seconds (1.5 hours) until midnight.",
            },
        ]

        # Get ground truth
        ground_truth_text = tokenizer.apply_chat_template(messages, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        ground_truth_tokens = tokenizer.encode(ground_truth_text, add_special_tokens=False)

        # Verify our formatter produces matching segments
        # Check first tool response
        messages_1 = messages[:2]
        text_1 = tokenizer.apply_chat_template(messages_1, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        messages_1_with_tool = messages[:3]
        text_1_with_tool = tokenizer.apply_chat_template(messages_1_with_tool, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)
        hf_tool_response_1 = text_1_with_tool[len(text_1) :]
        our_tool_response_1 = format_glm47(messages[2]["content"], add_generation_prompt=True)

        assert our_tool_response_1 == hf_tool_response_1, f"Turn 1 tool response mismatch:\n  HF:   {repr(hf_tool_response_1)}\n  Ours: {repr(our_tool_response_1)}"

        # Check second tool response
        messages_2 = messages[:4]
        text_2 = tokenizer.apply_chat_template(messages_2, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        messages_2_with_tool = messages[:5]
        text_2_with_tool = tokenizer.apply_chat_template(messages_2_with_tool, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)
        hf_tool_response_2 = text_2_with_tool[len(text_2) :]
        our_tool_response_2 = format_glm47(messages[4]["content"], add_generation_prompt=True)

        assert our_tool_response_2 == hf_tool_response_2, f"Turn 2 tool response mismatch:\n  HF:   {repr(hf_tool_response_2)}\n  Ours: {repr(our_tool_response_2)}"

    def test_glm47_database_query_complex_result(self, tokenizer):
        """Test database query with complex table result."""
        from examples.tool_calling.tools import format_glm47

        # Database query result from docs.z.ai
        db_tool = {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "SQL query statement"}},
                    "required": ["sql"],
                },
            },
        }

        tool_result = json.dumps(
            {
                "success": True,
                "data": [
                    ["Alice", 28, "Engineer"],
                    ["Bob", 35, "Manager"],
                    ["Charlie", 42, "Director"],
                ],
                "row_count": 3,
                "columns": ["name", "age", "position"],
            },
            ensure_ascii=False,
        )

        messages_before = [
            {"role": "user", "content": "Show me all employees"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {"name": "query_database", "arguments": {"sql": "SELECT * FROM employees"}},
                    }
                ],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": tool_result, "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=[db_tool], tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=[db_tool], tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_glm47(tool_result, add_generation_prompt=True)

        assert our_tool_response == hf_tool_response

    def test_glm47_special_characters_edge_case(self, tokenizer, multi_function_tools):
        """Test tool responses with special characters that could break parsing."""
        from examples.tool_calling.tools import format_glm47

        edge_cases = [
            # Quotes and backslashes
            '{"message": "He said \\"hello\\""}',
            # Newlines in content
            '{"output": "Line1\\nLine2\\nLine3"}',
            # XML-like content (could confuse parser)
            '{"html": "<div><p>test</p></div>"}',
            # Tool response tags in content (tricky!)
            '{"data": "Found </tool_response> in text"}',
            # Unicode escapes
            '{"emoji": "\\u2764\\ufe0f"}',
            # Empty and null values
            '{"empty": "", "null": null, "zero": 0}',
        ]

        for content in edge_cases:
            messages_before = [
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_0", "function": {"name": "calculate", "arguments": {"expression": "1+1"}}}],
                },
            ]
            messages_after = messages_before + [{"role": "tool", "content": content, "tool_call_id": "call_0"}]

            before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
            after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

            hf_tool_response = after_text[len(before_text) :]
            our_tool_response = format_glm47(content, add_generation_prompt=True)

            assert our_tool_response == hf_tool_response, f"Edge case failed for content {repr(content)}:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

    def test_glm47_with_thinking_tokens(self, tokenizer, multi_function_tools):
        """
        Test GLM-4.7 with thinking tokens in assistant messages.

        When using thinking mode, assistant messages contain <think>...</think> blocks.
        The tool response format remains the same, but we must verify token accumulation
        handles thinking content correctly.
        """
        from examples.tool_calling.tools import format_glm47

        # Messages with thinking content (this is how models generate in thinking mode)
        messages_before = [
            {"role": "user", "content": "What is 15 * 23 + 7?"},
            {
                "role": "assistant",
                "content": "<think>I need to calculate this mathematical expression.</think>Let me calculate that for you.",
                "tool_calls": [{"id": "call_0", "function": {"name": "calculate", "arguments": {"expression": "15 * 23 + 7"}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": '{"expression": "15 * 23 + 7", "result": 352}', "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_glm47('{"expression": "15 * 23 + 7", "result": 352}', add_generation_prompt=True)

        # Tool response format should be the same regardless of thinking content
        assert our_tool_response == hf_tool_response, f"With thinking - tool response mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

    def test_glm47_with_reasoning_content_field(self, tokenizer, multi_function_tools):
        """
        Test GLM-4.7 with reasoning_content field (alternative thinking format).

        Some models use a separate 'reasoning_content' field instead of inline <think> tags.
        """
        from examples.tool_calling.tools import format_glm47

        messages_before = [
            {"role": "user", "content": "Search for AI news"},
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "reasoning_content": "The user wants to know about AI developments. I should use the search tool.",
                "tool_calls": [{"id": "call_0", "function": {"name": "search_web", "arguments": {"query": "AI news 2024"}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "content": '{"results": ["AI breakthrough"]}', "tool_call_id": "call_0"}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_glm47('{"results": ["AI breakthrough"]}', add_generation_prompt=True)

        assert our_tool_response == hf_tool_response

    def test_glm47_multi_turn_with_thinking(self, tokenizer, multi_function_tools):
        """
        Test multi-turn conversation where every assistant turn has thinking content.

        This is the realistic scenario: model always generates thinking before tool calls.
        """
        from examples.tool_calling.tools import format_glm47

        messages = [
            {"role": "user", "content": "What time is it? Then calculate seconds until midnight."},
            # Turn 1: Get time
            {
                "role": "assistant",
                "content": "<think>First I need to get the current time.</think>Let me check the time.",
                "tool_calls": [{"id": "call_0", "function": {"name": "get_current_time", "arguments": {}}}],
            },
            {
                "role": "tool",
                "content": '{"current_time": "2024-01-15 22:30:00", "timezone": "Asia/London"}',
                "tool_call_id": "call_0",
            },
            # Turn 2: Calculate
            {
                "role": "assistant",
                "content": "<think>It's 22:30, so 1.5 hours until midnight. That's 1.5 * 3600 seconds.</think>Now calculating.",
                "tool_calls": [{"id": "call_1", "function": {"name": "calculate", "arguments": {"expression": "1.5 * 3600"}}}],
            },
            {
                "role": "tool",
                "content": '{"result": 5400}',
                "tool_call_id": "call_1",
            },
            # Final answer
            {
                "role": "assistant",
                "content": "<think>I have both pieces of information now.</think>It's 10:30 PM with 5400 seconds until midnight.",
            },
        ]

        # Get ground truth for full conversation
        ground_truth_text = tokenizer.apply_chat_template(messages, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
        ground_truth_tokens = tokenizer.encode(ground_truth_text, add_special_tokens=False)

        # Verify each tool response section matches our formatter
        for tool_idx in [2, 4]:  # Tool response message indices
            messages_before = messages[:tool_idx]
            messages_with_tool = messages[: tool_idx + 1]

            before_text = tokenizer.apply_chat_template(messages_before, tools=multi_function_tools, tokenize=False, add_generation_prompt=False)
            with_tool_text = tokenizer.apply_chat_template(messages_with_tool, tools=multi_function_tools, tokenize=False, add_generation_prompt=True)

            hf_tool_section = with_tool_text[len(before_text) :]
            our_tool_section = format_glm47(messages[tool_idx]["content"], add_generation_prompt=True)

            assert our_tool_section == hf_tool_section, f"Tool response {tool_idx} mismatch:\n  HF:   {repr(hf_tool_section)}\n  Ours: {repr(our_tool_section)}"

        # Verify token accumulation alignment
        all_tokens = []
        all_loss_mask = []
        current_text_position = 0

        for i in range(len(messages)):
            partial_messages = messages[: i + 1]
            add_gen = i < len(messages) - 1 and messages[i]["role"] == "tool"

            partial_text = tokenizer.apply_chat_template(
                partial_messages,
                tools=multi_function_tools,
                tokenize=False,
                add_generation_prompt=add_gen,
            )

            new_text = partial_text[current_text_position:]
            if not new_text:
                continue

            new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
            if new_tokens:
                all_tokens.extend(new_tokens)

                role = messages[i]["role"]
                if role == "tool":
                    all_loss_mask.extend([0] * len(new_tokens))
                elif role == "assistant":
                    # Thinking tokens ARE trainable (model generates them)
                    all_loss_mask.extend([1] * len(new_tokens))
                else:
                    all_loss_mask.extend([0] * len(new_tokens))

                current_text_position = len(partial_text)

        assert len(all_tokens) == len(all_loss_mask)
        assert len(all_tokens) == len(ground_truth_tokens)

    def test_glm47_token_accumulation_alignment(self, tokenizer, multi_function_tools):
        """
        Test critical token alignment for RL training.

        Verify: len(token_ids) == len(loss_mask) == len(log_probs)
        """
        from examples.tool_calling.tools import format_glm47

        # Multi-turn conversation (without explicit thinking for baseline test)
        messages = [
            {"role": "user", "content": "Calculate 2+2 then 3+3"},
            {
                "role": "assistant",
                "content": "First calculation:",
                "tool_calls": [{"id": "call_0", "function": {"name": "calculate", "arguments": {"expression": "2+2"}}}],
            },
            {"role": "tool", "content": '{"result": 4}', "tool_call_id": "call_0"},
            {
                "role": "assistant",
                "content": "Second calculation:",
                "tool_calls": [{"id": "call_1", "function": {"name": "calculate", "arguments": {"expression": "3+3"}}}],
            },
            {"role": "tool", "content": '{"result": 6}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "First is 4, second is 6."},
        ]

        all_tokens = []
        all_loss_mask = []
        all_log_probs = []

        current_text_position = 0

        for i in range(len(messages)):
            partial_messages = messages[: i + 1]
            add_gen = i < len(messages) - 1 and messages[i]["role"] == "tool"

            partial_text = tokenizer.apply_chat_template(
                partial_messages,
                tools=multi_function_tools,
                tokenize=False,
                add_generation_prompt=add_gen,
            )

            new_text = partial_text[current_text_position:]
            if not new_text:
                continue

            new_tokens = tokenizer.encode(new_text, add_special_tokens=False)

            if new_tokens:
                all_tokens.extend(new_tokens)

                role = messages[i]["role"]
                if role == "tool":
                    all_loss_mask.extend([0] * len(new_tokens))
                    all_log_probs.extend([0.0] * len(new_tokens))
                elif role == "assistant":
                    all_loss_mask.extend([1] * len(new_tokens))
                    all_log_probs.extend([-0.5] * len(new_tokens))
                else:
                    all_loss_mask.extend([0] * len(new_tokens))
                    all_log_probs.extend([0.0] * len(new_tokens))

                current_text_position = len(partial_text)

                # Assert alignment at each step
                assert len(all_tokens) == len(all_loss_mask) == len(all_log_probs), f"Alignment broken at message {i}:\n  tokens={len(all_tokens)}, loss_mask={len(all_loss_mask)}, log_probs={len(all_log_probs)}"

        # Final verification
        assert len(all_tokens) == len(all_loss_mask) == len(all_log_probs)
        trainable = sum(all_loss_mask)
        non_trainable = len(all_loss_mask) - trainable
        assert trainable > 0, "Should have trainable (assistant) tokens"
        assert non_trainable > 0, "Should have non-trainable (tool/user) tokens"


class TestThinkingModeGroundTruth:
    """
    Ground truth tests for thinking mode across all supported models.

    These tests verify that our formatters work correctly when messages
    contain thinking content (reasoning_content field or <think> tags).

    Key insight: The tool response format is INDEPENDENT of thinking content
    in the assistant's message. Thinking tokens appear in assistant turns,
    not in tool responses.
    """

    def test_qwen3_reasoning_content_field(self):
        """
        Test Qwen3 handling of reasoning_content field.

        Qwen3 wraps reasoning_content in <think>\\n{content}\\n</think>\\n\\n
        """
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
        tools = [CALCULATOR_TOOL]

        # Message with reasoning_content field
        messages_before = [
            {"role": "user", "content": "Calculate 15 * 23"},
            {
                "role": "assistant",
                "content": "Let me calculate.",
                "reasoning_content": "I need to multiply 15 by 23.",
                "tool_calls": [{"id": "call_0", "function": {"name": "calculator", "arguments": {"expression": "15 * 23"}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "tool_call_id": "call_0", "content": '{"result": 345}'}]

        # Get ground truth
        before_text = tokenizer.apply_chat_template(messages_before, tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=tools, tokenize=False, add_generation_prompt=True)

        # Verify reasoning_content is in the output
        assert "multiply 15 by 23" in before_text, "reasoning_content should be in output"
        assert "<think>" in before_text, "Should have <think> tags"

        # Extract tool response section
        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_qwen('{"result": 345}', add_generation_prompt=True)

        # Tool response format should match regardless of thinking
        assert our_tool_response == hf_tool_response

    def test_qwen3_inline_think_tags(self):
        """Test Qwen3 with inline <think> tags in content."""
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
        tools = [CALCULATOR_TOOL]

        messages_before = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>Simple addition.</think>I'll calculate that.",
                "tool_calls": [{"id": "call_0", "function": {"name": "calculator", "arguments": {"expression": "2+2"}}}],
            },
        ]
        messages_after = messages_before + [{"role": "tool", "tool_call_id": "call_0", "content": '{"result": 4}'}]

        before_text = tokenizer.apply_chat_template(messages_before, tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_after, tools=tools, tokenize=False, add_generation_prompt=True)

        # <think> tags should be preserved
        assert "<think>" in before_text
        assert "Simple addition" in before_text

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_qwen('{"result": 4}', add_generation_prompt=True)
        assert our_tool_response == hf_tool_response

    def test_deepseek_r1_thinking_stripped(self):
        """
        Test DeepSeek-R1 strips thinking from saved content.

        DeepSeek-R1 only keeps content AFTER </think> in the output.
        Generation prompt adds <think>\\n
        """
        from examples.tool_calling.tools import format_deepseek_v3

        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-R1")
        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}]

        # Message with <think> tags - should be stripped
        messages_with_think = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>Let me think about this.</think>The answer is 4.",
            },
        ]

        result = tokenizer.apply_chat_template(messages_with_think, tokenize=False, add_generation_prompt=False)

        # Thinking should be STRIPPED
        assert "Let me think about this" not in result, "DeepSeek-R1 should strip thinking"
        assert "The answer is 4" in result

        # Test tool response format
        messages_tool = [
            {"role": "user", "content": "calc"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": "{}"}, "id": "0"}],
            },
            {"role": "tool", "content": '{"result": 4}'},
        ]

        messages_before = messages_tool[:2]
        before_text = tokenizer.apply_chat_template(messages_before, tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages_tool, tools=tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_deepseek_v3('{"result": 4}')

        assert our_tool_response == hf_tool_response, f"DeepSeek-R1 tool response mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

    def test_deepseek_r1_generation_prompt_has_think(self):
        """Verify DeepSeek-R1 adds <think> in generation prompt."""
        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-R1")

        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        assert result.endswith("<think>\n"), f"DeepSeek-R1 should end with <think>\\n, got: {repr(result[-50:])}"

    def test_glm47_vs_glm4_thinking_difference(self):
        """
        Compare GLM-4 vs GLM-4.7 thinking handling.

        GLM-4: No thinking support, uses 'observation' role
        GLM-4.7: Full thinking support, uses 'tool' role, adds <think> in gen prompt
        """
        from examples.tool_calling.tools import format_glm, format_glm47

        tokenizer_glm4 = get_tokenizer("/tmp/glm4-full")
        tokenizer_glm47 = get_tokenizer("/tmp/glm47")

        # GLM-4: uses observation role
        messages_glm4 = [
            {"role": "user", "content": "test"},
            {"role": "observation", "content": "result"},
        ]
        glm4_result = tokenizer_glm4.apply_chat_template(messages_glm4, tokenize=False, add_generation_prompt=True)

        # Should have observation token
        assert "<|observation|>" in glm4_result
        assert glm4_result.endswith("<|assistant|>"), "GLM-4 gen prompt should NOT have <think>"

        # Extract observation section
        user_end = glm4_result.find("test") + 4
        obs_section = glm4_result[user_end:]
        our_obs = format_glm("result", add_generation_prompt=True)
        assert our_obs == obs_section

        # GLM-4.7: uses tool role with full thinking support
        tools_glm47 = [{"type": "function", "function": {"name": "t", "description": "t", "parameters": {}}}]
        messages_glm47 = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "0", "function": {"name": "t", "arguments": {}}}]},
            {"role": "tool", "content": "result", "tool_call_id": "0"},
        ]

        before_glm47 = tokenizer_glm47.apply_chat_template(messages_glm47[:2], tools=tools_glm47, tokenize=False, add_generation_prompt=False)
        after_glm47 = tokenizer_glm47.apply_chat_template(messages_glm47, tools=tools_glm47, tokenize=False, add_generation_prompt=True)

        tool_section_glm47 = after_glm47[len(before_glm47) :]
        our_tool_glm47 = format_glm47("result", add_generation_prompt=True)

        assert our_tool_glm47 == tool_section_glm47
        assert tool_section_glm47.endswith("<think>"), "GLM-4.7 gen prompt should have <think>"

    def test_qwen25_no_thinking_support(self):
        """
        Verify Qwen2.5 does NOT support reasoning_content (it's ignored).

        This is important: don't assume all models support thinking!
        """
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        tools = [CALCULATOR_TOOL]

        # Message with reasoning_content - should be ignored
        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": "response",
                "reasoning_content": "THIS SHOULD BE IGNORED",
            },
        ]

        result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # reasoning_content should NOT appear
        assert "THIS SHOULD BE IGNORED" not in result, "Qwen2.5 should ignore reasoning_content"

        # But inline <think> tags ARE preserved
        messages_think = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "<think>thinking</think>response"},
        ]
        result_think = tokenizer.apply_chat_template(messages_think, tokenize=False, add_generation_prompt=False)
        assert "<think>thinking</think>" in result_think, "Qwen2.5 preserves inline <think> tags"

    def test_multi_turn_with_thinking_token_alignment(self):
        """
        Test token alignment in multi-turn conversation with thinking.

        This is the critical RL training test: ensure thinking tokens
        are correctly included in loss_mask (they ARE trainable).
        """
        from examples.tool_calling.tools import format_glm47

        tokenizer = get_tokenizer("/tmp/glm47")
        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}]

        # Multi-turn with thinking in every assistant turn
        messages = [
            {"role": "user", "content": "Calculate 2+2 then 3+3"},
            # Turn 1
            {
                "role": "assistant",
                "content": "<think>First calc</think>Calculating first.",
                "tool_calls": [{"id": "c0", "function": {"name": "calc", "arguments": {"x": "2+2"}}}],
            },
            {"role": "tool", "content": '{"r": 4}', "tool_call_id": "c0"},
            # Turn 2
            {
                "role": "assistant",
                "content": "<think>Second calc</think>Calculating second.",
                "tool_calls": [{"id": "c1", "function": {"name": "calc", "arguments": {"x": "3+3"}}}],
            },
            {"role": "tool", "content": '{"r": 6}', "tool_call_id": "c1"},
            # Final
            {
                "role": "assistant",
                "content": "<think>Done</think>Results: 4 and 6.",
            },
        ]

        # Get ground truth
        ground_truth = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)

        # Verify thinking is in output
        assert "<think>First calc</think>" in ground_truth
        assert "<think>Second calc</think>" in ground_truth
        assert "<think>Done</think>" in ground_truth

        # Accumulate tokens and build loss_mask
        all_tokens = []
        all_loss_mask = []
        current_pos = 0

        for i in range(len(messages)):
            partial = messages[: i + 1]
            add_gen = i < len(messages) - 1 and messages[i]["role"] == "tool"

            partial_text = tokenizer.apply_chat_template(partial, tools=tools, tokenize=False, add_generation_prompt=add_gen)

            new_text = partial_text[current_pos:]
            if not new_text:
                continue

            new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
            if new_tokens:
                all_tokens.extend(new_tokens)

                role = messages[i]["role"]
                if role == "tool":
                    # Tool responses are NOT trainable
                    all_loss_mask.extend([0] * len(new_tokens))
                elif role == "assistant":
                    # Assistant tokens (INCLUDING thinking) ARE trainable
                    all_loss_mask.extend([1] * len(new_tokens))
                else:
                    all_loss_mask.extend([0] * len(new_tokens))

                current_pos = len(partial_text)

        # Verify alignment
        assert len(all_tokens) == len(all_loss_mask)

        # Verify thinking tokens are marked as trainable
        decoded = tokenizer.decode(all_tokens)
        assert "<think>" in decoded

        # Count trainable vs non-trainable
        trainable = sum(all_loss_mask)
        non_trainable = len(all_loss_mask) - trainable
        assert trainable > 0
        assert non_trainable > 0


class TestNewModelsGroundTruth:
    """
    Ground truth tests for newly added models:
    - Kimi-K2-Thinking
    - DeepSeek-V3.2
    - Qwen3-Next
    """

    def test_kimi_k2_thinking_tool_response_ground_truth(self):
        """
        Verify Kimi-K2-Thinking tool response matches HF ground truth.

        Key finding: Kimi-K2-Thinking uses 'tool' as role name, not the function name.
        """
        from examples.tool_calling.tools import format_kimi_k2

        tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}]

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "content": "Let me calculate.",
                "tool_calls": [{"id": "functions.calc:0", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
            },
            {"role": "tool", "content": '{"result": 4}', "tool_call_id": "functions.calc:0"},
        ]

        before_text = tokenizer.apply_chat_template(messages[:2], tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_kimi_k2(
            content='{"result": 4}',
            tool_call_id="functions.calc:0",
            add_generation_prompt=True,
        )

        assert our_tool_response == hf_tool_response, f"Kimi-K2-Thinking tool response mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

    def test_kimi_k2_instruct_tool_response_ground_truth(self):
        """Verify Kimi-K2-Instruct tool response also matches."""
        from examples.tool_calling.tools import format_kimi_k2

        tokenizer = get_tokenizer("moonshotai/Kimi-K2-Instruct")
        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}]

        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
            },
            {"role": "tool", "content": '{"r": 4}', "tool_call_id": "call_0"},
        ]

        before_text = tokenizer.apply_chat_template(messages[:2], tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_kimi_k2(
            content='{"r": 4}',
            tool_call_id="call_0",
            add_generation_prompt=True,
        )

        assert our_tool_response == hf_tool_response

    def test_kimi_k2_thinking_always_empty_think_tags(self):
        """
        Test Kimi-K2-Thinking's thinking handling.

        KEY FINDING: Kimi-K2-Thinking does NOT preserve reasoning_content at all.
        Unlike Qwen3, it always adds empty <think></think> tags for all assistant messages.
        The reasoning_content field is completely ignored by the template.

        This means thinking tokens must be generated during inference, not loaded from history.
        """
        tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")

        # Single assistant with reasoning_content
        messages_single = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "answer", "reasoning_content": "MY_REASONING"},
        ]

        result_single = tokenizer.apply_chat_template(messages_single, tokenize=False, add_generation_prompt=False)

        # reasoning_content is NEVER preserved
        assert "MY_REASONING" not in result_single, "Kimi-K2-Thinking ignores reasoning_content"
        # But empty <think></think> tags are always added
        assert "<think></think>" in result_single, "Should have empty think tags"

        # Multi-turn conversation
        messages_multi = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "FIRST_REASON"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It's 4.", "reasoning_content": "SECOND_REASON"},
        ]

        result_multi = tokenizer.apply_chat_template(messages_multi, tokenize=False, add_generation_prompt=False)

        # Neither reasoning_content should appear
        assert "FIRST_REASON" not in result_multi
        assert "SECOND_REASON" not in result_multi
        # Both assistants should have empty think tags
        assert result_multi.count("<think></think>") == 2

    def test_deepseek_v32_tool_response_ground_truth(self):
        """
        Verify DeepSeek-V3.2 (V3-0324) tool response matches HF ground truth.

        Key finding: V3.2 uses same format as V3, NO thinking in generation prompt.
        """
        from examples.tool_calling.tools import format_deepseek_v3

        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3-0324")
        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}]

        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "0", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
            },
            {"role": "tool", "content": '{"result": 42}'},
        ]

        before_text = tokenizer.apply_chat_template(messages[:2], tools=tools, tokenize=False, add_generation_prompt=False)
        after_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        hf_tool_response = after_text[len(before_text) :]
        our_tool_response = format_deepseek_v3('{"result": 42}')

        assert our_tool_response == hf_tool_response, f"DeepSeek-V3.2 tool response mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

    def test_deepseek_v32_no_thinking_in_gen_prompt(self):
        """Verify DeepSeek-V3.2 does NOT add <think> in generation prompt (unlike R1)."""
        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3-0324")

        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        assert "<think>" not in result, "DeepSeek-V3.2 should NOT add <think>"
        assert result.endswith("<｜Assistant｜>"), f"Should end with Assistant, got: {repr(result[-50:])}"

    def test_qwen3_next_tool_response_ground_truth(self):
        """
        Verify Qwen3-Next (235B) tool response matches HF ground truth.

        Same format as Qwen3.

        NOTE: Cannot use simple before/after string slicing because Qwen3 template
        changes history message format (strips <think> tags from history assistant).
        Instead, use regex to extract the tool_response section directly.
        """
        import re
        from examples.tool_calling.tools import format_qwen

        tokenizer = get_tokenizer("Qwen/Qwen3-235B-A22B")
        tools = [CALCULATOR_TOOL]

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_0", "function": {"name": "calculator", "arguments": {"expression": "2+2"}}}],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": '{"result": 4}'},
        ]

        after_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        # Extract tool_response section using regex (template changes history format)
        match = re.search(
            r"(<\|im_start\|>user\n<tool_response>.*?</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n)",
            after_text,
            re.DOTALL,
        )
        assert match, f"Could not find tool_response section in: {repr(after_text[-200:])}"

        hf_tool_response = match.group(1)
        our_tool_response = format_qwen('{"result": 4}', add_generation_prompt=True)

        assert our_tool_response == hf_tool_response, f"Qwen3-Next tool response mismatch:\n  HF:   {repr(hf_tool_response)}\n  Ours: {repr(our_tool_response)}"

    def test_qwen3_next_reasoning_content_supported(self):
        """Verify Qwen3-Next supports reasoning_content like Qwen3."""
        tokenizer = get_tokenizer("Qwen/Qwen3-235B-A22B")

        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "MY_REASONING",
            },
        ]

        result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        assert "MY_REASONING" in result, "Qwen3-Next should support reasoning_content"
        assert "<think>" in result, "Should be wrapped in <think> tags"


class TestFormatterConsistency:
    """Test that all formatters produce consistent output."""

    @pytest.mark.parametrize(
        "parser_name",
        ["qwen", "qwen25", "qwen3_coder", "glm", "glm45", "glm47", "mimo", "hermes"],
    )
    def test_simple_formatters(self, parser_name):
        """Test formatters that don't require extra parameters."""
        from examples.tool_calling.tools import get_tool_response_formatter

        formatter = get_tool_response_formatter(parser_name)
        content = '{"result": 42}'

        result = formatter(content)
        assert content in result
        assert isinstance(result, str)
        assert len(result) > len(content)

    @pytest.mark.parametrize("parser_name", ["deepseekv3", "deepseekv31"])
    def test_deepseek_v3_formatters(self, parser_name):
        """Test DeepSeek V3 formatters."""
        from examples.tool_calling.tools import get_tool_response_formatter

        formatter = get_tool_response_formatter(parser_name)
        content = '{"result": 42}'

        result = formatter(content)
        assert content in result
        assert "｜" in result  # DeepSeek special tokens

    def test_deepseek_v32_formatter(self):
        """Test DeepSeek V3.2 DSML formatter."""
        from examples.tool_calling.tools import format_deepseek_v32

        content = '{"result": 42}'
        result = format_deepseek_v32(content)

        assert "<function_results>" in result
        assert "<result>" in result
        assert content in result

    def test_minimax_formatter(self):
        """Test MiniMax M2 formatter."""
        from examples.tool_calling.tools import format_minimax

        content = '{"result": 42}'
        result = format_minimax(content)

        assert "]~b]tool" in result
        assert "<response>" in result
        assert "</response>" in result
        assert "[e~[" in result


class TestRealisticMultiHopWithReasoning:
    """
    Realistic multi-hop tool calling tests with reasoning tokens.

    Based on our analysis of how different models handle reasoning_content:
    - Qwen3: Preserves reasoning in ALL tool call messages
    - Kimi-K2-Thinking: Preserves reasoning in tool call messages
    - Regular chat (no tools): Only last reasoning preserved

    These tests verify the exact data flow for RL training.
    """

    @pytest.fixture
    def qwen3_tokenizer(self):
        return get_tokenizer("Qwen/Qwen3-235B-A22B")

    @pytest.fixture
    def kimi_tokenizer(self):
        return get_tokenizer("moonshotai/Kimi-K2-Thinking")

    @pytest.fixture
    def tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                    },
                },
            }
        ]

    def test_qwen3_multi_hop_reasoning_preserved_in_tool_calls(self, qwen3_tokenizer, tools):
        """
        Qwen3: ALL reasoning_content is preserved when messages have tool_calls.

        This is critical for RL training - the model sees its previous reasoning
        in context during multi-hop tool calling.
        """
        # Realistic multi-hop: (2+3) * 4
        # Hop 1: Model thinks and calls calculator(2+3)
        # Hop 2: Model thinks with result and calls calculator(5*4)
        # Hop 3: Model gives final answer

        messages = [
            {"role": "user", "content": "What is (2+3) * 4?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need to calculate 2+3 first.",
                "tool_calls": [{"id": "0", "function": {"name": "calculator", "arguments": {"expr": "2+3"}}}],
            },
            {"role": "tool", "tool_call_id": "0", "content": '{"result": 5}'},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Got 5. Now multiply by 4.",
                "tool_calls": [{"id": "1", "function": {"name": "calculator", "arguments": {"expr": "5*4"}}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": '{"result": 20}'},
        ]

        prompt = qwen3_tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        # Both reasoning contents should be preserved
        assert "I need to calculate 2+3 first" in prompt, "Hop 1 reasoning should be preserved"
        assert "Got 5. Now multiply by 4" in prompt, "Hop 2 reasoning should be preserved"

        # Both should be wrapped in <think> tags
        assert prompt.count("<think>") >= 2, "Should have at least 2 think blocks"
        assert prompt.count("</think>") >= 2, "Should have at least 2 think end tags"

    def test_qwen3_regular_chat_only_last_reasoning_preserved(self, qwen3_tokenizer):
        """
        Qwen3: In regular chat (no tool_calls), only LAST reasoning is preserved.

        History reasoning gets stripped - this is different from tool call behavior!
        """
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "HISTORY_REASONING"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It's 4.", "reasoning_content": "LAST_REASONING"},
        ]

        prompt = qwen3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # History reasoning is STRIPPED
        assert "HISTORY_REASONING" not in prompt, "History reasoning should be stripped"
        # Last reasoning is PRESERVED
        assert "LAST_REASONING" in prompt, "Last reasoning should be preserved"

    def test_kimi_k2_multi_hop_reasoning_preserved(self, kimi_tokenizer, tools):
        """
        Kimi-K2-Thinking: Reasoning preserved in tool call messages.

        Uses different format than Qwen but same preservation behavior.
        """
        messages = [
            {"role": "user", "content": "Calculate 5+5"},
            {
                "role": "assistant",
                "content": "Let me calculate.",
                "reasoning_content": "KIMI_FIRST_REASONING",
                "tool_calls": [{"id": "functions.calculator:0", "type": "function", "function": {"name": "calculator", "arguments": '{"expr": "5+5"}'}}],
            },
            {"role": "tool", "tool_call_id": "functions.calculator:0", "content": '{"result": 10}'},
        ]

        prompt = kimi_tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

        # Kimi should preserve reasoning in tool call context
        assert "KIMI_FIRST_REASONING" in prompt, "Kimi should preserve reasoning in tool calls"
        assert "<think>" in prompt, "Should have think tags"

    def test_rl_training_data_flow_qwen3(self, qwen3_tokenizer, tools):
        """
        Complete RL training data flow test for Qwen3.

        Verifies:
        1. Prompt construction with reasoning preserved
        2. Token alignment (can encode/decode properly)
        3. Loss mask construction logic
        """
        from examples.tool_calling.tools import format_qwen

        # Step 1: Initial prompt
        initial_messages = [{"role": "user", "content": "What is 2+2?"}]
        initial_prompt = qwen3_tokenizer.apply_chat_template(initial_messages, tools=tools, tokenize=False, add_generation_prompt=True)

        # Step 2: Simulate model generation with reasoning + tool call
        model_output = '<think>\nLet me calculate 2+2.\n</think>\n\n<tool_call>\n{"name": "calculator", "arguments": {"expr": "2+2"}}\n</tool_call>'
        model_tokens = qwen3_tokenizer.encode(model_output, add_special_tokens=False)

        # These tokens should have loss_mask = 1 (model generated)
        assert len(model_tokens) > 0

        # Step 3: Tool response formatting
        tool_result = '{"result": 4}'
        observation = format_qwen(tool_result)
        observation_tokens = qwen3_tokenizer.encode(observation, add_special_tokens=False)

        # These tokens should have loss_mask = 0 (not trained on)
        assert len(observation_tokens) > 0

        # Step 4: Verify observation format matches template expectation
        # After tool result, construct messages for next hop
        messages_after_tool = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Let me calculate 2+2.",
                "tool_calls": [{"id": "0", "function": {"name": "calculator", "arguments": {"expr": "2+2"}}}],
            },
            {"role": "tool", "tool_call_id": "0", "content": tool_result},
        ]

        next_hop_prompt = qwen3_tokenizer.apply_chat_template(messages_after_tool, tools=tools, tokenize=False, add_generation_prompt=True)

        # Verify tool_response format is in the prompt
        assert "<tool_response>" in next_hop_prompt
        assert tool_result in next_hop_prompt
        # Reasoning should be preserved for next hop
        assert "Let me calculate 2+2" in next_hop_prompt

    def test_token_alignment_invariants(self, qwen3_tokenizer):
        """
        Critical RL invariant: len(token_ids) == len(loss_mask) == len(log_probs)

        This test verifies that our token handling maintains alignment.
        """
        from examples.tool_calling.tools import format_qwen

        # Model generation phase
        generation_text = "<think>\nThinking...\n</think>\nAnswer"
        generation_tokens = qwen3_tokenizer.encode(generation_text, add_special_tokens=False)
        generation_loss_mask = [1] * len(generation_tokens)  # All trainable
        generation_log_probs = [-0.5] * len(generation_tokens)  # Dummy

        # Observation phase
        observation = format_qwen('{"result": 42}')
        observation_tokens = qwen3_tokenizer.encode(observation, add_special_tokens=False)
        observation_loss_mask = [0] * len(observation_tokens)  # Not trainable
        observation_log_probs = [0.0] * len(observation_tokens)  # Placeholder

        # Combined
        all_tokens = generation_tokens + observation_tokens
        all_loss_mask = generation_loss_mask + observation_loss_mask
        all_log_probs = generation_log_probs + observation_log_probs

        # Critical invariant
        assert len(all_tokens) == len(all_loss_mask) == len(all_log_probs), f"Alignment broken: tokens={len(all_tokens)}, loss_mask={len(all_loss_mask)}, log_probs={len(all_log_probs)}"

        # Verify loss_mask values
        assert all(m in (0, 1) for m in all_loss_mask), "Invalid loss_mask values"
        assert 1 in all_loss_mask, "Should have trainable tokens"
        assert 0 in all_loss_mask, "Should have non-trainable tokens"

    def test_multi_hop_complete_flow(self, qwen3_tokenizer, tools):
        """
        Complete 3-hop tool calling flow with full token tracking.

        Hop 1: User question → Model thinks + tool call
        Hop 2: Tool result → Model thinks + tool call
        Hop 3: Tool result → Model final answer

        Each hop builds on previous, preserving reasoning context.
        """
        from examples.tool_calling.tools import format_qwen

        # === Hop 1: Initial question ===
        messages_hop1 = [{"role": "user", "content": "What is (2+3)*4?"}]
        prompt_hop1 = qwen3_tokenizer.apply_chat_template(messages_hop1, tools=tools, tokenize=False, add_generation_prompt=True)
        prompt_hop1_tokens = qwen3_tokenizer.encode(prompt_hop1, add_special_tokens=False)

        # Model generates: think + tool_call
        hop1_output = '<think>\nFirst calculate 2+3.\n</think>\n\n<tool_call>\n{"name": "calculator", "arguments": {"expr": "2+3"}}\n</tool_call>'
        hop1_tokens = qwen3_tokenizer.encode(hop1_output, add_special_tokens=False)
        hop1_loss_mask = [1] * len(hop1_tokens)

        # Tool response
        tool_result_1 = '{"result": 5}'
        obs1 = format_qwen(tool_result_1)
        obs1_tokens = qwen3_tokenizer.encode(obs1, add_special_tokens=False)
        obs1_loss_mask = [0] * len(obs1_tokens)

        # === Hop 2: Continue with tool result ===
        messages_hop2 = [
            {"role": "user", "content": "What is (2+3)*4?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "First calculate 2+3.",
                "tool_calls": [{"id": "0", "function": {"name": "calculator", "arguments": {"expr": "2+3"}}}],
            },
            {"role": "tool", "tool_call_id": "0", "content": tool_result_1},
        ]
        prompt_hop2 = qwen3_tokenizer.apply_chat_template(messages_hop2, tools=tools, tokenize=False, add_generation_prompt=True)

        # Verify hop 1 reasoning is in hop 2 prompt
        assert "First calculate 2+3" in prompt_hop2, "Hop 1 reasoning should persist"

        # Model generates second tool call
        hop2_output = '<think>\nGot 5. Now multiply by 4.\n</think>\n\n<tool_call>\n{"name": "calculator", "arguments": {"expr": "5*4"}}\n</tool_call>'
        hop2_tokens = qwen3_tokenizer.encode(hop2_output, add_special_tokens=False)
        hop2_loss_mask = [1] * len(hop2_tokens)

        # Tool response
        tool_result_2 = '{"result": 20}'
        obs2 = format_qwen(tool_result_2)
        obs2_tokens = qwen3_tokenizer.encode(obs2, add_special_tokens=False)
        obs2_loss_mask = [0] * len(obs2_tokens)

        # === Hop 3: Final answer ===
        messages_hop3 = [
            {"role": "user", "content": "What is (2+3)*4?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "First calculate 2+3.",
                "tool_calls": [{"id": "0", "function": {"name": "calculator", "arguments": {"expr": "2+3"}}}],
            },
            {"role": "tool", "tool_call_id": "0", "content": tool_result_1},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Got 5. Now multiply by 4.",
                "tool_calls": [{"id": "1", "function": {"name": "calculator", "arguments": {"expr": "5*4"}}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": tool_result_2},
        ]
        prompt_hop3 = qwen3_tokenizer.apply_chat_template(messages_hop3, tools=tools, tokenize=False, add_generation_prompt=True)

        # Both reasoning contents should be in final prompt
        assert "First calculate 2+3" in prompt_hop3, "Hop 1 reasoning should persist in hop 3"
        assert "Got 5. Now multiply by 4" in prompt_hop3, "Hop 2 reasoning should persist in hop 3"

        # === Verify total token counts ===
        total_response_tokens = len(hop1_tokens) + len(obs1_tokens) + len(hop2_tokens) + len(obs2_tokens)
        total_loss_mask = hop1_loss_mask + obs1_loss_mask + hop2_loss_mask + obs2_loss_mask

        assert len(total_loss_mask) == total_response_tokens

        # Count trainable vs non-trainable
        trainable = sum(total_loss_mask)
        non_trainable = len(total_loss_mask) - trainable

        assert trainable > 0, "Should have trainable tokens (model generations)"
        assert non_trainable > 0, "Should have non-trainable tokens (observations)"


# ============================================================================
# Standardized Tool Calling Benchmark
# ============================================================================
# Combines the hardest cases from GLM and Kimi:
# 1. Tool Call ID (Kimi style: functions.{name}:{idx})
# 2. Multi-tool parallel calls with section wrapping
# 3. Thinking tokens with reasoning_content preservation
# 4. Interleaved thinking between tool calls
# 5. Multi-hop (multiple rounds of tool calls)
#
# ALL models must pass these tests to be considered compliant.
# ============================================================================


class TestStandardizedToolCallingBenchmark:
    """
    Standardized benchmark for tool calling + thinking token handling.

    Every model must pass these tests to be compliant with the slime RL training
    pipeline. Tests are designed to catch the most common failure modes:

    1. TOOL_CALL_ID: Some models require IDs, some don't
    2. PARALLEL_CALLS: Multiple tool calls in one turn
    3. THINKING_PRESERVATION: reasoning_content in history
    4. INTERLEAVED_THINKING: thinking between tool results
    5. MULTI_HOP: Sequential tool call → result → call chains

    Based on analysis of:
    - Kimi-K2: Requires tool_call_id in `functions.{name}:{idx}` format
    - GLM-4.7: Supports interleaved thinking with `<|assistant|><think>`
    - Qwen3: Preserves ALL reasoning in tool call context
    - DeepSeek: Strips history reasoning via custom encoding
    """

    # Standard tools used across all benchmark tests
    BENCHMARK_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate math expressions",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time for a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        },
    ]

    # Test scenario: Multi-hop with parallel calls and reasoning
    # User: "What time is it in Tokyo and London, and what's 5+3?"
    # Hop 1: Model thinks, calls get_time(Tokyo), get_time(London), calculator(5+3) in parallel
    # Hop 2: Model receives results, gives final answer

    @pytest.fixture
    def get_tokenizer_for_model(self):
        """Factory fixture for getting tokenizers."""

        def _get(model_name):
            return get_tokenizer(model_name)

        return _get

    @pytest.mark.parametrize(
        "model_id,parser_name,requires_tool_call_id",
        [
            ("Qwen/Qwen2.5-72B-Instruct", "qwen25", False),
            ("Qwen/Qwen3-235B-A22B", "qwen25", False),
            # Note: GLM-4 uses different message format (metadata field for function name)
            # It's tested separately in TestGLM47DifficultCases
            ("moonshotai/Kimi-K2-Instruct", "kimi_k2", True),
        ],
    )
    def test_benchmark_single_tool_call(self, get_tokenizer_for_model, model_id, parser_name, requires_tool_call_id):
        """
        BENCHMARK 1: Single tool call with thinking.

        Minimum viable test - every model must pass this.
        """
        from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS

        tokenizer = get_tokenizer_for_model(model_id)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        # Build tool call with optional ID
        tool_call = {
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
        }
        if requires_tool_call_id:
            tool_call["id"] = "functions.calculator:0"

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "Let me calculate that.",
                "tool_calls": [tool_call],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", "0"),
                "content": '{"result": 4}',
            },
        ]

        # Apply template
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=self.BENCHMARK_TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Verify tool result appears in prompt
        assert "4" in prompt, f"{model_id}: Tool result should appear in prompt"

        # Verify we can tokenize without errors
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(tokens) > 0, f"{model_id}: Should produce tokens"

        # Verify formatter produces non-empty output
        formatted = formatter(
            content='{"result": 4}',
            tool_name="calculator",
            tool_call_id=tool_call.get("id", ""),
        )
        assert len(formatted) > 0, f"{model_id}: Formatter should produce output"

    @pytest.mark.parametrize(
        "model_id,parser_name,requires_tool_call_id",
        [
            ("Qwen/Qwen3-235B-A22B", "qwen25", False),
            ("moonshotai/Kimi-K2-Instruct", "kimi_k2", True),
        ],
    )
    def test_benchmark_parallel_tool_calls(self, get_tokenizer_for_model, model_id, parser_name, requires_tool_call_id):
        """
        BENCHMARK 2: Parallel tool calls (multiple tools in one turn).

        Tests the model's ability to handle multiple tool calls simultaneously.
        This is critical for efficiency in agentic workflows.
        """
        _ = parser_name  # Used for test parameterization, not in test body
        tokenizer = get_tokenizer_for_model(model_id)

        # Three parallel tool calls
        tool_calls = []
        for idx, (name, args) in enumerate(
            [
                ("get_time", '{"timezone": "Asia/Tokyo"}'),
                ("get_time", '{"timezone": "Europe/London"}'),
                ("calculator", '{"expr": "5+3"}'),
            ]
        ):
            tc = {
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            if requires_tool_call_id:
                tc["id"] = f"functions.{name}:{idx}"
            else:
                tc["id"] = str(idx)
            tool_calls.append(tc)

        messages = [
            {"role": "user", "content": "What time is it in Tokyo and London, and what's 5+3?"},
            {
                "role": "assistant",
                "content": "Let me check all of that for you.",
                "tool_calls": tool_calls,
            },
        ]

        # Add tool responses
        tool_responses = [
            ("functions.get_time:0" if requires_tool_call_id else "0", "14:30 JST"),
            ("functions.get_time:1" if requires_tool_call_id else "1", "06:30 GMT"),
            ("functions.calculator:2" if requires_tool_call_id else "2", "8"),
        ]

        for tool_call_id, result in tool_responses:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
            )

        prompt = tokenizer.apply_chat_template(
            messages,
            tools=self.BENCHMARK_TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )

        # All three results should appear
        assert "14:30" in prompt or "JST" in prompt, f"{model_id}: Tokyo time should appear"
        assert "06:30" in prompt or "GMT" in prompt, f"{model_id}: London time should appear"
        assert "8" in prompt, f"{model_id}: Calculator result should appear"

    @pytest.mark.parametrize(
        "model_id,supports_reasoning_content",
        [
            ("Qwen/Qwen3-235B-A22B", True),
            ("moonshotai/Kimi-K2-Thinking", True),
        ],
    )
    def test_benchmark_reasoning_preservation_in_tool_calls(self, get_tokenizer_for_model, model_id, supports_reasoning_content):
        """
        BENCHMARK 3: reasoning_content preservation in tool call context.

        For RL training, we need to ensure the model sees its previous reasoning
        when making subsequent tool calls. This test verifies that behavior.
        """
        if not supports_reasoning_content:
            pytest.skip(f"{model_id} doesn't support reasoning_content")

        tokenizer = get_tokenizer_for_model(model_id)

        # Multi-hop with reasoning at each step
        messages = [
            {"role": "user", "content": "What is (10 / 2) + 3?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "STEP1_REASONING: I need to calculate 10/2 first.",
                "tool_calls": [{"id": "0", "function": {"name": "calculator", "arguments": {"expr": "10/2"}}}],
            },
            {"role": "tool", "tool_call_id": "0", "content": '{"result": 5}'},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "STEP2_REASONING: Got 5, now add 3.",
                "tool_calls": [{"id": "1", "function": {"name": "calculator", "arguments": {"expr": "5+3"}}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": '{"result": 8}'},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tools=self.BENCHMARK_TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Both reasoning steps should be preserved in tool call context
        assert "STEP1_REASONING" in prompt, f"{model_id}: Step 1 reasoning should be preserved"
        assert "STEP2_REASONING" in prompt, f"{model_id}: Step 2 reasoning should be preserved"

    @pytest.mark.parametrize(
        "model_id,parser_name",
        [
            ("Qwen/Qwen3-235B-A22B", "qwen25"),
            ("moonshotai/Kimi-K2-Instruct", "kimi_k2"),
        ],
    )
    def test_benchmark_multi_hop_chain(self, get_tokenizer_for_model, model_id, parser_name):
        """
        BENCHMARK 4: Multi-hop tool calling chain.

        Complex scenario: 3-hop chain where each result informs the next call.
        This tests the full agentic loop capability.

        Scenario: "Find population of France, convert to millions, format as percentage of world"
        Hop 1: search("France population")
        Hop 2: calculator("67000000 / 1000000")
        Hop 3: calculator("67 / 8000 * 100")
        """
        from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS

        tokenizer = get_tokenizer_for_model(model_id)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        # Simulate the multi-hop flow with token tracking
        all_tokens = []
        all_loss_mask = []

        # === HOP 1 ===
        messages_hop1 = [
            {"role": "user", "content": "What's France's population as a percentage of world population?"},
        ]
        prompt_hop1 = tokenizer.apply_chat_template(messages_hop1, tools=self.BENCHMARK_TOOLS, tokenize=False, add_generation_prompt=True)
        # Note: prompt tokens would have loss_mask = 0 (not trained on input)
        # We skip tracking them here since we focus on response tokens
        _ = tokenizer.encode(prompt_hop1, add_special_tokens=False)  # Verify tokenizable

        # Model generation for hop 1
        hop1_output = "Let me search for France population first."
        hop1_tokens = tokenizer.encode(hop1_output, add_special_tokens=False)
        all_tokens.extend(hop1_tokens)
        all_loss_mask.extend([1] * len(hop1_tokens))  # Model generation: train

        # Tool result injection
        tool_result_1 = "France population: 67 million"
        obs1 = formatter(content=tool_result_1, tool_name="search", tool_call_id="0")
        obs1_tokens = tokenizer.encode(obs1, add_special_tokens=False)
        all_tokens.extend(obs1_tokens)
        all_loss_mask.extend([0] * len(obs1_tokens))  # Observation: don't train

        # === HOP 2 ===
        hop2_output = "Converting to percentage..."
        hop2_tokens = tokenizer.encode(hop2_output, add_special_tokens=False)
        all_tokens.extend(hop2_tokens)
        all_loss_mask.extend([1] * len(hop2_tokens))

        tool_result_2 = "0.8375"
        obs2 = formatter(content=tool_result_2, tool_name="calculator", tool_call_id="1")
        obs2_tokens = tokenizer.encode(obs2, add_special_tokens=False)
        all_tokens.extend(obs2_tokens)
        all_loss_mask.extend([0] * len(obs2_tokens))

        # === HOP 3: Final answer ===
        hop3_output = "France's population is approximately 0.84% of the world population."
        hop3_tokens = tokenizer.encode(hop3_output, add_special_tokens=False)
        all_tokens.extend(hop3_tokens)
        all_loss_mask.extend([1] * len(hop3_tokens))

        # === VERIFY INVARIANTS ===
        assert len(all_tokens) == len(all_loss_mask), f"{model_id}: Token/loss_mask alignment broken: {len(all_tokens)} tokens, {len(all_loss_mask)} loss_mask"

        # Should have both trainable and non-trainable tokens
        trainable = sum(all_loss_mask)
        non_trainable = len(all_loss_mask) - trainable
        assert trainable > 0, f"{model_id}: No trainable tokens"
        assert non_trainable > 0, f"{model_id}: No observation tokens"

        # Verify ratio is reasonable (observations shouldn't dominate)
        train_ratio = trainable / len(all_loss_mask)
        assert 0.3 < train_ratio < 0.9, f"{model_id}: Unusual train ratio {train_ratio:.2f} - expected between 0.3 and 0.9"

    @pytest.mark.parametrize(
        "model_id,parser_name,id_format",
        [
            ("Qwen/Qwen2.5-72B-Instruct", "qwen25", "simple"),
            ("Qwen/Qwen3-235B-A22B", "qwen25", "simple"),
            ("moonshotai/Kimi-K2-Instruct", "kimi_k2", "kimi"),
        ],
    )
    def test_benchmark_tool_call_id_format_compliance(self, get_tokenizer_for_model, model_id, parser_name, id_format):
        """
        BENCHMARK 5: Tool call ID format compliance.

        Different models use different ID formats:
        - Kimi: functions.{name}:{idx} (e.g., "functions.calculator:0")
        - Others: Simple numeric or UUID

        This test verifies formatters handle IDs correctly.
        """
        _ = get_tokenizer_for_model, model_id  # Parameterized for documentation
        from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS

        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        if id_format == "kimi":
            tool_call_id = "functions.search:0"
        else:
            tool_call_id = "call_abc123"

        result = '{"data": "test"}'

        try:
            formatted = formatter(
                content=result,
                tool_name="search",
                tool_call_id=tool_call_id,
            )
            assert len(formatted) > 0, f"{model_id}: Formatter produced empty output"

            # For Kimi format, verify ID appears in output
            if id_format == "kimi":
                assert tool_call_id in formatted, f"{model_id}: Kimi ID should appear in formatted output"

        except ValueError as e:
            # Some formatters require tool_call_id
            if "requires" in str(e).lower() and not tool_call_id:
                pytest.skip(f"{model_id}: Formatter requires tool_call_id")
            raise

    def test_benchmark_special_characters_in_tool_output(self):
        """
        BENCHMARK 6: Special characters in tool output.

        Tool outputs often contain JSON, quotes, newlines, unicode, etc.
        The formatter must handle these correctly without breaking tokenization.
        """
        from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS

        test_cases = [
            ('{"key": "value with \\"quotes\\""}', "JSON with escaped quotes"),
            ("Line 1\nLine 2\nLine 3", "Multi-line output"),
            ("Price: $100 (10% off)", "Special chars"),
            ("日本語テスト", "Unicode (Japanese)"),
            ('{"nested": {"deep": {"value": 42}}}', "Nested JSON"),
            ("<xml>not html</xml>", "XML-like content"),
        ]

        for parser_name in ["qwen25", "glm", "kimi_k2"]:
            formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

            for content, description in test_cases:
                try:
                    formatted = formatter(
                        content=content,
                        tool_name="test_tool",
                        tool_call_id="functions.test_tool:0" if parser_name == "kimi_k2" else "0",
                    )
                    assert len(formatted) > 0, f"{parser_name}: Empty output for {description}"
                    # Content should be somewhere in output (may be escaped)
                    # Just verify no exceptions
                except Exception as e:
                    pytest.fail(f"{parser_name} failed on {description}: {e}")

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen3-235B-A22B",
            "moonshotai/Kimi-K2-Thinking",
        ],
    )
    def test_benchmark_thinking_then_answer_pattern(self, get_tokenizer_for_model, model_id):
        """
        BENCHMARK 7: Thinking-then-answer pattern without tool calls.

        After tool use is complete, model should be able to think and
        produce a final answer. This tests the clean transition.
        """
        tokenizer = get_tokenizer_for_model(model_id)

        messages = [
            {"role": "user", "content": "Summarize the search results"},
            {
                "role": "assistant",
                "content": "Based on my search, here is the summary.",
                "reasoning_content": "Let me organize these results clearly...",
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # No generation prompt - complete message
        )

        # Both content and reasoning should appear
        assert "Based on my search" in prompt, f"{model_id}: Content should appear"
        # Note: reasoning may or may not appear depending on add_generation_prompt
        # This is model-specific behavior we're documenting

        # Should tokenize cleanly
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(tokens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
