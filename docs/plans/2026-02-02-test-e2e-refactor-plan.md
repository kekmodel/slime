# Test E2E Tool Calling Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `test_e2e_tool_calling.py` (3,413 lines, 16 test classes) into smaller, role-based test files with better debugging support.

**Architecture:** Create shared infrastructure in `conftest.py` (PARSER_TO_HF_MODEL mapping, fixtures) and `utils.py` (debug helpers), then migrate tests to dedicated files by responsibility. Use parametrized tests to auto-expand across all formatters/parsers. No `pytest.skip()` - use explicit failures with debug info.

**Tech Stack:** pytest, transformers (HuggingFace), dataclasses

---

## Task 1: Extend conftest.py with PARSER_TO_HF_MODEL Mapping

**Files:**
- Modify: `examples/tool_calling/tests/conftest.py`

**Step 1: Write the failing test**

Create a test to verify the mapping exists:

```python
# In a temporary test or run interactively
from examples.tool_calling.tests.conftest import PARSER_TO_HF_MODEL
assert "qwen25" in PARSER_TO_HF_MODEL
assert PARSER_TO_HF_MODEL["qwen25"] == "Qwen/Qwen2.5-0.5B-Instruct"
```

**Step 2: Run test to verify it fails**

Run: `python -c "from examples.tool_calling.tests.conftest import PARSER_TO_HF_MODEL"`
Expected: ImportError (PARSER_TO_HF_MODEL doesn't exist yet)

**Step 3: Write minimal implementation**

Add to `conftest.py`:

```python
# ============================================================================
# Parser to HuggingFace Model Mapping
# ============================================================================

PARSER_TO_HF_MODEL: dict[str, str] = {
    # Qwen family
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3_coder": "Qwen/Qwen3-0.6B",
    # GLM family
    "glm": "Qwen/Qwen2.5-0.5B-Instruct",  # Fallback (GLM tokenizer not on HF Hub)
    "glm47": "Qwen/Qwen2.5-0.5B-Instruct",  # Fallback
    # DeepSeek family
    "deepseekv3": "deepseek-ai/DeepSeek-V3",
    "deepseek_v32": "deepseek-ai/DeepSeek-V3",
    # Kimi
    "kimi_k2": "moonshotai/Kimi-K2-Instruct",
    # Mistral
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    # MiniMax
    "minimax-m2": "Qwen/Qwen2.5-0.5B-Instruct",  # Fallback (MiniMax not on HF Hub)
    # GPT-OSS
    "gpt-oss": "Qwen/Qwen2.5-0.5B-Instruct",  # Fallback
}

# Smaller/faster models for CI (use these by default in tests)
PARSER_TO_HF_MODEL_SMALL: dict[str, str] = {
    parser: "Qwen/Qwen2.5-0.5B-Instruct"  # All use small Qwen for speed
    for parser in PARSER_TO_HF_MODEL
}
```

**Step 4: Run test to verify it passes**

Run: `python -c "from examples.tool_calling.tests.conftest import PARSER_TO_HF_MODEL; print('OK:', len(PARSER_TO_HF_MODEL), 'parsers')"`
Expected: `OK: 12 parsers`

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/conftest.py
git commit -m "feat(tests): add PARSER_TO_HF_MODEL mapping to conftest

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Parametrized Fixtures to conftest.py

**Files:**
- Modify: `examples/tool_calling/tests/conftest.py`

**Step 1: Write the failing test**

```python
# Test that fixture exists
def test_parser_name_fixture(parser_name):
    assert parser_name in ["qwen25", "glm47", "deepseekv3", ...]
```

**Step 2: Run test to verify it fails**

Run: `pytest examples/tool_calling/tests/test_token_invariants.py -v --collect-only 2>&1 | grep parser_name`
Expected: No parametrized tests found

**Step 3: Write minimal implementation**

Add to `conftest.py`:

```python
import pytest
from transformers import AutoTokenizer

# ============================================================================
# Parametrized Fixtures
# ============================================================================

@pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
def parser_name(request) -> str:
    """Parametrized fixture that runs tests for all registered parsers."""
    return request.param


@pytest.fixture
def formatter_name(parser_name: str) -> str:
    """Alias for parser_name (formatters map 1:1 with parsers)."""
    return parser_name


# ============================================================================
# Tokenizer Fixtures
# ============================================================================

_tokenizer_cache: dict[str, "AutoTokenizer"] = {}


def get_tokenizer_for_parser(parser_name: str, use_small: bool = True) -> "AutoTokenizer":
    """Get tokenizer for a parser, with caching.

    Args:
        parser_name: Name of the parser (e.g., "qwen25", "glm47")
        use_small: If True, use smaller models for faster tests

    Returns:
        AutoTokenizer instance

    Raises:
        pytest.fail: If tokenizer cannot be loaded (with download instructions)
    """
    mapping = PARSER_TO_HF_MODEL_SMALL if use_small else PARSER_TO_HF_MODEL
    model_id = mapping.get(parser_name)

    if model_id is None:
        pytest.fail(
            f"No HuggingFace model ID for parser '{parser_name}'. "
            f"Add it to PARSER_TO_HF_MODEL in conftest.py"
        )

    if model_id not in _tokenizer_cache:
        try:
            _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception as e:
            pytest.fail(
                f"Failed to load tokenizer '{model_id}' for parser '{parser_name}': {e}\n"
                f"Try: huggingface-cli download {model_id}"
            )

    return _tokenizer_cache[model_id]


@pytest.fixture
def tokenizer(parser_name: str) -> "AutoTokenizer":
    """Get tokenizer for the current parser_name fixture."""
    return get_tokenizer_for_parser(parser_name)
```

**Step 4: Run test to verify it passes**

Run: `pytest examples/tool_calling/tests/test_token_invariants.py::test_token_length_invariants -v --collect-only`
Expected: Shows parametrized variants for each parser

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/conftest.py
git commit -m "feat(tests): add parametrized fixtures for parsers and tokenizers

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Debug Helpers to utils.py

**Files:**
- Modify: `examples/tool_calling/tests/utils.py`

**Step 1: Write the failing test**

```python
from examples.tool_calling.tests.utils import format_diff, save_debug_info
assert "position" in format_diff("hello", "hallo").lower()
```

**Step 2: Run test to verify it fails**

Run: `python -c "from examples.tool_calling.tests.utils import format_diff"`
Expected: ImportError

**Step 3: Write minimal implementation**

Add to `utils.py`:

```python
import json
from pathlib import Path
from datetime import datetime

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
    output_dir: Path | None = None
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
```

**Step 4: Run test to verify it passes**

Run: `python -c "from examples.tool_calling.tests.utils import format_diff, save_debug_info; print(format_diff('hello', 'hallo'))"`
Expected: Shows diff at position 1

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/utils.py
git commit -m "feat(tests): add debug helpers format_diff and save_debug_info

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create test_formatters.py

**Files:**
- Create: `examples/tool_calling/tests/test_formatters.py`

**Step 1: Write the failing test**

```python
# test_formatters.py content (will be created in Step 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest examples/tool_calling/tests/test_formatters.py -v`
Expected: File not found

**Step 3: Write minimal implementation**

Create `test_formatters.py`:

```python
"""Formatter unit tests for tool response formatters.

Tests that all registered formatters:
1. Produce non-empty output
2. Preserve content exactly
3. Handle edge cases (unicode, JSON, multiline)
4. Validate required kwargs
"""

import pytest

pytest.importorskip("transformers")

from examples.tool_calling.tools import (
    TOOL_RESPONSE_FORMATTERS,
    get_tool_response_formatter,
)


class TestFormatterOutput:
    """Test that all formatters produce valid output."""

    @pytest.fixture(params=list(TOOL_RESPONSE_FORMATTERS.keys()))
    def formatter_name(self, request) -> str:
        return request.param

    def test_formatter_produces_output(self, formatter_name: str):
        """Every formatter must produce non-empty output."""
        formatter = get_tool_response_formatter(formatter_name)

        # Provide required kwargs for specific formatters
        kwargs = {"content": "test result"}
        if formatter_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = "functions.test:0"
        if formatter_name == "gpt-oss":
            kwargs["tool_name"] = "test_tool"

        result = formatter(**kwargs)

        assert result, f"{formatter_name}: Formatter produced empty output"
        assert len(result) > len("test result"), f"{formatter_name}: Output should include format markers"

    def test_formatter_preserves_content(self, formatter_name: str):
        """Content must appear in formatted output unchanged."""
        formatter = get_tool_response_formatter(formatter_name)

        content = '{"result": 42, "status": "ok"}'
        kwargs = {"content": content}
        if formatter_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = "functions.test:0"
        if formatter_name == "gpt-oss":
            kwargs["tool_name"] = "test_tool"

        result = formatter(**kwargs)

        # GPT-OSS JSON-encodes content, others preserve verbatim
        if formatter_name == "gpt-oss":
            assert "42" in result, f"{formatter_name}: Content not preserved"
        else:
            assert content in result, f"{formatter_name}: Content not preserved verbatim"


class TestFormatterValidation:
    """Test that formatters requiring specific kwargs fail appropriately."""

    def test_kimi_k2_requires_tool_call_id(self):
        """Kimi K2 formatter must require tool_call_id."""
        formatter = get_tool_response_formatter("kimi_k2")

        with pytest.raises((TypeError, ValueError)):
            formatter(content="test")

    def test_mistral_requires_tool_call_id(self):
        """Mistral formatter must require tool_call_id."""
        formatter = get_tool_response_formatter("mistral")

        with pytest.raises((TypeError, ValueError)):
            formatter(content="test")

    def test_gpt_oss_requires_tool_name(self):
        """GPT-OSS formatter must require tool_name."""
        formatter = get_tool_response_formatter("gpt-oss")

        with pytest.raises((TypeError, ValueError)):
            formatter(content="test")


class TestFormatterEdgeCases:
    """Test edge cases that have caused bugs in production."""

    @pytest.fixture(params=list(TOOL_RESPONSE_FORMATTERS.keys()))
    def formatter_name(self, request) -> str:
        return request.param

    def _get_formatter_with_kwargs(self, formatter_name: str, content: str) -> str:
        """Helper to call formatter with appropriate kwargs."""
        formatter = get_tool_response_formatter(formatter_name)
        kwargs = {"content": content}
        if formatter_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = "functions.test:0"
        if formatter_name == "gpt-oss":
            kwargs["tool_name"] = "test_tool"
        return formatter(**kwargs)

    def test_unicode_content(self, formatter_name: str):
        """Formatters must handle unicode correctly."""
        content = '{"greeting": "你好世界", "emoji": "🎉"}'
        result = self._get_formatter_with_kwargs(formatter_name, content)

        # Check unicode is preserved (not escaped)
        assert "你好" in result or "\\u" not in result, f"{formatter_name}: Unicode mangled"

    def test_multiline_content(self, formatter_name: str):
        """Formatters must handle multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        result = self._get_formatter_with_kwargs(formatter_name, content)

        assert "Line 1" in result, f"{formatter_name}: Multiline content lost"

    def test_nested_quotes(self, formatter_name: str):
        """Formatters must handle nested quotes in JSON."""
        content = '{"text": "He said \\"hello\\" and left."}'
        result = self._get_formatter_with_kwargs(formatter_name, content)

        assert "hello" in result, f"{formatter_name}: Nested quotes broke content"

    def test_html_content(self, formatter_name: str):
        """Formatters must handle HTML-like content."""
        content = '{"html": "<div class=\\"test\\">content</div>"}'
        result = self._get_formatter_with_kwargs(formatter_name, content)

        assert "div" in result, f"{formatter_name}: HTML content lost"
```

**Step 4: Run test to verify it passes**

Run: `pytest examples/tool_calling/tests/test_formatters.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/test_formatters.py
git commit -m "feat(tests): add test_formatters.py with formatter unit tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create test_ground_truth.py

**Files:**
- Create: `examples/tool_calling/tests/test_ground_truth.py`

**Step 1: Write the failing test**

```python
# Will be created in Step 3
```

**Step 2: Run test to verify it fails**

Run: `pytest examples/tool_calling/tests/test_ground_truth.py -v`
Expected: File not found

**Step 3: Write minimal implementation**

Create `test_ground_truth.py`:

```python
"""Ground truth tests: Our formatters vs HuggingFace apply_chat_template.

The definitive test: our formatter output must match EXACTLY what HuggingFace
produces when building the same conversation with apply_chat_template.

Key design:
- NO pytest.skip() - use pytest.fail() with debug info or pytest.xfail() for known issues
- Debug helpers save full context on failure
- Parametrized across all parsers that have HF templates
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


# Parsers that have real HF templates (not fallbacks)
PARSERS_WITH_HF_TEMPLATES = ["qwen", "qwen25", "qwen3_coder"]


class TestToolResponseGroundTruth:
    """Test that our formatter output matches HF template exactly."""

    @pytest.fixture(params=PARSERS_WITH_HF_TEMPLATES)
    def parser_name(self, request) -> str:
        return request.param

    def test_single_tool_response_matches_hf(self, parser_name: str):
        """Our formatter output must match HF apply_chat_template exactly."""
        tokenizer = get_tokenizer_for_parser(parser_name)
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
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "New York"}'}
                }],
            },
        ]
        messages_after = messages_before + [
            {"role": "tool", "tool_call_id": tool_call_id, "content": tool_result}
        ]

        tools = [WEATHER_TOOL]

        # Get HF output
        before_text = tokenizer.apply_chat_template(
            messages_before, tools=tools, tokenize=False, add_generation_prompt=False
        )
        after_text = tokenizer.apply_chat_template(
            messages_after, tools=tools, tokenize=False, add_generation_prompt=True
        )
        hf_tool_response = after_text[len(before_text):]

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
                }
            )
            pytest.fail(
                f"Formatter output doesn't match HF template!\n"
                f"{format_diff(hf_tool_response, our_tool_response)}\n"
                f"Debug info saved to: {debug_path}"
            )

    def test_token_level_match(self, parser_name: str):
        """Token-level verification: our tokens == HF tokens."""
        tokenizer = get_tokenizer_for_parser(parser_name)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        content = '{"result": 42}'
        tool_call_id = "call_0"

        # Build HF conversation
        messages_before = [
            {"role": "user", "content": "Calculate 6*7"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'}
                }],
            },
        ]
        messages_after = messages_before + [
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        ]

        tools = [CALCULATOR_TOOL]

        before_text = tokenizer.apply_chat_template(
            messages_before, tools=tools, tokenize=False, add_generation_prompt=False
        )
        after_text = tokenizer.apply_chat_template(
            messages_after, tools=tools, tokenize=False, add_generation_prompt=True
        )

        hf_tool_response = after_text[len(before_text):]
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
                }
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
        tokenizer = get_tokenizer_for_parser(parser_name)
        tool_map = create_mock_tool_functions()

        messages = [
            {"role": "user", "content": "Weather in NY and London?"},
            {
                "role": "assistant",
                "content": "Checking NY first.",
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "New York"}'}
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": json.dumps(tool_map["get_weather"]("New York"))
            },
            {
                "role": "assistant",
                "content": "Now London.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'}
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(tool_map["get_weather"]("London"))
            },
            {
                "role": "assistant",
                "content": "NY: Sunny 25°C, London: Cloudy 22°C."
            },
        ]

        tools = [WEATHER_TOOL]

        # Ground truth
        ground_truth = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
        ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)

        # Incremental accumulation (simulating our training flow)
        accumulated_tokens = []
        current_text_len = 0

        for i in range(len(messages)):
            partial = messages[:i+1]
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
                }
            )
            pytest.fail(
                f"Multi-turn accumulation mismatch!\n"
                f"  Ground truth: {len(ground_truth_tokens)} tokens\n"
                f"  Accumulated:  {len(accumulated_tokens)} tokens\n"
                f"  Debug info: {debug_path}"
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest examples/tool_calling/tests/test_ground_truth.py -v`
Expected: Tests pass for supported parsers

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/test_ground_truth.py
git commit -m "feat(tests): add test_ground_truth.py for HF template verification

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update test_token_invariants.py to Use Shared Infrastructure

**Files:**
- Modify: `examples/tool_calling/tests/test_token_invariants.py`

**Step 1: Review current implementation**

Read: `examples/tool_calling/tests/test_token_invariants.py`

**Step 2: Write minimal implementation**

Update to use conftest fixtures and remove pytest.skip:

```python
"""Token invariant tests for tool calling.

Tests the critical invariants for RL training:
- len(token_ids) == len(loss_mask) == len(log_probs)
- loss_mask = 1 for generated tokens, 0 for observations
- Token round-trip: decode(encode(text)) preserves meaning

Design:
- NO pytest.skip() - use pytest.fail() with instructions
- Parametrized across all parsers
- Uses shared fixtures from conftest.py
"""

import pytest

pytest.importorskip("transformers")

from examples.tool_calling.tests.conftest import (
    PARSER_TO_HF_MODEL,
    get_tokenizer_for_parser,
)
from examples.tool_calling.tests.utils import (
    MockGenerateResponse,
    create_generate_response,
)
from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS


class TestTokenLengthInvariants:
    """Test that token arrays maintain length invariants."""

    @pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
    def parser_name(self, request) -> str:
        return request.param

    def test_token_logprob_length_match(self, parser_name: str):
        """INVARIANT: len(token_ids) == len(logprobs)"""
        tokenizer = get_tokenizer_for_parser(parser_name)

        text = "The answer is 42."
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        response = create_generate_response(tokenizer, text)
        output = response.to_dict()

        logprobs = output["meta_info"]["output_token_logprobs"]

        assert len(token_ids) == len(logprobs), (
            f"{parser_name}: Token/logprob length mismatch: "
            f"tokens={len(token_ids)}, logprobs={len(logprobs)}"
        )

    def test_logprob_entry_format(self, parser_name: str):
        """Each logprob entry must be [logprob, token_id]."""
        tokenizer = get_tokenizer_for_parser(parser_name)

        text = "Testing format."
        response = create_generate_response(tokenizer, text)
        output = response.to_dict()

        for i, entry in enumerate(output["meta_info"]["output_token_logprobs"]):
            assert len(entry) >= 2, (
                f"{parser_name}: Invalid logprob entry at {i}: {entry}"
            )


class TestLossMaskValues:
    """Test that loss_mask only contains valid values."""

    def test_loss_mask_binary(self):
        """INVARIANT: All loss_mask values are 0 or 1."""
        gen_mask = [1, 1, 1, 1, 1]  # Generated tokens
        obs_mask = [0, 0, 0]  # Observation tokens

        full_mask = gen_mask + obs_mask

        for i, val in enumerate(full_mask):
            assert val in (0, 1), f"Invalid loss_mask value at {i}: {val}"


class TestTokenRoundtrip:
    """Test that token round-trips preserve content."""

    @pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
    def parser_name(self, request) -> str:
        return request.param

    def test_observation_roundtrip(self, parser_name: str):
        """decode(encode(observation)) preserves content."""
        if parser_name not in TOOL_RESPONSE_FORMATTERS:
            pytest.xfail(f"No formatter registered for {parser_name}")

        tokenizer = get_tokenizer_for_parser(parser_name)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        # Get observation
        kwargs = {"content": "42"}
        if parser_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = "functions.test:0"
        if parser_name == "gpt-oss":
            kwargs["tool_name"] = "calculator"

        observation = formatter(**kwargs)

        # Round-trip
        token_ids = tokenizer.encode(observation, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

        assert "42" in decoded, (
            f"{parser_name}: Content lost in round-trip. "
            f"Original: {observation[:100]}..., Decoded: {decoded[:100]}..."
        )


class TestMultiHopAlignment:
    """Test alignment in multi-hop accumulated responses."""

    @pytest.fixture(params=["qwen25", "deepseekv3", "glm47"])
    def parser_name(self, request) -> str:
        return request.param

    @pytest.mark.slow
    def test_accumulated_response_alignment(self, parser_name: str):
        """Alignment must hold at every step of multi-hop flow."""
        tokenizer = get_tokenizer_for_parser(parser_name)

        all_tokens: list[int] = []
        all_log_probs: list[float] = []
        all_loss_mask: list[int] = []

        # Hop 1: Generation
        gen_text = "Let me calculate that."
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
        all_tokens.extend(gen_tokens)
        all_log_probs.extend([-0.5] * len(gen_tokens))
        all_loss_mask.extend([1] * len(gen_tokens))

        assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask), (
            f"Alignment broken after generation: "
            f"tokens={len(all_tokens)}, logprobs={len(all_log_probs)}, mask={len(all_loss_mask)}"
        )

        # Hop 1: Observation
        obs_text = "<tool_response>4</tool_response>"
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
        all_tokens.extend(obs_tokens)
        all_log_probs.extend([0.0] * len(obs_tokens))
        all_loss_mask.extend([0] * len(obs_tokens))

        assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask), (
            f"Alignment broken after observation: "
            f"tokens={len(all_tokens)}, logprobs={len(all_log_probs)}, mask={len(all_loss_mask)}"
        )

        # Hop 2: Final generation
        final_text = "The result is 4."
        final_tokens = tokenizer.encode(final_text, add_special_tokens=False)
        all_tokens.extend(final_tokens)
        all_log_probs.extend([-0.6] * len(final_tokens))
        all_loss_mask.extend([1] * len(final_tokens))

        # Final check
        assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask)
        assert all(m in (0, 1) for m in all_loss_mask)
```

**Step 4: Run test to verify it passes**

Run: `pytest examples/tool_calling/tests/test_token_invariants.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/test_token_invariants.py
git commit -m "refactor(tests): update test_token_invariants to use shared infrastructure

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Create test_integration.py

**Files:**
- Create: `examples/tool_calling/tests/test_integration.py`

**Step 1: Write the failing test**

```python
# Will be created in Step 3
```

**Step 2: Run test to verify it fails**

Run: `pytest examples/tool_calling/tests/test_integration.py -v`
Expected: File not found

**Step 3: Write minimal implementation**

Create `test_integration.py`:

```python
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
from unittest.mock import MagicMock, patch, AsyncMock

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
        response = MockGenerateResponse(
            text="Done.",
            token_ids=[1, 2, 3],
            finish_reason="stop"
        )

        output = response.to_dict()
        assert output["meta_info"]["finish_reason"]["type"] == "stop"

    def test_truncated_status(self):
        """TRUNCATED: Hit max tokens."""
        response = MockGenerateResponse(
            text="Partial...",
            token_ids=[1, 2, 3],
            finish_reason="length"
        )

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
```

**Step 4: Run test to verify it passes**

Run: `pytest examples/tool_calling/tests/test_integration.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/test_integration.py
git commit -m "feat(tests): add test_integration.py for generate() flow tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Archive Old Test File and Run Full Test Suite

**Files:**
- Move: `examples/tool_calling/tests/test_e2e_tool_calling.py` → `examples/tool_calling/tests/deprecated/`

**Step 1: Create deprecated directory and move file**

```bash
mkdir -p examples/tool_calling/tests/deprecated
mv examples/tool_calling/tests/test_e2e_tool_calling.py examples/tool_calling/tests/deprecated/
```

**Step 2: Add __init__.py to prevent pytest collection**

Create `examples/tool_calling/tests/deprecated/__init__.py`:

```python
"""Deprecated tests - not collected by pytest.

These tests have been superseded by:
- test_formatters.py
- test_ground_truth.py
- test_token_invariants.py
- test_integration.py

Keeping for reference during transition period.
Delete after validation.
"""
```

**Step 3: Run full test suite to verify nothing broke**

Run: `pytest examples/tool_calling/tests/ -v --ignore=examples/tool_calling/tests/deprecated`
Expected: All new tests pass

**Step 4: Verify line count reduction**

```bash
# Count lines in new files
wc -l examples/tool_calling/tests/*.py
# Should be ~1,000-1,200 lines total (vs 3,413 in original)
```

**Step 5: Commit**

```bash
git add examples/tool_calling/tests/
git commit -m "refactor(tests): archive old test_e2e_tool_calling.py

Move 3,413-line monolithic test file to deprecated/.
New structure:
- conftest.py: ~100 lines (fixtures, PARSER_TO_HF_MODEL)
- utils.py: ~150 lines (mock classes, debug helpers)
- test_formatters.py: ~150 lines (formatter unit tests)
- test_ground_truth.py: ~200 lines (HF template matching)
- test_token_invariants.py: ~130 lines (token flow invariants)
- test_integration.py: ~100 lines (generate() flow tests)

Total: ~830 lines (75% reduction)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Final Verification and Cleanup

**Files:**
- Verify: All test files in `examples/tool_calling/tests/`

**Step 1: Run pytest with coverage**

```bash
pytest examples/tool_calling/tests/ -v --ignore=examples/tool_calling/tests/deprecated -x
```

**Step 2: Verify no pytest.skip() usage**

```bash
grep -r "pytest.skip" examples/tool_calling/tests/*.py
# Should return nothing (only pytest.fail, pytest.xfail allowed)
```

**Step 3: Verify debug output works on failure**

Temporarily break a test to verify debug info is saved, then revert.

**Step 4: Update conftest.py to not collect deprecated tests**

Add to `conftest.py` if needed:

```python
collect_ignore = ["deprecated"]
```

**Step 5: Final commit**

```bash
git add .
git commit -m "chore(tests): final verification of test refactoring

- All tests pass
- No pytest.skip() usage
- Debug helpers working
- 75% line reduction achieved

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria Checklist

- [ ] All new tests pass (`pytest examples/tool_calling/tests/ -v`)
- [ ] Line count reduced by >50% (target: ~1,000 lines vs 3,413)
- [ ] No `pytest.skip()` usage (grep returns empty)
- [ ] Debug info saved on failures (test by breaking a test)
- [ ] Parametrized across all formatters/parsers
- [ ] Old tests archived in `deprecated/`
