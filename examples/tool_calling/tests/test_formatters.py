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

        # Some formatters JSON-encode content, others preserve verbatim
        json_encoding_formatters = {"gpt-oss", "llama3"}
        if formatter_name in json_encoding_formatters:
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

        # Some formatters JSON-encode which escapes unicode - that's OK as long as content is recoverable
        # Check that the core content is present (possibly escaped)
        json_encoding_formatters = {"gpt-oss", "llama3"}
        if formatter_name in json_encoding_formatters:
            # JSON-encoded: check escaped form is present
            assert "greeting" in result, f"{formatter_name}: Unicode content lost"
        else:
            # Direct content: unicode should be preserved
            assert "你好" in result, f"{formatter_name}: Unicode mangled"

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
