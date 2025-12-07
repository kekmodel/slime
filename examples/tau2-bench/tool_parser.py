"""
Tool Call Parser for SLIME integration with tau2.

Uses sglang's FunctionCallParser when available (GPU environment).
Falls back to standalone implementation for CPU testing.

Supports:
- qwen/qwen25: <tool_call>{"name":"func", "arguments":{...}}</tool_call>
- llama3: <|python_tag|>{"name":"func", "parameters":{...}}
- kimi_k2: <|tool_call_begin|>{"name":"func", "arguments":{...}}<|tool_call_end|>
- gpt-oss: <|channel|>commentary to=...<|call|>
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Try to import sglang (requires GPU/triton)
_SGLANG_AVAILABLE = False
_sglang_FunctionCallParser = None
_sglang_Tool = None
_sglang_Function = None

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser as _sglang_FunctionCallParser
    from sglang.srt.managers.io_struct import Function as _sglang_Function, Tool as _sglang_Tool
    _SGLANG_AVAILABLE = True
    logger.info("sglang FunctionCallParser available")
except ImportError as e:
    logger.info(f"sglang not available ({e}), using fallback parser")


@dataclass
class ToolCallItem:
    """Parsed tool call item."""
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing tool calls from LLM response."""
    success: bool
    normal_text: str
    calls: list[dict[str, Any]]
    error: str | None = None


class BaseDetector:
    """Base class for format detectors."""

    bot_token: str = ""
    eot_token: str = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str) -> tuple[str, list[ToolCallItem]]:
        raise NotImplementedError()


class Qwen25Detector(BaseDetector):
    """
    Detector for Qwen 2.5 / Qwen 3 format.

    Format: <tool_call>
    {"name":"func", "arguments":{...}}
    </tool_call>
    """

    bot_token = "<tool_call>"
    eot_token = "</tool_call>"

    def detect_and_parse(self, text: str) -> tuple[str, list[ToolCallItem]]:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return normal_text, []

        # Pattern: <tool_call>\n?{...}\n?</tool_call>
        pattern = rf"{re.escape(self.bot_token)}\n?(.*?)\n?{re.escape(self.eot_token)}"
        matches = re.findall(pattern, text, re.DOTALL)

        calls = []
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                calls.append(ToolCallItem(
                    name=parsed.get("name", ""),
                    parameters=parsed.get("arguments", parsed.get("parameters", {})),
                ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

        return normal_text, calls


class Llama32Detector(BaseDetector):
    """
    Detector for Llama 3.2 format.

    Format: <|python_tag|>{"name":"func", "parameters":{...}}<|eom_id|>
    """

    bot_token = "<|python_tag|>"
    eot_token = "<|eom_id|>"

    def detect_and_parse(self, text: str) -> tuple[str, list[ToolCallItem]]:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return normal_text, []

        # Extract JSON between tokens
        pattern = rf"{re.escape(self.bot_token)}(.*?)(?:{re.escape(self.eot_token)}|$)"
        matches = re.findall(pattern, text, re.DOTALL)

        calls = []
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                # Llama format can be single call or array
                if isinstance(parsed, list):
                    for item in parsed:
                        calls.append(ToolCallItem(
                            name=item.get("name", ""),
                            parameters=item.get("parameters", item.get("arguments", {})),
                        ))
                else:
                    calls.append(ToolCallItem(
                        name=parsed.get("name", ""),
                        parameters=parsed.get("parameters", parsed.get("arguments", {})),
                    ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

        return normal_text, calls


class KimiK2Detector(BaseDetector):
    """
    Detector for Kimi K2 format.

    Format: <|tool_call_begin|>{"name":"func", "arguments":{...}}<|tool_call_end|>
    """

    bot_token = "<|tool_call_begin|>"
    eot_token = "<|tool_call_end|>"

    def detect_and_parse(self, text: str) -> tuple[str, list[ToolCallItem]]:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return normal_text, []

        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        matches = re.findall(pattern, text, re.DOTALL)

        calls = []
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                calls.append(ToolCallItem(
                    name=parsed.get("name", ""),
                    parameters=parsed.get("arguments", parsed.get("parameters", {})),
                ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

        return normal_text, calls


class MistralDetector(BaseDetector):
    """
    Detector for Mistral format.

    Format: [TOOL_CALLS] [{"name":"func", "arguments":{...}}]
    """

    bot_token = "[TOOL_CALLS]"
    eot_token = ""

    def detect_and_parse(self, text: str) -> tuple[str, list[ToolCallItem]]:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return normal_text, []

        # Extract JSON array after [TOOL_CALLS]
        after_token = text[idx + len(self.bot_token):].strip()

        calls = []
        try:
            # Find JSON array
            if after_token.startswith("["):
                bracket_count = 0
                end_idx = 0
                for i, c in enumerate(after_token):
                    if c == "[":
                        bracket_count += 1
                    elif c == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break

                json_str = after_token[:end_idx]
                parsed = json.loads(json_str)

                for item in parsed:
                    calls.append(ToolCallItem(
                        name=item.get("name", ""),
                        parameters=item.get("arguments", item.get("parameters", {})),
                    ))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Mistral tool calls: {e}")

        return normal_text, calls


class GptOssDetector(BaseDetector):
    """
    Detector for GPT-OSS / T4 format (used by some OpenAI-compatible models).

    Format: <|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{args}<|call|>

    Example:
    <|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "NYC"}<|call|>
    """

    bot_token = "<|channel|>commentary to="
    eot_token = "<|call|>"

    def __init__(self):
        super().__init__()
        # Pattern to extract function name and JSON
        self.tool_extract_pattern = re.compile(
            r"<\|channel\|>commentary\s+to=([a-zA-Z_][a-zA-Z0-9_.]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def detect_and_parse(self, text: str) -> tuple[str, list[ToolCallItem]]:
        if self.bot_token not in text:
            return text, []

        # Find normal text before first tool call
        idx = text.find("<|channel|>commentary to=")
        normal_text = text[:idx].strip() if idx != -1 else ""

        # Find all tool calls
        matches = self.tool_extract_pattern.findall(text)

        calls = []
        for full_name, json_content in matches:
            # Extract function name (last part after .)
            func_name = full_name.split(".")[-1] if "." in full_name else full_name

            try:
                arguments = json.loads(json_content.strip()) if json_content.strip() else {}
                calls.append(ToolCallItem(
                    name=func_name,
                    parameters=arguments,
                ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse gpt-oss tool call JSON: {e}")
                continue

        return normal_text, calls


# Parser registry
DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "qwen": Qwen25Detector,
    "qwen25": Qwen25Detector,
    "qwen3": Qwen25Detector,
    "llama3": Llama32Detector,
    "llama32": Llama32Detector,
    "kimi_k2": KimiK2Detector,
    "mistral": MistralDetector,
    "gpt-oss": GptOssDetector,
    "gpt_oss": GptOssDetector,
}


def _parse_with_sglang(
    response: str,
    tools_info: list[dict[str, Any]],
    parser_type: str,
) -> ParseResult:
    """Parse using sglang's FunctionCallParser (GPU environment)."""
    # Convert tools_info to sglang Tool format
    tools_list = [
        _sglang_Tool(
            function=_sglang_Function(
                name=tool["function"]["name"],
                description=tool["function"].get("description", ""),
                parameters=tool["function"].get("parameters", {}),
            ),
            type=tool.get("type", "function"),
        )
        for tool in tools_info
    ]

    # Create parser and parse response
    parser = _sglang_FunctionCallParser(tools=tools_list, tool_call_parser=parser_type)
    normal_text, calls = parser.parse_non_stream(response)

    # Convert pydantic objects to dictionaries
    calls_dicts = []
    for call in calls:
        call_dict = call.model_dump()
        # Normalize to expected format
        params = call_dict.get("parameters", "{}")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        calls_dicts.append({
            "name": call_dict.get("name", ""),
            "parameters": params,
        })

    return ParseResult(
        success=True,
        normal_text=normal_text,
        calls=calls_dicts,
        error=None,
    )


def _parse_with_fallback(
    response: str,
    tools_info: list[dict[str, Any]] | None,
    parser_type: str,
) -> ParseResult:
    """Parse using fallback detector (CPU environment)."""
    detector_class = DETECTOR_REGISTRY.get(parser_type)
    if not detector_class:
        return ParseResult(
            success=False,
            normal_text=response,
            calls=[],
            error=f"Unsupported parser type: {parser_type}. "
                  f"Available: {list(DETECTOR_REGISTRY.keys())}",
        )

    detector = detector_class()
    normal_text, calls = detector.detect_and_parse(response)

    # Convert ToolCallItem to dict
    calls_dicts = [
        {"name": call.name, "parameters": call.parameters}
        for call in calls
    ]

    # Validate against tools_info if provided
    if tools_info and calls_dicts:
        valid_names = {t["function"]["name"] for t in tools_info}
        calls_dicts = [c for c in calls_dicts if c["name"] in valid_names]

    return ParseResult(
        success=True,
        normal_text=normal_text,
        calls=calls_dicts,
        error=None,
    )


def parse_tool_calls(
    response: str,
    tools_info: list[dict[str, Any]] | None = None,
    parser_type: str = "qwen",
) -> ParseResult:
    """
    Parse tool calls from LLM response.

    Uses sglang's FunctionCallParser when available (GPU environment).
    Falls back to standalone implementation for CPU testing.

    Args:
        response: Raw response text from LLM
        tools_info: List of tool specifications (optional, for validation)
        parser_type: Parser type to use (qwen, llama3, kimi_k2, mistral, gpt-oss)

    Returns:
        ParseResult containing parsed tool calls
    """
    try:
        # Use sglang if available and tools_info provided
        if _SGLANG_AVAILABLE and tools_info:
            return _parse_with_sglang(response, tools_info, parser_type)
        else:
            return _parse_with_fallback(response, tools_info, parser_type)
    except Exception as e:
        logger.error(f"Error parsing tool calls: {e}")
        return ParseResult(
            success=False,
            normal_text=response,
            calls=[],
            error=str(e),
        )


def is_sglang_available() -> bool:
    """Check if sglang FunctionCallParser is available."""
    return _SGLANG_AVAILABLE


class ToolCallAdapter:
    """Adapter class for tool call parsing."""

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        parser_type: str = "qwen",
    ):
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse(self, response: str) -> ParseResult:
        """Parse tool calls from response."""
        return parse_tool_calls(response, self.tools_info, self.parser_type)


def create_tool_adapter(
    tools_info: list[dict[str, Any]],
    parser_type: str = "qwen",
) -> ToolCallAdapter:
    """Factory function to create a ToolCallAdapter."""
    return ToolCallAdapter(tools_info, parser_type)
