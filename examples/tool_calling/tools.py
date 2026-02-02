"""
Tool Registry for slime tool calling.

Provides a simple registry for tools that can be called by the model.
Compatible with OpenAI tools format and SGLang FunctionCallParser.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolSchema:
    """Tool schema for parsing/prompting (serializable, no executable code).

    This represents the "shape" of a tool for the model to understand,
    without including the actual implementation. Used for dataset metadata
    and tool parsing configuration.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema format


@dataclass
class ToolSpec:
    """Full tool specification with executable function.

    Extends ToolSchema with runtime configuration (func, timeout, etc).
    Cannot be serialized - use for runtime registry only.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema format
    func: Callable[..., str | Coroutine[Any, Any, str]]
    timeout: float = 30.0
    max_output_chars: int = 10000

    def to_schema(self) -> ToolSchema:
        """Convert to serializable schema (drops func)."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass
class ToolCall:
    """A parsed tool call from model output."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    raw: str | None = None  # Original text that was parsed


@dataclass
class ToolResult:
    """Result from executing a tool."""

    name: str
    call_id: str | None
    ok: bool
    output: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: int = 0


class ToolRegistry:
    """
    Registry for tools that can be called by the model.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="calculator",
            description="Evaluate a mathematical expression",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
            func=lambda expr: str(eval(expr))
        ))

        # Get OpenAI-compatible tools list for prompt
        tools = registry.to_openai_format()

        # Execute a tool call
        result = await registry.execute(ToolCall(name="calculator", arguments={"expr": "2+2"}))
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._semaphore = asyncio.Semaphore(32)  # Limit concurrent executions

    def register(self, spec: ToolSpec) -> None:
        """Register a tool."""
        if spec.name in self._tools:
            logger.warning(f"Overwriting existing tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get a tool by name."""
        return self._tools.get(name)

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        Convert registry to OpenAI tools format.
        This format is compatible with SGLang FunctionCallParser.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call with timeout and error handling.

        Returns ToolResult with ok=True on success, ok=False on error.
        Never raises - errors are captured in ToolResult.
        """
        start_time = time.perf_counter()

        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(
                name=call.name,
                call_id=call.call_id,
                ok=False,
                output="",
                error_type="ToolNotFoundError",
                error_message=f"Tool '{call.name}' not found in registry",
            )

        try:
            async with self._semaphore:
                # Call the function
                if asyncio.iscoroutinefunction(spec.func):
                    result = await asyncio.wait_for(spec.func(**call.arguments), timeout=spec.timeout)
                else:
                    # Run sync function in thread pool (Python 3.9+)
                    result = await asyncio.wait_for(
                        asyncio.to_thread(spec.func, **call.arguments),
                        timeout=spec.timeout,
                    )

                # Ensure result is string
                if not isinstance(result, str):
                    result = json.dumps(result)

                # Truncate if needed
                if len(result) > spec.max_output_chars:
                    result = result[: spec.max_output_chars] + "\n... [truncated]"

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                return ToolResult(
                    name=call.name,
                    call_id=call.call_id,
                    ok=True,
                    output=result,
                    latency_ms=latency_ms,
                )

        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return ToolResult(
                name=call.name,
                call_id=call.call_id,
                ok=False,
                output="",
                error_type="TimeoutError",
                error_message=f"Tool execution timed out after {spec.timeout}s",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return ToolResult(
                name=call.name,
                call_id=call.call_id,
                ok=False,
                output="",
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=latency_ms,
            )

    async def execute_batch(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls concurrently."""
        return await asyncio.gather(*[self.execute(call) for call in calls])


# ============================================================================
# Example Tools
# ============================================================================


def create_calculator_tool() -> ToolSpec:
    """Create a simple calculator tool for testing."""

    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        # Only allow safe characters
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return ToolSpec(
        name="calculator",
        description="Evaluate a mathematical expression. Only basic arithmetic is supported (+, -, *, /, parentheses).",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        },
        func=calculate,
    )


# ============================================================================
# Tool Binding Registry (Global)
# ============================================================================
# Maps tool names to their factory functions. Allows metadata to reference
# tools by name rather than defining full ToolSpec (which requires func).


ToolFactory = Callable[[], ToolSpec]

# Global registry of available tool factories
TOOL_BINDINGS: dict[str, ToolFactory] = {}


def register_tool_binding(name: str, factory: ToolFactory) -> None:
    """Register a tool factory in the global binding registry."""
    TOOL_BINDINGS[name] = factory


def get_tool_binding(name: str) -> ToolFactory | None:
    """Get a tool factory by name from the global registry."""
    return TOOL_BINDINGS.get(name)


def list_available_tools() -> list[str]:
    """List names of all registered tool bindings."""
    return list(TOOL_BINDINGS.keys())


# Register built-in tools
register_tool_binding("calculator", create_calculator_tool)


def create_registry_from_names(tool_names: list[str]) -> ToolRegistry:
    """Create a ToolRegistry from a list of tool binding names.

    Args:
        tool_names: List of tool names registered in TOOL_BINDINGS

    Returns:
        ToolRegistry with the requested tools

    Raises:
        ValueError: If a tool name is not found in TOOL_BINDINGS
    """
    registry = ToolRegistry()
    missing = []

    for name in tool_names:
        factory = get_tool_binding(name)
        if factory is None:
            missing.append(name)
            continue
        registry.register(factory())

    if missing:
        available = ", ".join(sorted(list_available_tools()))
        raise ValueError(f"Unknown tool(s): {', '.join(missing)}. Available: {available}")

    return registry


def create_default_registry() -> ToolRegistry:
    """Create a registry with default example tools."""
    return create_registry_from_names(["calculator"])


# ============================================================================
# Tool Response Formatters
# ============================================================================
# Format tool execution results for injection back into model context.
# Each formatter matches SGLang's parser name exactly.


def format_qwen(content: str, add_generation_prompt: bool = False, **ctx) -> str:
    """Qwen 2.5 / Qwen 3 / Qwen3-Coder / MIMO / Hermes format.

    Full format with chat template markers:
    <|im_start|>user
    <tool_response>
    {content}
    </tool_response><|im_end|>
    <|im_start|>assistant  (if add_generation_prompt=True)

    Args:
        content: Tool result content
        add_generation_prompt: If True, append assistant start token for continuation
    """
    base = f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    if add_generation_prompt:
        return base + "<|im_start|>assistant\n"
    return base


def format_glm(content: str, add_generation_prompt: bool = False, **ctx) -> str:
    """GLM-4 / GLM-4.5 format (uses observation role).

    Format: <|observation|>\n{content}<|assistant|> (if add_generation_prompt=True)

    Args:
        content: Tool result content
        add_generation_prompt: If True, append assistant start token for continuation
    """
    base = f"<|observation|>\n{content}"
    if add_generation_prompt:
        return base + "<|assistant|>"
    return base


def format_glm47(content: str, add_generation_prompt: bool = False, **ctx) -> str:
    """GLM-4.7 format (uses tool role with <tool_response> wrapper).

    Full format: <|observation|><tool_response>{content}</tool_response><|assistant|><think>

    Args:
        content: Tool result content
        add_generation_prompt: If True, include observation prefix and assistant suffix

    Note for parallel tool calls:
        For multiple tool responses in one turn, call with add_generation_prompt=False
        for all but the last one, then wrap the concatenated results with:
        <|observation|>{all_tool_responses}<|assistant|><think>
    """
    core = f"<tool_response>{content}</tool_response>"
    if add_generation_prompt:
        return f"<|observation|>{core}<|assistant|><think>"
    return core


def format_deepseek_v3(content: str, **ctx) -> str:
    """DeepSeek V3/V3.1 format (special token wrapper)."""
    return f"<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{content}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>"


def format_deepseek_v32(content: str, **ctx) -> str:
    """DeepSeek V3.2 format (DSML-based with function_results wrapper)."""
    return f"\n\n<function_results>\n<result>{content}</result>\n</function_results>"


def format_llama3(content: str, **ctx) -> str:
    """Llama 3.x ipython format.

    The ipython role is used for tool call outputs.
    Content is JSON-encoded to handle special characters properly.
    """
    # JSON-encode content to handle quotes, backslashes, etc.
    encoded_content = json.dumps(content)
    return f'\n<|start_header_id|>ipython<|end_header_id|>\n\n{{"output": {encoded_content}}}<|eot_id|>\n'


def format_mistral(content: str, tool_call_id: str = "", **ctx) -> str:
    """Mistral format. Requires tool_call_id.

    Content is embedded as a JSON value. If content is not valid JSON,
    it will be JSON-encoded as a string.
    """
    if not tool_call_id:
        raise ValueError("format_mistral requires 'tool_call_id'")

    # Ensure content is valid JSON; if not, encode it as a JSON string
    try:
        json.loads(content)
    except (json.JSONDecodeError, TypeError):
        content = json.dumps(content)

    return f'[TOOL_RESULTS] {{"content": {content}, "call_id": "{tool_call_id}"}}[/TOOL_RESULTS]'


def format_gpt_oss(content: str, tool_name: str = "", **ctx) -> str:
    """GPT-OSS Harmony format. Requires tool_name.

    Note: Content is JSON-encoded per the Jinja template's |tojson filter.
    Tool name is prefixed with 'functions.' namespace.
    """
    if not tool_name:
        raise ValueError("format_gpt_oss requires 'tool_name'")
    return f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{json.dumps(content)}<|end|>"


def format_kimi_k2(content: str, tool_call_id: str = "", add_generation_prompt: bool = False, **ctx) -> str:
    """Kimi K2 format (Instruct and Thinking). Requires tool_call_id.

    Full format:
    <|im_system|>tool<|im_middle|>## Return of {tool_call_id}\\n{content}<|im_end|>
    <|im_assistant|>assistant<|im_middle|>  (if add_generation_prompt=True)

    Args:
        content: Tool result content
        tool_call_id: The tool call ID (required, e.g., "functions.get_weather:0")
        add_generation_prompt: If True, append assistant start token for continuation
    """
    if not tool_call_id:
        raise ValueError("format_kimi_k2 requires 'tool_call_id'")
    base = f"<|im_system|>tool<|im_middle|>## Return of {tool_call_id}\n{content}<|im_end|>"
    if add_generation_prompt:
        return base + "<|im_assistant|>assistant<|im_middle|>"
    return base


def format_minimax(content: str, **ctx) -> str:
    """MiniMax M2 format."""
    return f"]~b]tool\n<response>{content}</response>[e~[\n"


# Registry: SGLang parser name -> formatter function
TOOL_RESPONSE_FORMATTERS: Dict[str, Callable[..., str]] = {
    # Qwen family
    "qwen": format_qwen,
    "qwen25": format_qwen,
    "qwen3_coder": format_qwen,
    # GLM family
    "glm": format_glm,
    "glm45": format_glm,
    "glm47": format_glm47,
    # DeepSeek family
    "deepseekv3": format_deepseek_v3,
    "deepseekv31": format_deepseek_v3,  # V3.1 uses same format as V3
    "deepseekv32": format_deepseek_v32,  # V3.2 uses DSML format
    # Complex formats
    "gpt-oss": format_gpt_oss,
    "kimi_k2": format_kimi_k2,
    "minimax-m2": format_minimax,
    # Others
    "mimo": format_qwen,
    "llama3": format_llama3,
    "mistral": format_mistral,
    "hermes": format_qwen,
}


def get_tool_response_formatter(parser_name: str) -> Callable[..., str]:
    """Get formatter for the given parser name."""
    if parser_name not in TOOL_RESPONSE_FORMATTERS:
        available = ", ".join(sorted(TOOL_RESPONSE_FORMATTERS.keys()))
        raise ValueError(f"Unsupported parser: '{parser_name}'. Available parsers: {available}")
    return TOOL_RESPONSE_FORMATTERS[parser_name]


def format_observation(
    result: ToolResult, parser_name: str, add_generation_prompt: bool = True
) -> str:
    """Format tool result for injection back into model context.

    Args:
        result: Tool execution result
        parser_name: Parser/formatter name (e.g., 'qwen25', 'glm47')
        add_generation_prompt: If True, include assistant start token for continuation.
            Set to False for parallel tool calls (except the last one).
    """
    if parser_name not in TOOL_RESPONSE_FORMATTERS:
        available = ", ".join(sorted(TOOL_RESPONSE_FORMATTERS.keys()))
        raise ValueError(f"Unsupported parser: '{parser_name}'. Available parsers: {available}")
    formatter = TOOL_RESPONSE_FORMATTERS[parser_name]
    if result.ok:
        content = result.output
    else:
        content = f"Error ({result.error_type}): {result.error_message}"
    return formatter(
        content=content,
        tool_name=result.name,
        tool_call_id=result.call_id or "",
        add_generation_prompt=add_generation_prompt,
    )
