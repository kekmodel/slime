"""
Tool Calling module for slime.

Provides multi-hop tool calling with SGLang FunctionCallParser integration.
"""

from .tools import (
    ToolSchema,
    ToolSpec,
    ToolCall,
    ToolResult,
    ToolRegistry,
    TOOL_RESPONSE_FORMATTERS,
    TOOL_BINDINGS,
    register_tool_binding,
    get_tool_binding,
    list_available_tools,
    create_registry_from_names,
    create_calculator_tool,
    create_default_registry,
    get_tool_response_formatter,
    format_observation,
)


# Lazy imports for generate module (requires sglang)
# This allows importing tools without sglang dependency
_lazy_loading = False


def __getattr__(name):
    """Lazy import for generate module components."""
    global _lazy_loading

    # Prevent re-entry during lazy loading
    if _lazy_loading:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in (
        "generate",
        "ToolCallingConfig",
        "ModelProfile",
        "MODEL_PROFILES",
        "get_model_profile",
        "parse_tool_calls",
        "extract_tokens_from_logprobs",
        "reward_func",
        "_get_registry",
    ):
        _lazy_loading = True
        try:
            from . import generate as gen_module

            return getattr(gen_module, name)
        finally:
            _lazy_loading = False

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Tool Schema & Spec
    "ToolSchema",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    # Formatters
    "TOOL_RESPONSE_FORMATTERS",
    "get_tool_response_formatter",
    "format_observation",
    # Tool Bindings (executable registry)
    "TOOL_BINDINGS",
    "register_tool_binding",
    "get_tool_binding",
    "list_available_tools",
    "create_registry_from_names",
    "create_calculator_tool",
    "create_default_registry",
    # Generate (lazy loaded - requires sglang)
    "generate",
    "ToolCallingConfig",
    "ModelProfile",
    "MODEL_PROFILES",
    "get_model_profile",
    "parse_tool_calls",
    "extract_tokens_from_logprobs",
    "reward_func",
    "_get_registry",
]
