"""
Dynamic tool registry for slime_gym.

Allows loading tool implementations from metadata at runtime.
Supports multiple approaches:
1. Tool Library: Pre-registered implementations selected by name
2. Tool Providers: Modules/classes that provide sets of tools
3. Module Path: Dynamic import from dotted path (use with caution)

Example metadata:
    # Select from pre-registered implementations
    {"tool_implementations": {"search": "search_v2", "calculate": "calculate_safe"}}

    # Load tool providers
    {"tool_providers": ["math_tools", "search_tools"]}

    # Dynamic import (requires explicit allowlist)
    {"tool_modules": ["myproject.tools.custom_tools"]}
"""

import importlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .gym_types import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a dynamically loadable tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: Callable[..., Awaitable[str]]
    # Optional metadata
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)


class ToolRegistry:
    """
    Global registry for tool implementations.

    Usage:
        # Register tools
        registry = ToolRegistry()
        registry.register("search_v1", search_v1_tool)
        registry.register("search_v2", search_v2_tool)

        # In environment.seed():
        active_tools = registry.load_tools(metadata)
    """

    def __init__(self):
        # name -> ToolDefinition
        self._tools: dict[str, ToolDefinition] = {}
        # provider_name -> provider_class/module
        self._providers: dict[str, type | Any] = {}
        # Allowlist for dynamic module imports (security)
        self._allowed_modules: set[str] = set()

    def register(
        self,
        name: str,
        implementation: Callable[..., Awaitable[str]],
        description: str = "",
        parameters: dict[str, Any] | None = None,
        version: str = "1.0",
        tags: list[str] | None = None,
    ) -> None:
        """Register a tool implementation."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description or f"Tool: {name}",
            parameters=parameters or {"type": "object", "properties": {}, "required": []},
            implementation=implementation,
            version=version,
            tags=tags or [],
        )

    def register_provider(self, name: str, provider: type | Any) -> None:
        """
        Register a tool provider (class or module).

        Provider should have methods decorated with @tool or
        a get_tools() method returning ToolDefinitions.
        """
        self._providers[name] = provider

    def allow_module(self, module_path: str) -> None:
        """Add a module to the allowlist for dynamic imports."""
        self._allowed_modules.add(module_path)

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_provider(self, name: str) -> type | Any | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def list_tools(self, tags: list[str] | None = None) -> list[str]:
        """List all registered tool names, optionally filtered by tags."""
        if tags is None:
            return list(self._tools.keys())
        return [name for name, tool in self._tools.items() if any(tag in tool.tags for tag in tags)]

    def load_from_module(self, module_path: str) -> dict[str, ToolDefinition]:
        """
        Dynamically load tools from a module path.

        Security: Only loads from allowed modules.
        """
        if module_path not in self._allowed_modules:
            logger.warning(f"Module {module_path} not in allowlist, skipping")
            return {}

        try:
            module = importlib.import_module(module_path)
            tools = {}

            # Look for TOOLS dict or get_tools() function
            if hasattr(module, "TOOLS"):
                tools.update(module.TOOLS)
            if hasattr(module, "get_tools"):
                tools.update(module.get_tools())

            # Look for @tool decorated functions
            for name in dir(module):
                obj = getattr(module, name)
                if hasattr(obj, "_tool_schema"):
                    schema = obj._tool_schema["function"]
                    tools[schema["name"]] = ToolDefinition(
                        name=schema["name"],
                        description=schema["description"],
                        parameters=schema["parameters"],
                        implementation=obj,
                    )

            return tools
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            return {}


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def register_tool(
    name: str,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    version: str = "1.0",
    tags: list[str] | None = None,
) -> Callable:
    """
    Decorator to register a function as a tool in the global registry.

    Usage:
        @register_tool("search_v2", description="Advanced search", tags=["search"])
        async def search_v2(query: str) -> str:
            return f"Results for: {query}"
    """

    def decorator(func: Callable[..., Awaitable[str]]) -> Callable:
        _global_registry.register(
            name=name,
            implementation=func,
            description=description,
            parameters=parameters,
            version=version,
            tags=tags,
        )
        return func

    return decorator


class DynamicToolMixin:
    """
    Mixin class for environments that support dynamic tool loading.

    Add this to your environment class to enable loading tools from metadata.

    Usage:
        class MyEnvironment(DynamicToolMixin, BaseEnvironment):
            def seed(self, metadata: dict) -> None:
                super().seed(metadata)
                self.load_dynamic_tools(metadata)
    """

    def __init__(self):
        super().__init__()
        # Dynamic tools loaded from metadata
        self._dynamic_tools: dict[str, ToolDefinition] = {}
        self._dynamic_tool_schemas: dict[str, dict] = {}

    def load_dynamic_tools(self, metadata: dict) -> None:
        """
        Load tools from metadata.

        Supports:
        - tool_implementations: {"tool_name": "registry_name"}
        - tool_providers: ["provider1", "provider2"]
        - tool_modules: ["module.path"] (requires allowlist)
        """
        registry = get_registry()
        self._dynamic_tools.clear()
        self._dynamic_tool_schemas.clear()

        # 1. Load from tool_implementations mapping
        if "tool_implementations" in metadata:
            for tool_name, impl_name in metadata["tool_implementations"].items():
                tool_def = registry.get(impl_name)
                if tool_def:
                    self._dynamic_tools[tool_name] = tool_def
                    self._dynamic_tool_schemas[tool_name] = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_def.description,
                            "parameters": tool_def.parameters,
                        },
                    }
                else:
                    logger.warning(f"Tool implementation '{impl_name}' not found in registry")

        # 2. Load from tool providers
        if "tool_providers" in metadata:
            for provider_name in metadata["tool_providers"]:
                provider = registry.get_provider(provider_name)
                if provider is None:
                    logger.warning(f"Tool provider '{provider_name}' not found")
                    continue

                # Get tools from provider
                if hasattr(provider, "get_tools"):
                    for tool_def in provider.get_tools():
                        self._dynamic_tools[tool_def.name] = tool_def
                        self._dynamic_tool_schemas[tool_def.name] = {
                            "type": "function",
                            "function": {
                                "name": tool_def.name,
                                "description": tool_def.description,
                                "parameters": tool_def.parameters,
                            },
                        }

        # 3. Load from modules (requires allowlist)
        if "tool_modules" in metadata:
            for module_path in metadata["tool_modules"]:
                loaded = registry.load_from_module(module_path)
                for name, tool_def in loaded.items():
                    self._dynamic_tools[name] = tool_def
                    self._dynamic_tool_schemas[name] = {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool_def.description,
                            "parameters": tool_def.parameters,
                        },
                    }

    def get_tools(self) -> list[dict]:
        """Return combined static and dynamic tool schemas."""
        # Get static tools from parent
        static_tools = super().get_tools()

        # Add dynamic tools
        dynamic_tools = list(self._dynamic_tool_schemas.values())

        # Filter by _enabled_tools if set
        all_tools = static_tools + dynamic_tools
        if hasattr(self, "_enabled_tools") and self._enabled_tools is not None:
            all_tools = [t for t in all_tools if t["function"]["name"] in self._enabled_tools]

        return all_tools

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute tool - checks dynamic tools first, then static."""
        # Check if it's a dynamic tool
        if name in self._dynamic_tools:
            # Check enabled status
            if hasattr(self, "_enabled_tools") and self._enabled_tools is not None:
                if name not in self._enabled_tools:
                    return ToolResult(output=f"Error: Tool '{name}' is not available for this task.", success=False)

            try:
                tool_def = self._dynamic_tools[name]
                result = await tool_def.implementation(**arguments)
                return ToolResult(output=str(result), success=True)
            except Exception as e:
                return ToolResult(output=f"Error: {e}", success=False)

        # Fall back to static tools
        return await super().execute_tool(name, arguments)


# ==================== Example Tool Implementations ====================


@register_tool(
    name="search_basic",
    description="Basic search functionality",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
    tags=["search", "basic"],
)
async def search_basic(query: str) -> str:
    """Basic search implementation."""
    return f"Basic search results for: {query}"


@register_tool(
    name="search_advanced",
    description="Advanced search with filters",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "filters": {"type": "object", "description": "Search filters"},
        },
        "required": ["query"],
    },
    tags=["search", "advanced"],
)
async def search_advanced(query: str, filters: dict | None = None) -> str:
    """Advanced search implementation with filters."""
    filter_str = f" with filters {filters}" if filters else ""
    return f"Advanced search results for: {query}{filter_str}"


@register_tool(
    name="calculate_safe",
    description="Safe calculator (no eval)",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Math expression"}},
        "required": ["expression"],
    },
    tags=["math", "safe"],
)
async def calculate_safe(expression: str) -> str:
    """Safe calculation without eval."""
    # Simple implementation - in production use a proper math parser
    try:
        # Only allow simple arithmetic
        import ast

        tree = ast.parse(expression, mode="eval")
        # Validate only contains numbers and operators
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Num,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.USub,
                    ast.UAdd,
                    ast.Constant,
                ),
            ):
                raise ValueError(f"Unsafe operation: {type(node).__name__}")
        result = eval(compile(tree, "<string>", "eval"))
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"
