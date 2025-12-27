"""
Dynamic tool registry for slime_gym.

Allows loading tool implementations from metadata at runtime.
Supports:
1. Tool Library: Pre-registered implementations selected by name
2. Tool Providers: Modules/classes that provide sets of tools
3. Module Path: Dynamic import from dotted path (requires allowlist)
"""

import ast
import importlib
import logging
import operator
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from .types import ToolDefinition, ToolResult

if TYPE_CHECKING:
    from .types import ExecutionState


class _HasBaseEnvironmentAttrs(Protocol):
    """Protocol for BaseEnvironment attributes accessed by DynamicToolMixin."""

    _enabled_tools: set[str] | None
    state: "ExecutionState"

    def get_tools(self) -> list[dict[str, Any]]: ...
    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for tool implementations.

    Usage:
        registry = ToolRegistry()
        registry.register("search_v1", search_v1_impl, description="...")

        # In environment.setup():
        tools = registry.get_tools(["search_v1", "calculate"])
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._providers: dict[str, type | Any] = {}
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
        """Register a tool provider (class or module with get_tools())."""
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
        """List registered tool names, optionally filtered by tags."""
        if tags is None:
            return list(self._tools.keys())
        return [name for name, tool in self._tools.items() if any(tag in tool.tags for tag in tags)]

    def load_from_module(self, module_path: str) -> dict[str, ToolDefinition]:
        """Dynamically load tools from an allowed module."""
        if module_path not in self._allowed_modules:
            logger.warning(f"Module {module_path} not in allowlist, skipping")
            return {}

        try:
            module = importlib.import_module(module_path)
            tools = {}

            if hasattr(module, "TOOLS"):
                tools.update(module.TOOLS)
            if hasattr(module, "get_tools"):
                tools.update(module.get_tools())

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


# Global registry with lazy initialization
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _register_default_tools(_global_registry)
    return _global_registry


def set_registry(registry: ToolRegistry) -> None:
    """Set the global registry (useful for testing)."""
    global _global_registry
    _global_registry = registry


ToolImplFunc = Callable[..., Awaitable[str]]


def register_tool(
    name: str,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    version: str = "1.0",
    tags: list[str] | None = None,
) -> Callable[[ToolImplFunc], ToolImplFunc]:
    """Decorator to register a function in the global registry."""

    def decorator(func: ToolImplFunc) -> ToolImplFunc:
        get_registry().register(
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
    Mixin for environments that support dynamic tool loading.

    Usage:
        class MyEnvironment(DynamicToolMixin, BaseEnvironment):
            def setup(self, metadata: dict[str, Any]) -> None:
                super().setup(metadata)
                self.load_dynamic_tools(metadata)
    """

    # These will be provided by BaseEnvironment when used as a mixin
    _enabled_tools: set[str] | None = None
    state: Any = None  # ExecutionState from BaseEnvironment

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry or get_registry()
        self._dynamic_tools: dict[str, ToolDefinition] = {}
        self._dynamic_tool_schemas: dict[str, dict[str, Any]] = {}

    def load_dynamic_tools(self, metadata: dict[str, Any]) -> None:
        """Load tools from metadata."""
        self._dynamic_tools.clear()
        self._dynamic_tool_schemas.clear()

        # 1. Load from tool_implementations mapping
        if "tool_implementations" in metadata:
            for tool_name, impl_name in metadata["tool_implementations"].items():
                tool_def = self._registry.get(impl_name)
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
                    logger.warning(f"Tool implementation '{impl_name}' not found")

        # 2. Load from tool providers
        if "tool_providers" in metadata:
            for provider_name in metadata["tool_providers"]:
                provider = self._registry.get_provider(provider_name)
                if provider is None:
                    logger.warning(f"Tool provider '{provider_name}' not found")
                    continue

                if hasattr(provider, "get_tools"):
                    for tool_def in provider.get_tools():
                        self._dynamic_tools[tool_def.name] = tool_def
                        self._dynamic_tool_schemas[tool_def.name] = tool_def.to_schema()

        # 3. Load from modules (requires allowlist)
        if "tool_modules" in metadata:
            for module_path in metadata["tool_modules"]:
                loaded = self._registry.load_from_module(module_path)
                for name, tool_def in loaded.items():
                    self._dynamic_tools[name] = tool_def
                    self._dynamic_tool_schemas[name] = tool_def.to_schema()

    def get_tools(self) -> list[dict[str, Any]]:
        """Return combined static and dynamic tool schemas."""
        static_tools: list[dict[str, Any]] = super().get_tools()  # pyright: ignore[reportAttributeAccessIssue]
        dynamic_tools = list(self._dynamic_tool_schemas.values())

        all_tools = static_tools + dynamic_tools
        if self._enabled_tools is not None:
            all_tools = [t for t in all_tools if t["function"]["name"] in self._enabled_tools]

        return all_tools

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute tool - checks dynamic tools first, then static."""
        if name in self._dynamic_tools:
            if self._enabled_tools is not None:
                if name not in self._enabled_tools:
                    return ToolResult(output=f"Error: Tool '{name}' is not available.", success=False)

            try:
                tool_def = self._dynamic_tools[name]
                result = await tool_def.implementation(**arguments)
                # Track execution in state
                if hasattr(self, "state") and self.state is not None:
                    self.state.record_execution(name, result)
                return ToolResult(output=str(result), success=True)
            except Exception as e:
                return ToolResult(output=f"Error: {e}", success=False)

        # Call BaseEnvironment's execute_tool
        return await super().execute_tool(name, arguments)  # pyright: ignore[reportAttributeAccessIssue]


# ==================== Safe Math Evaluator ====================


class SafeMathEvaluator:
    """
    Safe math expression evaluator without eval().

    Supports: +, -, *, /, ** (power), unary - and +
    Only allows numbers and basic arithmetic operations.
    """

    # Binary operators
    BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    # Unary operators
    UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """Safely evaluate a math expression."""
        try:
            tree = ast.parse(expression, mode="eval")
            return cls._eval_node(tree.body)
        except (SyntaxError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid expression: {e}") from e

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.BINARY_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.BINARY_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.UNARY_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            operand = cls._eval_node(node.operand)
            return cls.UNARY_OPERATORS[op_type](operand)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# ==================== Default Tool Implementations ====================


def _register_default_tools(registry: ToolRegistry) -> None:
    """Register default tool implementations."""

    async def search_basic(query: str) -> str:
        return f"Basic search results for: {query}"

    async def search_advanced(query: str, filters: dict[str, Any] | None = None) -> str:
        filter_str = f" with filters {filters}" if filters else ""
        return f"Advanced search results for: {query}{filter_str}"

    async def calculate_safe(expression: str) -> str:
        try:
            result = SafeMathEvaluator.evaluate(expression)
            return f"Result: {result}"
        except ValueError as e:
            return f"Calculation error: {e}"

    registry.register(
        name="search_basic",
        implementation=search_basic,
        description="Basic search functionality",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
        tags=["search", "basic"],
    )

    registry.register(
        name="search_advanced",
        implementation=search_advanced,
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

    registry.register(
        name="calculate_safe",
        implementation=calculate_safe,
        description="Safe calculator (basic arithmetic only)",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}},
            "required": ["expression"],
        },
        tags=["math", "safe"],
    )
