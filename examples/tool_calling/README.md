# Tool Calling Example

Multi-hop tool calling with SGLang FunctionCallParser integration for RL training.

## Features

- **FunctionCallParser Integration**: Uses SGLang's built-in parser (qwen25, llama3, deepseekv3, etc.)
- **Token-based Processing**: Extracts token IDs from logprobs as source of truth
- **Multi-hop Support**: Automatic tool call → execute → observation loop
- **RL Compatible**: Proper loss_mask (1 for generation, 0 for observations)
- **Model Profiles**: Centralized model-specific behavior configuration
- **Schema/Binding Separation**: Serializable tool schemas + executable bindings

## Quick Start

```bash
python train.py \
    --custom-generate-function-path "examples.tool_calling.generate.generate" \
    --sglang-tool-call-parser qwen25 \
    --hf-checkpoint Qwen/Qwen2.5-7B-Instruct \
    ...
```

## Custom Tools

### Method 1: Tool Binding Registry (Recommended)

Register tools globally and reference by name in dataset metadata:

```python
from examples.tool_calling import register_tool_binding, ToolSpec

# Define your tool
def my_tool(query: str) -> str:
    return f"Result for: {query}"

# Register globally
register_tool_binding("my_tool", lambda: ToolSpec(
    name="my_tool",
    description="A custom tool that does something useful",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The query"}
        },
        "required": ["query"]
    },
    func=my_tool
))

# Reference in dataset metadata
sample.metadata["tool_names"] = ["my_tool", "calculator"]
```

### Method 2: Direct Registry (Legacy)

```python
from examples.tool_calling import ToolSpec, ToolRegistry

registry = ToolRegistry()
registry.register(ToolSpec(
    name="my_tool",
    description="A custom tool",
    parameters={...},
    func=my_tool
))
```

## Configuration

### Tool Calling Config

```python
from examples.tool_calling import ToolCallingConfig

config = ToolCallingConfig(
    max_hops=16,                    # Maximum tool calling rounds
    max_tool_calls_per_hop=4,       # Max tools per generation
    tool_parser="qwen25",           # FunctionCallParser type
    stop_on_no_tool_call=True,      # Stop when no tool call
    stop_on_error=False,            # Continue on tool errors
)
```

### Model Profiles

Model-specific behavior is configured via `MODEL_PROFILES`:

```python
from examples.tool_calling import MODEL_PROFILES, get_model_profile

# Get profile for a parser
profile = get_model_profile("kimi_k2")
# profile.call_id_format = "functions.{name}:{index}"
# profile.requires_matching_call_id = True
```

| Model | call_id_format | requires_matching_call_id |
|-------|---------------|---------------------------|
| qwen25, deepseekv3, glm, llama3 | `call_{index}` | False |
| kimi_k2 | `functions.{name}:{index}` | True |
| mistral | `call_{index}` | True |
| gpt-oss | `functions.{name}` | False |

## Architecture

### Tool Schema vs Tool Binding

```python
from examples.tool_calling import ToolSchema, ToolSpec

# ToolSchema: Serializable (for dataset metadata)
schema = ToolSchema(
    name="calculator",
    description="Evaluate expressions",
    parameters={"type": "object", "properties": {...}}
)

# ToolSpec: Executable (for runtime)
spec = ToolSpec(
    name="calculator",
    description="Evaluate expressions",
    parameters={...},
    func=calculate,  # <-- executable function
    timeout=30.0
)
```

### Dataset Metadata Format

```python
# Recommended: Use tool binding names
sample.metadata = {
    "tool_names": ["calculator", "get_weather"],  # Names from TOOL_BINDINGS
}

# Legacy: Full tool specs (deprecated)
# sample.metadata["tools"] = [...]  # Don't use - can't serialize func
```

## Key Differences from retool

| retool | tool_calling |
|--------|--------------|
| Regex parsing | FunctionCallParser (multi-model) |
| code_interpreter only | Any tool via registry |
| Manual token decode | Token-based processing |
| Hardcoded format | Model profiles + configurable |
| ToolSpec only | ToolSchema + ToolSpec separation |

## Supported Parsers

- `qwen25` / `qwen` - Qwen 2.5, Qwen 3
- `llama3` - Llama 3.2, 3.3, 4
- `deepseekv3` / `deepseekv31` / `deepseekv32` - DeepSeek V3 series
- `mistral` - Mistral
- `glm` / `glm45` / `glm47` - GLM-4 series
- `kimi_k2` - Kimi K2
- `minimax-m2` - MiniMax M2
- `gpt-oss` - GPT-OSS Harmony
- See SGLang docs for full list

## Testing

```bash
# Run tests (requires transformers)
pytest examples/tool_calling/tests/ -v

# Run specific test file
pytest examples/tool_calling/tests/test_token_invariants.py -v

# Skip slow tests (large model downloads)
pytest examples/tool_calling/tests/ -m "not slow"
```

## API Reference

### Exports

```python
from examples.tool_calling import (
    # Tool Schema & Spec
    ToolSchema,          # Serializable tool definition
    ToolSpec,            # Executable tool definition
    ToolCall,            # Parsed tool call
    ToolResult,          # Tool execution result
    ToolRegistry,        # Tool registry
    
    # Tool Bindings
    TOOL_BINDINGS,       # Global binding registry
    register_tool_binding,  # Register new tool
    get_tool_binding,    # Get tool factory
    create_registry_from_names,  # Create registry from names
    
    # Model Profiles
    ModelProfile,        # Model-specific config
    MODEL_PROFILES,      # All profiles
    get_model_profile,   # Get profile by parser name
    
    # Generation (lazy-loaded, requires sglang)
    generate,            # Main generate function
    ToolCallingConfig,   # Configuration
    parse_tool_calls,    # Parse tool calls from text
    extract_tokens_from_logprobs,  # Extract tokens from response
)
```
