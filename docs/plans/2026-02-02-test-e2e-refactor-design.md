# Test E2E Tool Calling Refactor Design

## Overview

Refactor `examples/tool_calling/tests/test_e2e_tool_calling.py` (3,413 lines, 73 tests) into smaller, role-based test files with better debugging support.

## Goals

1. **Code deduplication** - Remove repeated setup code and fixtures
2. **Legacy cleanup** - Remove tests that overlap with other validation scripts
3. **File separation** - Split by responsibility (formatters, ground truth, token flow)
4. **Parametrized tests** - Auto-expand to all registered formatters/parsers
5. **Debug support** - Clear error messages, no silent skips

## Target Structure

```
examples/tool_calling/tests/
├── conftest.py              # ~100 lines: Fixtures, PARSER_TO_HF_MODEL
├── utils.py                 # ~150 lines: MockGenerateResponse, tools
├── test_formatters.py       # ~150 lines: Formatter unit tests
├── test_ground_truth.py     # ~300 lines: HF template matching + debug
├── test_token_flow.py       # ~200 lines: Token/loss_mask invariants
├── test_integration.py      # ~150 lines: generate() mock tests
├── test_token_invariants.py # ~130 lines: Keep existing
└── deprecated/              # Archive old test_e2e_tool_calling.py
```

**Total: ~1,180 lines** (65% reduction from 3,413 lines)

## Key Design Decisions

### 1. Parser to HF Model ID Mapping

```python
PARSER_TO_HF_MODEL: dict[str, str] = {
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3_coder": "Qwen/Qwen3-0.6B",
    "glm47": "zai-org/GLM-4.7-Flash",
    "deepseekv3": "deepseek-ai/DeepSeek-V3",
    "kimi_k2": "moonshotai/Kimi-K2-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "minimax-m2": "MiniMaxAI/MiniMax-M2.1",
    "gpt-oss": "openai/gpt-oss-120b",
    # ... etc
}

# Small models for CI speed
PARSER_TO_HF_MODEL_SMALL: dict[str, str] = {
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
    "glm47": "Qwen/Qwen2.5-0.5B-Instruct",  # Fallback
    # ...
}
```

### 2. No pytest.skip() - Explicit Failures

| Situation | Handling |
|-----------|----------|
| Missing config | `pytest.fail()` + fix instructions |
| Network/download failure | `pytest.fail()` + download command |
| Known unsupported feature | `pytest.xfail()` + reason |

### 3. Debug Support for Ground Truth Tests

```python
def format_diff(expected: str, actual: str) -> str:
    """Show exact character differences with positions."""
    ...

def save_debug_info(parser_name: str, test_name: str, data: dict):
    """Save full context to JSON for debugging."""
    ...
```

### 4. Parametrized Tests Auto-Expand

```python
@pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
def parser_name(request):
    return request.param

def test_formatter_output(self, formatter_name):
    """Runs for ALL registered formatters automatically."""
```

## File Responsibilities

### conftest.py
- `PARSER_TO_HF_MODEL` mapping
- `get_tokenizer()` with caching
- `get_tokenizer_for_parser()`
- Parametrized fixtures: `formatter_name`, `parser_name`, `tokenizer`
- Tool definitions: `CALCULATOR_TOOL`, `WEATHER_TOOL`

### utils.py
- `MockGenerateResponse` class
- `create_generate_response()` factory
- `create_mock_tool_functions()`
- Debug helpers: `format_diff()`, `save_debug_info()`

### test_formatters.py
- `TestFormatterOutput`: Basic output validation
- `TestFormatterValidation`: Required kwargs (mistral, kimi_k2, gpt-oss)
- `TestFormatterEdgeCases`: Unicode, JSON, multiline, special chars

### test_ground_truth.py
- `TestToolResponseGroundTruth`: HF template == our formatter
- `TestMultiTurnGroundTruth`: Multi-hop accumulation
- `TestThinkingContentGroundTruth`: reasoning_content handling

### test_token_flow.py
- `TestTokenInvariants`: len(tokens) == len(loss_mask) == len(log_probs)
- `TestLossMaskValues`: Binary values only
- `TestMultiHopAlignment`: Alignment after each hop
- `TestTokenRoundtrip`: encode/decode preserves content

### test_integration.py
- `TestGenerateFunctionFlow`: Mock SGLang, test generate() behavior
- Status handling: COMPLETED, TRUNCATED, FAILED

## Migration Plan

1. Create new test files with shared infrastructure
2. Migrate tests incrementally (formatters → ground truth → token flow)
3. Run both old and new tests to verify parity
4. Move `test_e2e_tool_calling.py` to `deprecated/`
5. Remove deprecated folder after validation period

## Success Criteria

- [ ] All new tests pass
- [ ] Line count reduced by >50%
- [ ] No `pytest.skip()` usage
- [ ] Debug info saved on failures
- [ ] Parametrized across all formatters/parsers
