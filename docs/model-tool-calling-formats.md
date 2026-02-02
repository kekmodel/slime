# Model-Specific Tool Calling & Thinking Token Formats

This document provides a comprehensive reference for how different LLMs handle tool calling and reasoning/thinking tokens. Understanding these differences is critical for:
- **RL Training**: Ensuring `len(token_ids) == len(loss_mask) == len(log_probs)`
- **Multi-hop Tool Calling**: Correctly injecting tool results back into context
- **Reasoning Token Preservation**: Managing `reasoning_content` in conversation history

## Table of Contents
1. [Quick Reference Table](#quick-reference-table)
2. [Model-Specific Details](#model-specific-details)
3. [SGLang Parser Mappings](#sglang-parser-mappings)
4. [Tool Response Formatters](#tool-response-formatters)
5. [RL Training Considerations](#rl-training-considerations)

---

## Quick Reference Table

| Model | Thinking Format | Tool Call Format | Tool Response Format | Tool Call ID | Multi-Tool Wrap | History Reasoning |
|-------|----------------|------------------|---------------------|--------------|-----------------|-------------------|
| Qwen 2.5 | ❌ None | `<tool_call>JSON</tool_call>` | `<tool_response>` in user | ❌ No | ❌ No | N/A |
| Qwen 3 | `<think>...</think>` | Same as Qwen 2.5 | Same as Qwen 2.5 | ❌ No | ❌ No | Tool: ALL, Chat: LAST |
| Qwen3-Coder | `<think>...</think>` | Same as Qwen 2.5 | Same as Qwen 2.5 | ❌ No | ❌ No | Tool: ALL, Chat: LAST |
| GLM-4/4.5 | ❌ None | XML format | `<\|observation\|>` role | ❌ No | ❌ No | N/A |
| GLM-4.7 | `<think>...</think>` | XML format | `<\|observation\|><tool_response>` | ❌ No | ✅ Yes | Tool: ALL, Chat: LAST |
| DeepSeek-R1 | `<think>...</think>` | N/A (no tools) | N/A | N/A | N/A | Stripped by SGLang |
| DeepSeek-V3 | `<think>...</think>` | Special tokens | Special tokens | ❌ No | ❌ No | Stripped by SGLang |
| DeepSeek-V3.2 | `<think>...</think>` | DSML format | `<function_results>` | ❌ No | ❌ No | Stripped by SGLang |
| Kimi-K2 | ❌ None | `<\|tool_call_begin\|>` | `## Return of {id}` | ✅ Yes | ✅ Yes | N/A |
| Kimi-K2-Thinking | `<think>...</think>` | Same as Kimi-K2 | Same as Kimi-K2 | ✅ Yes | ✅ Yes | Tool: preserved, Chat: empty |
| Llama 3.x | ❌ None | `<\|python_tag\|>` or JSON | `ipython` role | ❌ No | ❌ No | N/A |
| Mistral | ❌ None | `[TOOL_CALLS]` | `[TOOL_RESULTS]` | ✅ Yes | ✅ Yes | N/A |
| MiniMax-M2 | `<think>...</think>` | Custom format | `]~b]tool<response>` | ❌ No | ❌ No | Unknown |

---

## Model-Specific Details

### Qwen 2.5 / Qwen 3 / Qwen3-Coder

**Thinking Output (Qwen 3+ only):**
```
<think>
Let me analyze this step by step...
</think>
```

**Tool Call Output:**
```
<tool_call>
{"name": "calculator", "arguments": {"expression": "2 + 3"}}
</tool_call>
```

**Tool Response Input:**
```
<|im_start|>user
<tool_response>
5
</tool_response><|im_end|>
<|im_start|>assistant
```

**History Reasoning Behavior:**
- Tool call context: ALL `reasoning_content` preserved in history
- Regular chat: Only LAST message's `reasoning_content` preserved
- This is controlled by the Jinja template's conditional logic

**Key Insight:** The template strips `<think>` tags when rendering history but preserves `reasoning_content` field when tool_calls are present.

---

### GLM-4 / GLM-4.5

**Thinking Output:** Not supported

**Tool Call Output (XML format):**
```
<|assistant|>
calculator
```python
{"expression": "2 + 3"}
```
```

**Tool Response Input:**
```
<|observation|>
5<|assistant|>
```

**Special Tokens:**
- `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`
- Uses metadata field for function names

---

### GLM-4.7

**Thinking Output:**
```
<think>
Analyzing the request...
</think>
```

**Tool Call Output:**
```xml
<|assistant|><think>
Let me calculate...
</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+3"}}
</tool_call>
```

**Tool Response Input:**
```
<|observation|><tool_response>5</tool_response><|assistant|><think>
```

**Multi-Tool Parallel Calls:**
```
<|observation|><tool_response>result1</tool_response><tool_response>result2</tool_response><|assistant|><think>
```

**Interleaved Thinking:** Supported via `clear_thinking: False` parameter

**History Reasoning Behavior:** Same as Qwen 3 - preserves all when tool_calls present, only last otherwise

---

### DeepSeek Family

#### DeepSeek-R1 (Reasoning only, no tools)
**Thinking Output:**
```
<think>
Deep reasoning process...
</think>
```

**History:** SGLang's custom encoding explicitly strips `reasoning_content` from history messages.

#### DeepSeek-V3 / V3.1
**Thinking Output:** `<think>...</think>`

**Tool Call Output (Special tokens):**
```
<｜tool▁call▁begin｜>calculator
<｜tool▁call▁argument▁begin｜>{"expression": "2+3"}<｜tool▁call▁end｜>
```

**Tool Response Input:**
```
<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>5<｜tool▁output▁end｜><｜tool▁outputs▁end｜>
```

**Note:** Uses full-width characters (｜) not regular pipe (|)

#### DeepSeek-V3.2
**Thinking Output:** `<think>...</think>`

**Tool Call Output (DSML format):**
```xml
<function_calls>
<invoke name="calculator">
<parameter name="expression">2+3</parameter>
</invoke>
</function_calls>
```

**Tool Response Input:**
```xml

<function_results>
<result>5</result>
</function_results>
```

**History Management:** `drop_thinking_messages()` in `encoding_dsv32.py` explicitly removes `reasoning_content` from history before last user message.

---

### Kimi-K2 Family

**Tool Call ID Format:** `functions.{function_name}:{index}` (e.g., `functions.calculator:0`)

**Provider Note:** Some providers may return IDs without the `functions.` prefix (e.g., just `calculator:0`).
Verified providers returning correct format:
- Parasail (kimi-k2.5)
- BaseTen (kimi-k2-thinking)

**Tool Call Output:**
```
<|tool_calls_section_begin|><|tool_call_begin|>functions.calculator:0<|tool_call_argument_begin|>{"expression": "2+3"}<|tool_call_end|><|tool_calls_section_end|>
```

**Tool Response Input:**
```
<|im_system|>tool<|im_middle|>## Return of functions.calculator:0
5<|im_end|><|im_assistant|>assistant<|im_middle|>
```

**Key Differences:**
- Uses `tool` as role name (not function name)
- Tool call ID is required and uses `functions.{name}:{index}` format
- Multi-tool calls wrapped in `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`

#### Kimi-K2-Thinking
**Thinking Output:** `<think>...</think>`

**History Reasoning Behavior:**
- Tool call messages: `reasoning_content` preserved
- Regular chat: Empty `<think></think>` tags rendered (no content preserved)

---

### Llama 3.x

**Tool Call Output:**
```
<|python_tag|>{"name": "calculator", "parameters": {"expression": "2+3"}}
```

Or with `<function>` tag depending on configuration.

**Tool Response Input:**
```
<|start_header_id|>ipython<|end_header_id|>

{"output": "5"}<|eot_id|>
```

**Note:** Uses `ipython` role for tool responses. Content is JSON-encoded.

---

### Mistral

**Tool Call Output:**
```
[TOOL_CALLS] [{"name": "calculator", "arguments": {"expression": "2+3"}, "id": "call_123"}]
```

**Tool Response Input:**
```
[TOOL_RESULTS] {"content": "5", "call_id": "call_123"}[/TOOL_RESULTS]
```

**Key:** Requires `call_id` matching the tool call.

---

### MiniMax-M2

**Thinking Output:** `<think>...</think>`

**Tool Response Input:**
```
]~b]tool
<response>5</response>[e~[
```

---

## SGLang Parser Mappings

### Reasoning Parser (`reasoning_parser.py`)

```python
DetectorMap = {
    "deepseek-r1": DeepSeekR1Detector,    # <think>...</think>, force_reasoning=True
    "deepseek-v3": Qwen3Detector,         # <think>...</think>, force_reasoning=False
    "glm45": Qwen3Detector,
    "gpt-oss": GptOssDetector,            # <|channel|>analysis<|message|>...<|end|>
    "kimi": KimiDetector,                 # ◁think▷...◁/think▷
    "kimi_k2": Qwen3Detector,             # <think>...</think>
    "qwen3": Qwen3Detector,
    "qwen3-thinking": Qwen3Detector,      # force_reasoning=True
    "minimax": Qwen3Detector,
}
```

### Function Call Parser (`function_call_parser.py`)

```python
ToolCallParserEnum = {
    "deepseekv3": DeepSeekV3Detector,
    "deepseekv31": DeepSeekV31Detector,
    "deepseekv32": DeepSeekV32Detector,   # DSML format
    "glm": Glm4MoeDetector,
    "glm45": Glm4MoeDetector,
    "glm47": Glm47MoeDetector,
    "gpt-oss": GptOssDetector,
    "kimi_k2": KimiK2Detector,
    "llama3": Llama32Detector,
    "mistral": MistralDetector,
    "qwen": Qwen25Detector,
    "qwen25": Qwen25Detector,
    "qwen3_coder": Qwen3CoderDetector,
    "hermes": HermesDetector,
    "mimo": Qwen25Detector,
    "minimax-m2": MinimaxM2Detector,
}
```

---

## Tool Response Formatters

Located in `examples/tool_calling/tools.py`:

```python
TOOL_RESPONSE_FORMATTERS = {
    # Qwen family - same format
    "qwen": format_qwen,
    "qwen25": format_qwen,
    "qwen3_coder": format_qwen,
    "mimo": format_qwen,
    "hermes": format_qwen,

    # GLM family
    "glm": format_glm,        # <|observation|>\n{content}<|assistant|>
    "glm45": format_glm,
    "glm47": format_glm47,    # <|observation|><tool_response>{content}</tool_response><|assistant|><think>

    # DeepSeek family
    "deepseekv3": format_deepseek_v3,   # Special token wrappers
    "deepseekv31": format_deepseek_v3,
    "deepseekv32": format_deepseek_v32, # <function_results> XML

    # Others
    "llama3": format_llama3,      # ipython role with JSON
    "mistral": format_mistral,    # [TOOL_RESULTS] with call_id
    "kimi_k2": format_kimi_k2,    # ## Return of {id} format
    "minimax-m2": format_minimax, # ]~b]tool<response>
    "gpt-oss": format_gpt_oss,    # functions.{name} prefix
}
```

---

## RL Training Considerations

### The Core Invariant
```
len(token_ids) == len(loss_mask) == len(log_probs)
```

### loss_mask Semantics
- `1`: Model-generated tokens (train on these)
  - Includes thinking/reasoning tokens
  - Includes tool call tokens
  - Includes final answer tokens
- `0`: Observation/input tokens (don't train)
  - Tool response content
  - System prompts
  - User messages

### Multi-Hop Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Turn 1: Initial Query                                            │
├─────────────────────────────────────────────────────────────────┤
│ INPUT:  System + User message                          loss=0   │
│ OUTPUT: <think>reasoning</think> + tool_call           loss=1   │
├─────────────────────────────────────────────────────────────────┤
│ Turn 2: Tool Result                                              │
├─────────────────────────────────────────────────────────────────┤
│ INPUT:  Tool response (formatter output)               loss=0   │
│ OUTPUT: <think>more reasoning</think> + answer         loss=1   │
└─────────────────────────────────────────────────────────────────┘
```

### Model-Specific Considerations

1. **DeepSeek Models**: SGLang strips `reasoning_content` from history. For RL, you may need to preserve it separately.

2. **Qwen3/GLM-4.7**: Preserve all reasoning in tool-call context but only last in regular chat. Ensure your training data matches this behavior.

3. **Kimi-K2**: Requires tool_call_id in specific format. Training data must include proper IDs.

4. **Interleaved Thinking (GLM-4.7)**: When `clear_thinking: False`, thinking can appear between tool calls. Training data should reflect this.

### Verification Checklist

- [ ] Token counts match across all arrays
- [ ] Tool response format matches model's expected input
- [ ] Reasoning tokens have loss_mask=1
- [ ] Observation tokens have loss_mask=0
- [ ] Tool call IDs match between call and response (if required)
- [ ] Special tokens are properly tokenized (not split)

---

## References

- SGLang: `python/sglang/srt/parser/reasoning_parser.py`
- SGLang: `python/sglang/srt/function_call/function_call_parser.py`
- SGLang: `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- HuggingFace Model Repos: Chat templates in `tokenizer_config.json`
- slime: `examples/tool_calling/tools.py`
