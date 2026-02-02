# Model Tool Calling Reference

각 모델의 tool calling 특성을 정리한 문서입니다.

> **Note**: Tool Response Format은 `add_generation_prompt=True` 기준입니다.
> Formatter 기본값은 `add_generation_prompt=False`로, generation prompt를 제외합니다.

## 요약 테이블

| Model | Parser | Parallel Support | call_id Format | Response Format |
|-------|--------|------------------|----------------|-----------------|
| Qwen 2.5/3/Coder | `qwen25` | ✓ | `call_{i}` | `<tool_response>` in user role |
| GLM-4.7 | `glm47` | ✓ | `call_{i}` | `<\|observation\|><tool_response>` |
| DeepSeek V3/V3.1 | `deepseekv3` | ✓ | `call_{i}` | Special tokens `<｜tool▁output▁begin｜>` |
| DeepSeek V3.2 | `deepseekv32` | ✓ | `call_{i}` | DSML `<function_results>` |
| Kimi-K2 | `kimi_k2` | ✗ | `functions.{name}:{i}` | `## Return of {id}` in tool role |
| GPT-OSS | `gpt-oss` | ✗ | `functions.{name}` | Harmony format |
| MiniMax M2 | `minimax-m2` | ✓ | `call_{i}` | `<response>` in tool role |
| Llama 3.x | `llama3` | ✓ | `call_{i}` | ipython role with JSON |
| Mistral | `mistral` | ✓ | UUID (model-generated) | `[TOOL_RESULTS]` |
| MIMO | `mimo` | ✓ | `call_{i}` | Qwen 형식과 동일 |
| Hermes | `hermes` | ✓ | `call_{i}` | Qwen 형식과 동일 |
| GLM-4 / GLM-4.5 | `glm` | ✓ | `call_{i}` | `<\|observation\|>` (wrapper 없음) |

---

## Qwen 2.5 / Qwen 3 / Qwen3-Coder / MIMO / Hermes

**Parser**: `qwen25` (또는 `mimo`, `hermes` - 동일 형식)

### Tool Call Format
```
<|im_start|>assistant
<tool_call>
{"name": "calculator", "arguments": {"expression": "10+5"}}
</tool_call><|im_end|>
```

### Tool Response Format
```
<|im_start|>user
<tool_response>
15
</tool_response><|im_end|>
<|im_start|>assistant
```

### 특징
- **Parallel 지원**: ✓ 여러 `<tool_call>` 블록을 한 번에 생성
- Tool response는 `user` role로 주입
- `<think>` 토큰은 모델이 자동 생성 (주입 불필요)

---

## GLM-4.7

**Parser**: `glm47`

### Tool Call Format
```
<|assistant|><think>...</think><tool_call>calculator<arg_key>expression</arg_key><arg_value>10+5</arg_value></tool_call>
```

### Tool Response Format
```
<|observation|><tool_response>15</tool_response><|assistant|>
```

### Parallel Tool Response (여러 결과)
```
<|observation|><tool_response>15</tool_response><tool_response>Sunny, 20C</tool_response><|assistant|>
```

### 특징
- **Parallel 지원**: ✓
- `<|observation|>` 토큰으로 tool response 시작
- `<think>` 토큰은 모델이 자동 생성 (formatter에서 제외)
- GLM-4/4.5와 다른 형식 (4.7은 `<tool_response>` wrapper 사용)

---

## GLM-4 / GLM-4.5

**Parser**: `glm` (또는 `glm45`)

### Tool Response Format
```
<|observation|>
15<|assistant|>
```

### 특징
- **Parallel 지원**: ✓
- GLM-4.7과 달리 `<tool_response>` wrapper 없음
- 더 단순한 형식

---

## DeepSeek V3 / V3.1

**Parser**: `deepseekv3`

### Tool Call Format
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>calculator
```json
{"expression": "10+5"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

### Tool Response Format
```
<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>15<｜tool▁output▁end｜><｜tool▁outputs▁end｜>
```

### 특징
- **Parallel 지원**: ✓
- 특수 유니코드 토큰 사용 (전각 문자)
- V3.1도 동일 형식

---

## DeepSeek V3.2

**Parser**: `deepseekv32`

### Tool Call Format
```
<|Assistant|><functioncall>
calculator({"expression": "10+5"})
</functioncall>
```

### Tool Response Format
```

<function_results>
<result>15</result>
</function_results>
```

### Parallel Tool Response
```

<function_results>
<result>15</result>
<result>Sunny, 20C</result>
</function_results>
```

### 특징
- **Parallel 지원**: ✓
- DSML (DeepSeek Markup Language) 기반
- V3/V3.1과 완전히 다른 형식
- 별도 인코딩 스크립트 제공: `encoding_dsv32.py`

---

## Kimi-K2 (Instruct / Thinking)

**Parser**: `kimi_k2`

### Tool Call Format
```
<|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|><|tool_call_begin|>functions.calculator:0<|tool_call_argument_begin|>{"expression": "10+5"}<|tool_call_end|><|tool_calls_section_end|><|im_end|>
```

### Tool Response Format
```
<|im_system|>tool<|im_middle|>## Return of functions.calculator:0
15<|im_end|><|im_assistant|>assistant<|im_middle|>
```

### call_id 형식
- **필수**: `functions.{name}:{index}` (예: `functions.calculator:0`)
- 모델이 생성한 ID와 정확히 매칭해야 함

### 특징
- **Parallel 지원**: ✗ (한 번에 1개의 tool call만 생성)
- Sequential fallback: 다음 turn에서 추가 tool call 생성
- `tool` role 사용 (다른 모델과 다름)
- Thinking 모델은 reasoning 토큰 포함

---

## GPT-OSS (Harmony Format)

**Parser**: `gpt-oss`

### Tool Call Format
```
<|start|>assistant<|channel|>commentary to=functions.calculator<|constrain|>json<|message|>{"expression": "10+5"}<|call|>
```

### Tool Response Format
```
<|start|>functions.calculator to=assistant<|channel|>commentary<|message|>"15"<|end|>
```

### 특징
- **Parallel 지원**: ✗ (한 번에 1개의 tool call만 생성)
- Harmony 프로토콜 기반
- Content는 JSON으로 인코딩됨
- `<|call|>`로 tool call 종료, `<|end|>`로 response 종료

---

## MiniMax M2

**Parser**: `minimax-m2`

### Tool Call Format
```
[b~[assistant
<tool_call>{"name": "calculator", "arguments": {"expression": "10+5"}}</tool_call>[e~[
```

### Tool Response Format
```
]~b]tool
<response>15</response>[e~[
```

### 특징
- **Parallel 지원**: ✓
- 독특한 구분자 사용: `[b~[`, `[e~[`, `]~b]`
- tool role 사용

---

## Llama 3.x

**Parser**: `llama3`

### Tool Call Format
```
<|start_header_id|>assistant<|end_header_id|>

<|python_tag|>calculator.call(expression="10+5")<|eom_id|>
```

### Tool Response Format
```
<|start_header_id|>ipython<|end_header_id|>

{"output": "15"}<|eot_id|>
```

### 특징
- **Parallel 지원**: ✓
- `ipython` role 사용
- Content는 JSON 객체로 래핑

---

## Mistral

**Parser**: `mistral`

### Tool Call Format
```
[TOOL_CALLS] [{"name": "calculator", "arguments": {"expression": "10+5"}, "id": "abc123"}]
```

### Tool Response Format
```
[TOOL_RESULTS] {"content": "15", "call_id": "abc123"}[/TOOL_RESULTS]
```

### 특징
- **Parallel 지원**: ✓
- 모델이 UUID 형식의 call_id 생성
- call_id 매칭 필수

---

## Parallel vs Sequential 처리

### generate.py 동작

```python
# 1. Tool call 파싱 (개수 상관없이 모두 추출)
_, tool_calls = parse_tool_calls(response_text, tools, parser_name)

# 2. 병렬 실행 (asyncio.gather는 순서 보장)
results = await registry.execute_batch(tool_calls)

# 3. Observation 순차 추가 (loss_mask=0)
for result in results:
    observation = format_observation(result, parser_name)
```

### Parallel 지원 모델
- 한 hop에서 여러 tool call → 모두 실행 → 모든 observation 추가 → 다음 hop

### Sequential 전용 모델 (Kimi-K2, GPT-OSS)
- Hop 1: 1개 tool call → 실행 → observation
- Hop 2: 1개 tool call → 실행 → observation
- Hop N: Final answer

---

## Loss Mask 규칙

| Token Type | loss_mask | 설명 |
|------------|-----------|------|
| 모델 생성 (tool call 포함) | 1 | RL 학습 대상 |
| Observation (tool response) | 0 | 학습 제외 |
| `<think>` (모델 생성) | 1 | 학습 대상 (주입하면 안 됨) |

---

## 테스트 검증 결과

### Formatter 검증 (apply_chat_template 대비)
- ✓ Qwen3-30B-A3B-Thinking
- ✓ GLM-4.7-Flash
- ✓ Kimi-K2-Instruct
- ✓ DeepSeek-V3

### Parallel Tool Call 테스트
- ✓ Qwen: 2개 이상 동시 호출
- ✓ GLM-4.7: 2개 이상 동시 호출
- ✓ DeepSeek: 2개 이상 동시 호출
- ✓ MiniMax: 2개 이상 동시 호출
- ✗ Kimi-K2: 항상 1개만 호출 (모델 제한)
- ✗ GPT-OSS: 항상 1개만 호출 (모델 제한)

### Multi-hop 테스트
- 모든 모델에서 sequential fallback 정상 동작
