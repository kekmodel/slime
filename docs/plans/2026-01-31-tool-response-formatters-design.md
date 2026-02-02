# Tool Response Formatters 설계

## 개요

`OBSERVATION_FORMATS`를 `generate.py`에서 분리하여 `tools.py`에 통합.
- 이름 변경: `OBSERVATION_FORMATS` → `TOOL_RESPONSE_FORMATTERS`
- 구현 방식: 함수 기반 formatter
- SGLang parser name과 정확히 일치하도록 키 네이밍

## 지원 Parser 목록

| Parser Name | Formatter | 비고 |
|-------------|-----------|------|
| qwen | format_qwen | |
| qwen25 | format_qwen | |
| qwen3_coder | format_qwen | |
| glm | format_glm | |
| glm45 | format_glm | |
| glm47 | format_glm | |
| deepseekv3 | format_deepseek_v3 | |
| deepseekv31 | format_deepseek_v31 | |
| deepseekv32 | format_deepseek_v31 | 같은 포맷 |
| gpt-oss | format_gpt_oss | tool_name 필수 |
| kimi_k2 | format_kimi_k2 | tool_call_id 필수 |
| minimax-m2 | format_minimax | |
| mimo | format_qwen | Qwen과 동일 |
| llama3 | format_llama3 | |
| mistral | format_mistral | |
| hermes | format_qwen | Qwen과 동일 |

## 구현

### Formatter 함수들

```python
def format_qwen(content: str, **ctx) -> str:
    return f"\n<tool_response>\n{content}\n</tool_response>\n"

def format_glm(content: str, **ctx) -> str:
    return f"\n<|observation|>\n{content}\n"

def format_deepseek_v3(content: str, **ctx) -> str:
    return f"\n<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{content}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>\n"

def format_deepseek_v31(content: str, **ctx) -> str:
    return f"\n<｜tool▁output▁begin｜>{content}<｜tool▁output▁end｜>\n"

def format_llama3(content: str, **ctx) -> str:
    return f'\n<|start_header_id|>ipython<|end_header_id|>\n\n{{"output": "{content}"}}<|eot_id|>\n'

def format_mistral(content: str, **ctx) -> str:
    return f"\n[TOOL_RESULTS]{content}[/TOOL_RESULTS]\n"

def format_gpt_oss(content: str, tool_name: str = "", **ctx) -> str:
    if not tool_name:
        raise ValueError("format_gpt_oss requires 'tool_name'")
    return f"<|start|>{tool_name} to=assistant<|channel|>commentary<|message|>{content}<|end|>"

def format_kimi_k2(content: str, tool_call_id: str = "", **ctx) -> str:
    if not tool_call_id:
        raise ValueError("format_kimi_k2 requires 'tool_call_id'")
    return f"<|im_system|>tool<|im_middle|>\n## Return of {tool_call_id}\n{content}\n<|im_end|>"

def format_minimax(content: str, **ctx) -> str:
    return f"]~b]tool\n<response>\n{content}\n</response>\n[e~["
```

### Registry

```python
TOOL_RESPONSE_FORMATTERS = {
    "qwen": format_qwen,
    "qwen25": format_qwen,
    "qwen3_coder": format_qwen,
    "glm": format_glm,
    "glm45": format_glm,
    "glm47": format_glm,
    "deepseekv3": format_deepseek_v3,
    "deepseekv31": format_deepseek_v31,
    "deepseekv32": format_deepseek_v31,
    "gpt-oss": format_gpt_oss,
    "kimi_k2": format_kimi_k2,
    "minimax-m2": format_minimax,
    "mimo": format_qwen,
    "llama3": format_llama3,
    "mistral": format_mistral,
    "hermes": format_qwen,
}
```

### 사용법

```python
formatter = TOOL_RESPONSE_FORMATTERS["qwen25"]
formatted = formatter(content="result", tool_name="calc", tool_call_id="call_0")
```

## 변경 사항

1. `tools.py`: formatter 함수들과 `TOOL_RESPONSE_FORMATTERS` 추가
2. `generate.py`: `OBSERVATION_FORMATS` 제거, `tools.py`에서 import
3. `__init__.py`: 필요시 export 추가

## 설계 원칙

- 필수 파라미터 없으면 명시적 `ValueError` (조용한 fallback 금지)
- 없는 parser name이면 `KeyError` (자동 fallback 금지)
- 같은 포맷 쓰는 parser들은 같은 함수 참조
