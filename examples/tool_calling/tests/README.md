# Tool Calling Tests

Tool calling 기능의 pytest 테스트 및 검증.

---

## 폴더 구조

```
examples/tool_calling/
├── tests/                          # pytest 자동 테스트
│   ├── test_e2e_tool_calling.py    # Mock 기반 E2E 테스트
│   ├── test_token_invariants.py    # 토큰 정렬 불변성 테스트
│   ├── conftest.py                 # pytest 설정
│   ├── utils.py                    # 공통 유틸리티
│   └── outputs/                    # 테스트 결과 (git 제외)
│
├── benchmarks/                     # 성능 측정
│   ├── README.md                   # 벤치마크 결과 및 사용법
│   └── reasoning_length.py         # Reasoning 길이 벤치마크
│
└── scripts/                        # 수동 실행 스크립트
    ├── cross_validation.py         # chat/completions vs completions 비교
    ├── multiturn_template.py       # HF apply_chat_template 검증
    └── openrouter_e2e.py           # OpenRouter Live E2E
```

---

## 테스트 실행

```bash
# pytest 자동 테스트
uv run pytest examples/tool_calling/tests/ -v

# 특정 테스트 파일
uv run pytest examples/tool_calling/tests/test_token_invariants.py -v

# 특정 테스트만
uv run pytest examples/tool_calling/tests/ -k "test_qwen" -v
```

### 수동 스크립트

```bash
# Live E2E 테스트 (OPENROUTER_API_KEY 필요)
python examples/tool_calling/scripts/openrouter_e2e.py

# 멀티턴 템플릿 검증
python examples/tool_calling/scripts/multiturn_template.py              # 전체
python examples/tool_calling/scripts/multiturn_template.py qwen3-30b    # 필터

# Cross-validation 테스트
python examples/tool_calling/scripts/cross_validation.py
```

### 벤치마크

벤치마크 실행 및 결과는 [benchmarks/README.md](../benchmarks/README.md) 참조.

---

## Formatter 사용법

모든 formatter는 `add_generation_prompt` 파라미터를 지원합니다:

```python
from examples.tool_calling.tools import format_qwen, format_glm47, format_kimi_k2

# 기본값 (add_generation_prompt=False) - tool response만
format_qwen('{"result": 42}')
# → <|im_start|>user\n<tool_response>\n{"result": 42}\n</tool_response><|im_end|>\n

# add_generation_prompt=True - HF 템플릿과 동일 (generation prompt 포함)
format_qwen('{"result": 42}', add_generation_prompt=True)
# → ...<|im_end|>\n<|im_start|>assistant\n
```

모델별 상세 형식은 [MODEL_TOOL_CALLING.md](../MODEL_TOOL_CALLING.md) 참조.

---

## 검증 완료 항목

### Formatter 정확성
- [x] Qwen: `apply_chat_template` 출력과 일치
- [x] GLM-4.7: `apply_chat_template` 출력과 일치
- [x] Kimi-K2: `apply_chat_template` 출력과 일치
- [x] DeepSeek-V3: `apply_chat_template` 출력과 일치

### generate.py 동작
- [x] Parallel 지원 모델: 한 hop에서 여러 tool call 처리
- [x] Sequential 전용 모델: Multi-hop으로 순차 처리
- [x] call_id 형식: 모델별 올바른 형식 생성
- [x] 순서 보장: `asyncio.gather`로 입력 순서 유지
- [x] Loss mask: 생성 토큰=1, observation=0

### Known Issues
- [ ] GPT-OSS: 단일 tool call 간헐적 실패
- [ ] GLM-4.7, DeepSeek-V3.2: Unicode 처리 불완전
