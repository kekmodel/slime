# slime_gym

SLIME을 위한 Gym 스타일 환경 추상화. RL 학습용 에이전트 도구 호출 환경.

## 핵심 설계

### 1. State 기반 검증

도구 실행 시 state flag 설정 → `verify()`에서 state 확인:

```python
@tool(description="Process refund")
async def process_refund(self, order_id: str, reason: str) -> str:
    self.state.refund_processed = True  # state 설정
    return f"Refund processed"

async def verify(self, sample: Sample) -> float:
    if not self.state.refund_processed:  # state 확인
        return 0.0
    return 1.0
```

### 2. submit_result 패턴

자연어 답변은 검증 어려움 → 구조화된 dict 제출 후 exact match:

```python
# 태스크 정의
{
    "expected_actions": ["get_info", "process", "submit_result"],
    "expected_result": {"status": "done", "amount": 100}
}

# 에이전트가 submit_result 호출
<tool_call>{"name": "submit_result", "arguments": {"result": {"status": "done", "amount": 100}}}</tool_call>

# verify()에서 비교
if self.state.submitted_result != self.expected_result:
    return 0.0
```

### 3. Dynamic max_turns

`expected_actions` 개수 기반 자동 설정:

```python
# 우선순위
1. metadata["max_turns"]           # 샘플별 고정
2. len(expected_actions) + buffer  # 동적 (기본)
3. CONFIGS["max_turns"]            # 전역 폴백
```

## 빠른 시작

### 테스트
```bash
python examples/slime_gym/test_gym_standalone.py
```

### 학습
```bash
./examples/slime_gym/run_retail.sh

# 옵션
NUM_GPUS=4 ./run_retail.sh
MAX_TURNS=15 DYNAMIC_MAX_TURNS=false ./run_retail.sh
```

## 환경 만들기

```python
from slime.utils.types import Sample
from examples.slime_gym import BaseEnvironment, tool

class MyEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.state = None
        self.expected_actions = set()
        self.expected_result = None

    def seed(self, metadata: dict) -> None:
        super().seed(metadata)
        self.state = MyState()
        self.expected_actions = set(metadata.get("expected_actions", []))
        self.expected_result = metadata.get("expected_result")

    @tool(
        description="Do something",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
    )
    async def my_tool(self, input: str) -> str:
        self.state.tool_executed = True
        return f"Done: {input}"

    @tool(description="Submit final result")
    async def submit_result(self, result: dict) -> str:
        self.state.submitted_result = result
        return "Submitted"

    async def verify(self, sample: Sample) -> float:
        # 1. 도구 실행 확인
        if "my_tool" in self.expected_actions and not self.state.tool_executed:
            return 0.0
        # 2. 결과 확인
        if self.expected_result and self.state.submitted_result != self.expected_result:
            return 0.0
        return 1.0
```

등록:
```python
# generate_with_gym.py
ENVIRONMENTS = {
    "retail_service": RetailServiceEnvironment,
    "my_env": MyEnvironment,  # 추가
}
```

## 태스크 정의

```python
{
    "prompt": [{"role": "user", "content": "Process my refund for order ORD-123"}],
    "metadata": {
        "env_name": "retail_service",
        "task_type": "refund",  # 도구 세트 선택
        "customer": {"id": "CUST-001", "name": "Alice", "tier": "gold"},
        "order": {"id": "ORD-123", "price": 89.99, "status": "delivered"},
        "expected_actions": [
            "get_customer_info",
            "get_order_details",
            "process_refund",
            "submit_result"
        ],
        "expected_result": {
            "customer_name": "Alice",
            "order_id": "ORD-123",
            "refund_amount": 89.99,
            "status": "refunded"
        }
    }
}
```

## 설정

환경 변수:
```bash
SLIME_GYM_MAX_TURNS=10           # 기본 max_turns
SLIME_GYM_MAX_TURNS_BUFFER=0     # 동적 모드 버퍼
SLIME_GYM_DYNAMIC_MAX_TURNS=true # true=동적, false=고정
```

## 파일 구조

| 파일 | 설명 |
|------|------|
| `base.py` | `BaseEnvironment`, `@tool`, `run_episode()` |
| `retail_env.py` | 예제: 소매 고객 서비스 환경 |
| `generate_with_gym.py` | SLIME 통합 (`generate`, `reward_func`) |
| `tasks.py` | 예제 태스크 |
| `run_retail.sh` | 학습 스크립트 |
