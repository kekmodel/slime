# slime_gym

SLIME을 위한 Gym 스타일 환경 추상화. RL 학습용 에이전트 도구 호출 환경.

## 빠른 시작

### 테스트
```bash
python examples/slime_gym/test_gym_standalone.py
```

### 학습
```bash
./examples/slime_gym/run_retail.sh
```

## SLIME 통합

```bash
python -m slime.train \
    --custom-generate-function-path examples.slime_gym.generate_with_gym.generate \
    --custom-rm-path examples.slime_gym.generate_with_gym.reward_func \
    ...
```

## 환경 만들기

```python
from examples.slime_gym import BaseEnvironment, EnvironmentRegistry, tool

@EnvironmentRegistry.register("my_env")
class MyEnvironment(BaseEnvironment):
    def setup(self, metadata: dict) -> None:
        super().setup(metadata)
        self.expected_result = metadata.get("expected_result")

    @tool(description="Do something")
    async def my_tool(self, input: str) -> str:
        return f"Done: {input}"

    @tool(description="Submit final result")
    async def submit_result(self, result: dict) -> str:
        self.state.submitted_result = result
        return "Submitted"

    def verify(self) -> float:
        if not self.state.has_executed_all(self.expected_actions):
            return 0.0
        if self.expected_result and self.state.submitted_result != self.expected_result:
            return 0.0
        return 1.0
```

## 태스크 정의

```python
{
    "prompt": [{"role": "user", "content": "Process my refund for order ORD-123"}],
    "metadata": {
        "env_name": "retail_service",
        "expected_actions": ["get_customer_info", "process_refund", "submit_result"],
        "expected_result": {"status": "refunded", "amount": 89.99}
    }
}
```

## 설정

환경 변수:
```bash
SLIME_GYM_MAX_TURNS=10           # 기본 max_turns
SLIME_GYM_MAX_TURNS_BUFFER=0     # 동적 모드 버퍼
SLIME_GYM_DYNAMIC_MAX_TURNS=true # 동적 max_turns 사용
```

## 파일 구조

| 파일 | 설명 |
|------|------|
| `base.py` | `BaseEnvironment`, `@tool` 데코레이터 |
| `types.py` | `ExecutionState`, `ToolCall`, `ToolResult` |
| `config.py` | `MAX_TURNS`, `resolve_max_turns()` |
| `env_registry.py` | `@EnvironmentRegistry.register()` |
| `formatters.py` | `ChatMLFormatter` |
| `retail_env.py` | 예제: 소매 고객 서비스 환경 |
| `dynamic_env.py` | 예제: 동적 도구 로딩 환경 |
| `generate_with_gym.py` | SLIME 통합 (`generate`, `reward_func`) |
| `tool_registry.py` | 고급: 동적 도구 레지스트리 |
