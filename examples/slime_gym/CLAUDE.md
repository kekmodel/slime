# CLAUDE.md

## Overview

**slime_gym** - SLIME RL 학습용 Gym 스타일 환경 추상화.

## Commands

```bash
# 테스트
python examples/slime_gym/test_gym_standalone.py

# 학습
./examples/slime_gym/run_retail.sh

# SLIME 통합
python -m slime.train \
    --custom-generate-function-path examples.slime_gym.generate_with_gym.generate \
    --custom-rm-path examples.slime_gym.generate_with_gym.reward_func
```

## Architecture

```
slime_gym/
├── base.py               # BaseEnvironment, @tool
├── types.py              # ExecutionState, ToolCall, ToolResult
├── config.py             # MAX_TURNS, resolve_max_turns()
├── env_registry.py       # @EnvironmentRegistry.register()
├── formatters.py         # ChatMLFormatter
├── generate_with_gym.py  # generate(), reward_func()
├── retail_env.py         # RetailServiceEnvironment
├── dynamic_env.py        # DynamicServiceEnvironment
└── tool_registry.py      # ToolRegistry (advanced)
```

## Core Pattern

```python
@EnvironmentRegistry.register("my_env")
class MyEnv(BaseEnvironment):
    def setup(self, metadata: dict) -> None:
        super().setup(metadata)
        # Initialize from metadata

    @tool(description="Tool description")
    async def my_tool(self, arg: str) -> str:
        return "result"

    def verify(self) -> float:
        # 1.0 = success, 0.0 = failure
        return 1.0 if self.state.has_executed_all(self.expected_actions) else 0.0
```

## Key Components

### ExecutionState
```python
self.state.executed_tools    # set[str] - 실행된 도구들
self.state.tool_results      # dict - 도구 결과
self.state.submitted_result  # Any - submit_result로 제출된 값
```

### SLIME Interface
```python
# generate_with_gym.py
async def generate(args, sample: Sample, sampling_params) -> Sample
async def reward_func(args, sample: Sample, **kwargs) -> dict
```

- `generate()`: 에이전트 루프 실행, reward를 sample.reward에 저장
- `reward_func()`: sample.reward 반환 (generate에서 미리 계산됨)

### Configuration
```bash
SLIME_GYM_MAX_TURNS=10
SLIME_GYM_MAX_TURNS_BUFFER=0
SLIME_GYM_DYNAMIC_MAX_TURNS=true
```

## Task Format

```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "metadata": {
        "env_name": "retail_service",
        "expected_actions": ["tool1", "tool2"],
        "expected_result": {"key": "value"}  # optional
    }
}
```
