"""
generate_with_gym.py 로직 테스트.

sglang 없이 SlimeGymEnv + mock LLM으로 Sample 생성 흐름 확인.
실제 학습 데이터 구조를 파일로 저장하여 검증.

실행:
    python slime/examples/tau2-bench/test_generate_sample.py
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

# 환경 변수 설정
os.environ.setdefault("TAU2_DOMAIN", "telecom")
os.environ.setdefault("TAU2_USER_LLM", "claude-haiku-4-5")

from dotenv import load_dotenv

load_dotenv()

from message_utils import get_token_delta
from slime_env import SlimeGymEnv, TextAction, ToolCallAction
from transformers import AutoTokenizer


@dataclass
class MockSample:
    """SLIME Sample 모의 클래스."""

    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float = 0.0
    status: str = "pending"
    metadata: dict = field(default_factory=dict)
    rollout_log_probs: list[float] = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "response": self.response,
            "response_length": self.response_length,
            "reward": self.reward,
            "status": self.status,
            "metadata": self.metadata,
        }


def generate_sample_with_mock_llm(
    task_id: str,
    tokenizer: AutoTokenizer,
    mock_responses: list[dict],
    max_turns: int = 30,
) -> MockSample:
    """
    Mock LLM 응답으로 Sample 생성.

    Args:
        task_id: tau2 태스크 ID
        tokenizer: HuggingFace tokenizer
        mock_responses: 미리 정의된 LLM 응답 리스트
        max_turns: 최대 턴 수

    Returns:
        생성된 MockSample
    """
    sample = MockSample(prompt=task_id)
    sample.metadata["task_id"] = task_id

    # 환경 초기화
    env = SlimeGymEnv(
        domain="telecom",
        task_id=task_id,
        max_steps=100,
        user_llm="claude-haiku-4-5",
    )

    obs, _ = env.reset()

    policy = obs.get("policy", "")
    tool_specs = obs.get("tools", [])
    initial_messages = obs.get("messages", [])

    print(f"Policy length: {len(policy)}")
    print(f"Tools: {[t['function']['name'] for t in tool_specs]}")
    print(f"Initial messages: {len(initial_messages)}")

    # Context 초기화
    context_messages: list[dict[str, Any]] = [
        {"role": "system", "content": policy},
    ]
    context_messages.extend(initial_messages)

    # 추적 변수
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    tool_call_idx = 0

    terminated = False
    step_count = 0
    turn_count = 0
    total_reward = 0.0
    response_idx = 0

    # 멀티턴 상호작용 루프
    while (
        not terminated
        and turn_count < max_turns
        and response_idx < len(mock_responses)
    ):
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")

        # Mock LLM 응답 사용
        mock_resp = mock_responses[response_idx]
        response_idx += 1

        assistant_response = mock_resp.get("content", "")
        is_tool_call = mock_resp.get("is_tool_call", False)
        tool_name = mock_resp.get("tool_name", "")
        tool_args = mock_resp.get("tool_args", {})

        print(f"Mock response: {assistant_response[:100]}...")

        if is_tool_call:
            # Tool call action
            call_id = f"call_{tool_call_idx}"
            tool_call_idx += 1

            actions = [
                ToolCallAction(
                    name=tool_name,
                    arguments=tool_args,
                    id=call_id,
                )
            ]

            # Context에 assistant 메시지 추가
            assistant_msg = {
                "role": "assistant",
                "content": assistant_response,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
            context_messages.append(assistant_msg)

            # Assistant 토큰 계산 (loss_mask=1)
            token_ids, mask = get_token_delta(
                tokenizer, context_messages, tool_specs
            )
            response_token_ids.extend(token_ids)
            loss_masks.extend(mask)
            print(
                f"  Assistant tokens: {len(token_ids)}, mask sum: {sum(mask)}"
            )

            # 환경 step
            obs, reward, terminated, _, _ = env.step(actions)
            step_count += 1
            total_reward = reward

            # Tool response 추가
            for msg in obs["messages"]:
                if msg.get("role") == "tool":
                    context_messages.append(msg)
                    token_ids, mask = get_token_delta(
                        tokenizer, context_messages, tool_specs
                    )
                    response_token_ids.extend(token_ids)
                    loss_masks.extend(mask)
                    print(
                        f"  Tool response tokens: {len(token_ids)}, mask sum: {sum(mask)}"
                    )

        else:
            # Text action
            action = TextAction(content=assistant_response)

            if assistant_response:
                context_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_response,
                    }
                )

                # Assistant 토큰 계산 (loss_mask=1)
                token_ids, mask = get_token_delta(
                    tokenizer, context_messages, tool_specs
                )
                response_token_ids.extend(token_ids)
                loss_masks.extend(mask)
                print(
                    f"  Assistant tokens: {len(token_ids)}, mask sum: {sum(mask)}"
                )

            # 환경 step
            obs, reward, terminated, _, _ = env.step(action)
            step_count += 1
            total_reward = reward

            # User response 추가
            for msg in obs["messages"]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    if not user_content:
                        continue
                    context_messages.append(msg)
                    token_ids, mask = get_token_delta(
                        tokenizer, context_messages, tool_specs
                    )
                    response_token_ids.extend(token_ids)
                    loss_masks.extend(mask)
                    print(
                        f"  User response tokens: {len(token_ids)}, mask sum: {sum(mask)}"
                    )

        print(f"  Terminated: {terminated}, Reward: {reward}")

    # Prompt 토큰 계산
    initial_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": policy}] + initial_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tool_specs,
    )
    prompt_token_ids = tokenizer.encode(
        initial_prompt, add_special_tokens=False
    )

    # Response 텍스트 구성
    response_parts = []
    for msg in context_messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                response_parts.append(content)

    # Sample 설정
    sample.tokens = prompt_token_ids + response_token_ids
    sample.loss_mask = [0] * len(prompt_token_ids) + loss_masks
    sample.response = "\n\n".join(response_parts)
    sample.response_length = len(response_token_ids)
    sample.reward = total_reward
    sample.status = "completed" if terminated else "truncated"
    sample.metadata["domain"] = "telecom"
    sample.metadata["steps"] = step_count
    sample.metadata["num_turns"] = turn_count
    sample.metadata["context_messages"] = context_messages

    return sample


def main():
    print("=" * 60)
    print("Generate Sample Test")
    print("=" * 60)

    # Tokenizer 로드
    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    # 테스트 태스크
    task_id = "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]"

    # Mock LLM 응답 시나리오 - 현실적인 해결 과정
    # 태스크: airplane_mode_on | user_abroad_roaming_enabled_off
    # 해결 조건: 1) 사용자가 airplane mode 끄기, 2) agent가 roaming 활성화
    mock_responses = [
        # Turn 1: 디바이스 상태 확인 (airplane mode 확인)
        {
            "content": "I understand you're having mobile data issues while abroad. Let me check your device settings.",
            "is_tool_call": True,
            "tool_name": "get_user_device_info",
            "tool_args": {},
        },
        # Turn 2: Airplane mode 발견 → 끄라고 안내
        {
            "content": "I can see that Airplane Mode is currently ON on your device. This is blocking all wireless connections. Please go to Settings and turn OFF Airplane Mode, then let me know when you've done that.",
            "is_tool_call": False,
        },
        # Turn 3: 사용자가 껐다고 함 → 다시 확인
        {
            "content": "Let me verify the settings now.",
            "is_tool_call": True,
            "tool_name": "get_user_device_info",
            "tool_args": {},
        },
        # Turn 4: Roaming 활성화 필요 → enable_roaming 호출
        {
            "content": "Airplane Mode is now off. Since you're abroad, I need to enable Data Roaming for you.",
            "is_tool_call": True,
            "tool_name": "enable_roaming",
            "tool_args": {"customer_id": "C1001"},
        },
        # Turn 5: 완료 안내
        {
            "content": "I've enabled Data Roaming on your account. Your mobile data should now work. Is there anything else?",
            "is_tool_call": False,
        },
        # Turn 6: done
        {
            "content": "You're welcome! Have a great trip!",
            "is_tool_call": True,
            "tool_name": "done",
            "tool_args": {},
        },
    ]

    print(f"\nTask ID: {task_id}")
    print(f"Mock responses: {len(mock_responses)}")

    # Sample 생성
    sample = generate_sample_with_mock_llm(
        task_id=task_id,
        tokenizer=tokenizer,
        mock_responses=mock_responses,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("Sample Result")
    print("=" * 60)

    print(f"\nStatus: {sample.status}")
    print(f"Reward: {sample.reward}")
    print(f"Steps: {sample.metadata.get('steps')}")
    print(f"Turns: {sample.metadata.get('num_turns')}")

    print(f"\nTokens:")
    print(f"  Total: {len(sample.tokens)}")
    print(f"  Prompt: {len(sample.tokens) - sample.response_length}")
    print(f"  Response: {sample.response_length}")

    print(f"\nLoss Mask:")
    print(f"  Total: {len(sample.loss_mask)}")
    print(f"  Zeros (prompt + env): {sample.loss_mask.count(0)}")
    print(f"  Ones (agent): {sample.loss_mask.count(1)}")

    # 토큰과 마스크 정렬 확인
    assert len(sample.tokens) == len(
        sample.loss_mask
    ), f"Token/mask mismatch: {len(sample.tokens)} vs {len(sample.loss_mask)}"
    print("\n[OK] Token and loss_mask lengths match!")

    # 샘플 저장
    output_path = "slime/examples/tau2-bench/test_generate_sample_output.json"

    # context_messages는 별도 저장 (너무 큼)
    sample_dict = sample.to_dict()
    context_messages = sample_dict["metadata"].pop("context_messages", [])

    with open(output_path, "w") as f:
        json.dump(sample_dict, f, indent=2, ensure_ascii=False)
    print(f"\nSample saved to: {output_path}")

    # Context messages 별도 저장
    context_path = "slime/examples/tau2-bench/test_generate_context.json"
    with open(context_path, "w") as f:
        json.dump(context_messages, f, indent=2, ensure_ascii=False)
    print(f"Context saved to: {context_path}")

    # 토큰 디코딩 확인
    print("\n" + "=" * 60)
    print("Token Analysis")
    print("=" * 60)

    # Prompt 부분
    prompt_len = len(sample.tokens) - sample.response_length
    prompt_tokens = sample.tokens[:prompt_len]
    prompt_text = tokenizer.decode(prompt_tokens)
    print(f"\n[Prompt] ({prompt_len} tokens)")
    print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)

    # Response 부분 - mask 기준으로 분석
    response_tokens = sample.tokens[prompt_len:]
    response_mask = sample.loss_mask[prompt_len:]

    print(f"\n[Response] ({len(response_tokens)} tokens)")

    # mask=1인 토큰들 (agent)
    agent_token_indices = [i for i, m in enumerate(response_mask) if m == 1]
    if agent_token_indices:
        agent_tokens = [response_tokens[i] for i in agent_token_indices]
        agent_text = tokenizer.decode(agent_tokens)
        print(f"\n  Agent tokens (mask=1): {len(agent_tokens)}")
        print(
            f"  Text: {agent_text[:300]}..."
            if len(agent_text) > 300
            else f"  Text: {agent_text}"
        )

    # mask=0인 토큰들 (env)
    env_token_indices = [i for i, m in enumerate(response_mask) if m == 0]
    if env_token_indices:
        env_tokens = [response_tokens[i] for i in env_token_indices]
        env_text = tokenizer.decode(env_tokens)
        print(f"\n  Env tokens (mask=0): {len(env_tokens)}")
        print(
            f"  Text: {env_text[:300]}..."
            if len(env_text) > 300
            else f"  Text: {env_text}"
        )

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
