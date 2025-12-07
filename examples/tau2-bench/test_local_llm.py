"""
Qwen3-0.6B 로컬 LLM으로 에이전트 테스트.

실행:
    python slime/examples/tau2-bench/test_local_llm.py
"""

import json
import os
from typing import Any

os.environ.setdefault("TAU2_DOMAIN", "telecom")
os.environ.setdefault("TAU2_USER_LLM", "claude-haiku-4-5")

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slime_env import SlimeGymEnv, ToolCallAction, TextAction
from message_utils import get_token_delta
from tool_parser import create_tool_adapter


def run_local_agent(
    task_id: str,
    model_name: str = "Qwen/Qwen3-0.6B",
    max_turns: int = 15,
    max_new_tokens: int = 512,
) -> dict:
    """
    로컬 LLM으로 에이전트 실행.
    """
    print(f"Loading model: {model_name}")

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # MPS (Mac)에서는 bfloat16 미지원 → CPU로 로드 후 float32 변환
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU 호환
        device_map="cpu",  # 먼저 CPU로 로드
        trust_remote_code=True,
    )

    # MPS는 메모리 제한으로 긴 프롬프트에서 실패 → CPU 유지
    # (프롬프트가 7000+ 토큰으로 매우 김)
    print("Running on CPU (MPS memory limit for long prompts)")

    model.eval()

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

    # Context 초기화
    context_messages: list[dict[str, Any]] = [
        {"role": "system", "content": policy},
    ]
    context_messages.extend(initial_messages)

    # Tool adapter
    tool_adapter = create_tool_adapter(tool_specs, "qwen")

    # 추적 변수
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    tool_call_idx = 0

    terminated = False
    turn_count = 0
    total_reward = 0.0

    # 멀티턴 루프
    while not terminated and turn_count < max_turns:
        turn_count += 1
        print(f"\n{'='*50}")
        print(f"Turn {turn_count}")
        print('='*50)

        # 프롬프트 생성
        prompt = tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tool_specs,
        )

        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 새로 생성된 토큰만 추출
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # EOS 토큰 제거
        if assistant_response.endswith("<|im_end|>"):
            assistant_response = assistant_response[:-len("<|im_end|>")]
        if assistant_response.endswith("<|endoftext|>"):
            assistant_response = assistant_response[:-len("<|endoftext|>")]

        print(f"\n[Assistant]: {assistant_response[:300]}...")

        # Tool call 파싱
        parsed = tool_adapter.parse(assistant_response)

        if parsed.success and parsed.calls:
            print(f"\n[Tool Calls]: {[c['name'] for c in parsed.calls]}")

            # Tool call action
            actions = []
            for call in parsed.calls:
                call_id = f"call_{tool_call_idx}"
                tool_call_idx += 1
                actions.append(ToolCallAction(
                    name=call["name"],
                    arguments=call.get("parameters", {}),
                    id=call_id,
                ))

            # Context에 assistant 추가
            assistant_msg = {
                "role": "assistant",
                "content": parsed.normal_text or "",
                "tool_calls": [
                    {
                        "id": a.id,
                        "type": "function",
                        "function": {
                            "name": a.name,
                            "arguments": json.dumps(a.arguments),
                        },
                    }
                    for a in actions
                ],
            }
            context_messages.append(assistant_msg)

            # 토큰 계산
            token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
            response_token_ids.extend(token_ids)
            loss_masks.extend(mask)

            # 환경 step
            obs, reward, terminated, _, _ = env.step(actions)
            total_reward = reward

            # Tool response 추가
            for msg in obs["messages"]:
                if msg.get("role") == "tool":
                    print(f"\n[Tool Response]: {msg.get('content', '')[:200]}...")
                    context_messages.append(msg)
                    token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
                    response_token_ids.extend(token_ids)
                    loss_masks.extend(mask)

        else:
            # Text action
            print(f"\n[Text Action]")
            action = TextAction(content=assistant_response)

            if assistant_response.strip():
                context_messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                })

                token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
                response_token_ids.extend(token_ids)
                loss_masks.extend(mask)

            # 환경 step
            obs, reward, terminated, _, _ = env.step(action)
            total_reward = reward

            # User response 추가
            for msg in obs["messages"]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    if user_content:
                        print(f"\n[User]: {user_content[:200]}...")
                        context_messages.append(msg)
                        token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
                        response_token_ids.extend(token_ids)
                        loss_masks.extend(mask)

        print(f"\n[Status] Terminated: {terminated}, Reward: {reward}")

    # 결과
    result = {
        "task_id": task_id,
        "status": "completed" if terminated else "truncated",
        "reward": total_reward,
        "turns": turn_count,
        "response_tokens": len(response_token_ids),
        "agent_tokens": sum(loss_masks),
        "env_tokens": len(loss_masks) - sum(loss_masks),
        "context_messages": context_messages,
    }

    return result


def main():
    print("=" * 60)
    print("Local LLM Agent Test (Qwen3-0.6B)")
    print("=" * 60)

    task_id = "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]"

    result = run_local_agent(
        task_id=task_id,
        model_name="Qwen/Qwen3-0.6B",
        max_turns=10,
    )

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Reward: {result['reward']}")
    print(f"Turns: {result['turns']}")
    print(f"Response tokens: {result['response_tokens']}")
    print(f"  Agent (mask=1): {result['agent_tokens']}")
    print(f"  Env (mask=0): {result['env_tokens']}")

    # 저장
    output_path = "slime/examples/tau2-bench/test_local_llm_output.json"
    with open(output_path, "w") as f:
        # context_messages는 별도 처리
        save_result = {k: v for k, v in result.items() if k != "context_messages"}
        json.dump(save_result, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
