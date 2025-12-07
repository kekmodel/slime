"""
새 SlimeGymEnv 테스트.

실행:
    python slime/examples/tau2-bench/test_slime_env.py
"""

import os
import sys

# 환경 변수 설정
os.environ.setdefault("TAU2_DOMAIN", "telecom")
os.environ.setdefault("TAU2_USER_LLM", "claude-haiku-4-5")

from dotenv import load_dotenv
load_dotenv()

from slime_env import (
    SlimeGymEnv,
    ToolCallAction,
    TextAction,
    make_tool_call_action,
    make_text_action,
)


def test_basic_reset():
    """기본 reset 테스트."""
    print("\n" + "=" * 60)
    print("TEST: Basic Reset")
    print("=" * 60)

    env = SlimeGymEnv(
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        max_steps=10,
        user_llm="claude-haiku-4-5",
    )

    obs, info = env.reset()

    print(f"\nObservation keys: {obs.keys()}")
    print(f"Number of initial messages: {len(obs['messages'])}")
    print(f"Number of tools: {len(obs['tools'])}")
    print(f"Policy length: {len(obs['policy'])} chars")

    if obs['messages']:
        print(f"\nFirst message:")
        print(f"  Role: {obs['messages'][0]['role']}")
        print(f"  Content: {obs['messages'][0]['content'][:200]}...")

    print(f"\nInfo: {info}")

    assert "messages" in obs
    assert "tools" in obs
    assert "policy" in obs
    print("\n[PASS] Basic reset test passed!")


def test_tool_call_action():
    """Tool call action 테스트."""
    print("\n" + "=" * 60)
    print("TEST: Tool Call Action")
    print("=" * 60)

    env = SlimeGymEnv(
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        max_steps=10,
        user_llm="claude-haiku-4-5",
    )

    obs, info = env.reset()
    print(f"Initial message: {obs['messages'][0]['content'][:100] if obs['messages'] else 'None'}...")

    # Tool call action
    action = [
        make_tool_call_action(
            name="get_customer_by_phone",
            arguments={"phone_number": "555-1234"},
            call_id="call_001",
        )
    ]

    print(f"\nExecuting tool call: get_customer_by_phone(phone_number='555-1234')")
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nResult:")
    print(f"  New messages: {len(obs['messages'])}")
    print(f"  Terminated: {terminated}")
    print(f"  Reward: {reward}")

    if obs['messages']:
        for i, msg in enumerate(obs['messages']):
            print(f"\n  Message {i}:")
            print(f"    Role: {msg['role']}")
            if msg['role'] == 'tool':
                print(f"    Tool call ID: {msg.get('tool_call_id', 'N/A')}")
                print(f"    Name: {msg.get('name', 'N/A')}")
            content = msg.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    Content: {content}")

    print("\n[PASS] Tool call action test passed!")


def test_multi_tool_call():
    """Multi-tool call 테스트."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Tool Call")
    print("=" * 60)

    env = SlimeGymEnv(
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        max_steps=10,
        user_llm="claude-haiku-4-5",
    )

    obs, info = env.reset()

    # Multiple tool calls
    action = [
        make_tool_call_action(
            name="get_customer_by_phone",
            arguments={"phone_number": "555-1234"},
            call_id="call_001",
        ),
        make_tool_call_action(
            name="get_data_usage",
            arguments={"phone_number": "555-1234"},
            call_id="call_002",
        ),
    ]

    print(f"Executing 2 tool calls simultaneously...")
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nResult:")
    print(f"  New messages: {len(obs['messages'])}")

    # 2개의 tool response가 있어야 함
    tool_responses = [m for m in obs['messages'] if m['role'] == 'tool']
    print(f"  Tool responses: {len(tool_responses)}")

    for i, msg in enumerate(tool_responses):
        print(f"\n  Tool Response {i}:")
        print(f"    ID: {msg.get('tool_call_id', 'N/A')}")
        print(f"    Name: {msg.get('name', 'N/A')}")

    assert len(tool_responses) == 2, f"Expected 2 tool responses, got {len(tool_responses)}"
    assert tool_responses[0]['tool_call_id'] == 'call_001'
    assert tool_responses[1]['tool_call_id'] == 'call_002'

    print("\n[PASS] Multi-tool call test passed!")


def test_text_action():
    """Text action 테스트."""
    print("\n" + "=" * 60)
    print("TEST: Text Action")
    print("=" * 60)

    env = SlimeGymEnv(
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        max_steps=10,
        user_llm="claude-haiku-4-5",
    )

    obs, info = env.reset()

    # Text action
    action = make_text_action("Hello! How can I help you today?")

    print(f"Sending text: 'Hello! How can I help you today?'")
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nResult:")
    print(f"  New messages: {len(obs['messages'])}")
    print(f"  Terminated: {terminated}")

    if obs['messages']:
        print(f"\n  User response:")
        print(f"    Content: {obs['messages'][0]['content'][:200]}...")

    print("\n[PASS] Text action test passed!")


def test_context_messages():
    """get_context_messages() 테스트."""
    print("\n" + "=" * 60)
    print("TEST: Context Messages")
    print("=" * 60)

    env = SlimeGymEnv(
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        max_steps=10,
        user_llm="claude-haiku-4-5",
    )

    obs, info = env.reset()

    # 몇 가지 action 실행
    env.step([make_tool_call_action("get_customer_by_phone", {"phone_number": "555-1234"}, "call_001")])
    env.step(make_text_action("I found your account."))

    # 전체 context 가져오기
    context = env.get_context_messages()

    print(f"Total context messages: {len(context)}")
    for i, msg in enumerate(context):
        role = msg['role']
        content = msg.get('content', '')
        if content and len(content) > 80:
            content = content[:80] + "..."
        print(f"  [{i}] {role}: {content}")

    assert len(context) >= 3  # user, assistant (tool call), tool, assistant (text), user
    print("\n[PASS] Context messages test passed!")


def test_full_episode():
    """전체 에피소드 테스트."""
    print("\n" + "=" * 60)
    print("TEST: Full Episode")
    print("=" * 60)

    env = SlimeGymEnv(
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        max_steps=20,
        user_llm="claude-haiku-4-5",
    )

    obs, info = env.reset()
    print(f"Initial user message: {obs['messages'][0]['content'][:100] if obs['messages'] else 'None'}...")

    step_count = 0
    terminated = False

    while not terminated and step_count < 5:  # 최대 5 step만 테스트
        step_count += 1
        print(f"\n--- Step {step_count} ---")

        # 간단히 tool call 후 text 응답
        if step_count == 1:
            action = [make_tool_call_action("get_customer_by_phone", {"phone_number": "555-1234"}, f"call_{step_count}")]
        elif step_count == 2:
            action = make_text_action("I've found your account. Let me check your subscription details.")
        elif step_count == 3:
            action = [make_tool_call_action("done", {}, f"call_{step_count}")]
        else:
            action = make_text_action("Thank you for contacting us!")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  New messages: {len(obs['messages'])}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Reward: {reward}")

    print(f"\nFinal context length: {len(env.get_context_messages())}")
    print("\n[PASS] Full episode test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("SlimeGymEnv Tests")
    print("=" * 60)

    try:
        test_basic_reset()
        test_tool_call_action()
        test_multi_tool_call()
        test_text_action()
        test_context_messages()
        test_full_episode()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
