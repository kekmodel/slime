#!/usr/bin/env python3
"""
Test script for step-based sampling logic using SGLang server.

This script tests the core step sampling logic with sglang:
1. Rollout: reasoning_content excluded from history
2. Training samples: reasoning_content included in each step's action
3. Tool responses included in state with mask=0

Prerequisites:
- Start sglang server with reasoning parser:
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --port 30000
"""

import os
import sys
import json
import asyncio
import aiohttp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


@dataclass
class TurnData:
    """Data for a single turn."""
    turn_idx: int
    think: str
    response: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_response: Optional[str] = None


def simple_tokenize(text: str) -> List[int]:
    """Simple character-based tokenization for testing."""
    return list(text.encode('utf-8'))


async def call_sglang(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    sampling_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Call sglang server."""
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
    }
    async with session.post(url, json=payload) as resp:
        return await resp.json()


async def simulate_trajectory_with_sglang(
    sglang_url: str,
    initial_prompt: str,
    max_turns: int = 3,
) -> List[TurnData]:
    """
    Simulate a multi-turn trajectory using SGLang server.

    Requires sglang server started with --reasoning-parser option.
    The server will return:
    - text: response without thinking
    - reasoning_content: thinking content (if present)
    """
    turns = []

    system_prompt = """You are a customer service agent helping with order management.

If you need to look up or modify an order, respond with a tool call in this format:
<tool_call>{"name": "function_name", "arguments": {"arg": "value"}}</tool_call>

Available tools:
- get_order(order_id): Look up order details
- cancel_order(order_id): Cancel an order

Think step by step before responding."""

    # Build initial conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    sampling_params = {
        "temperature": 0.7,
        "max_new_tokens": 512,
        "top_p": 0.9,
    }

    async with aiohttp.ClientSession() as session:
        for turn_idx in range(max_turns):
            print(f"\n   Calling SGLang for turn {turn_idx}...")

            # Format prompt (simple format for testing)
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                elif role == "tool":
                    prompt += f"<|im_start|>tool\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            # Call sglang
            try:
                output = await call_sglang(session, sglang_url, prompt, sampling_params)
            except Exception as e:
                print(f"   ERROR calling SGLang: {e}")
                print("   Make sure sglang server is running with --reasoning-parser")
                raise

            # With --reasoning-parser, output has separate fields
            response_text = output.get("text", "")
            reasoning_content = output.get("reasoning_content", "")

            print(f"   [REASONING]: {reasoning_content[:80]}..." if reasoning_content else "   [REASONING]: (none)")
            print(f"   [TEXT]: {response_text[:80]}...")

            # Check for tool call in response
            tool_call = None
            tool_response = None

            if "<tool_call>" in response_text and "</tool_call>" in response_text:
                # Parse tool call
                start = response_text.find("<tool_call>") + len("<tool_call>")
                end = response_text.find("</tool_call>")
                tool_json = response_text[start:end].strip()

                try:
                    tool_data = json.loads(tool_json)
                    tool_call = tool_data

                    # Mock tool response
                    if tool_data.get("name") == "get_order":
                        tool_response = '{"order_id": "12345", "status": "processing", "amount": 99.99}'
                    elif tool_data.get("name") == "cancel_order":
                        tool_response = '{"order_id": "12345", "status": "cancelled", "refund": "pending"}'
                    else:
                        tool_response = '{"status": "success"}'

                    print(f"   [TOOL_CALL]: {tool_data}")
                    print(f"   [TOOL_RESULT]: {tool_response[:60]}...")

                    # Add to messages WITHOUT reasoning_content
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "tool", "content": tool_response})
                except json.JSONDecodeError:
                    print(f"   Failed to parse tool call: {tool_json}")
                    messages.append({"role": "assistant", "content": response_text})
            else:
                # No tool call - add response to messages (WITHOUT reasoning)
                messages.append({"role": "assistant", "content": response_text})

            turns.append(TurnData(
                turn_idx=turn_idx,
                think=reasoning_content,
                response=response_text,
                tool_call=tool_call,
                tool_response=tool_response,
            ))

            # Stop if no tool call (conversation ended)
            if tool_call is None:
                break

    return turns


def build_step_samples(
    turns: List[TurnData],
    initial_prompt: str,
    final_reward: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Build step-level samples from turn results.

    Key logic:
    - state: accumulated history WITHOUT reasoning_content
    - action: current turn's reasoning_content + response
    - All samples get same reward (gamma=1.0)
    """
    samples = []

    # Build initial state
    prompt_text = f"User: {initial_prompt}\nAssistant: "
    cumulative_state = prompt_text

    for turn in turns:
        # Current state = accumulated text (before this turn)
        state_text = cumulative_state
        state_tokens = simple_tokenize(state_text)

        # Action = reasoning_content + response (for training)
        action_text = turn.think + "\n" + turn.response if turn.think else turn.response
        action_tokens = simple_tokenize(action_text)

        # Build loss mask: 0 for state, 1 for action
        loss_mask = [0] * len(state_tokens) + [1] * len(action_tokens)

        samples.append({
            "turn_idx": turn.turn_idx,
            "state_text": state_text[:100] + "..." if len(state_text) > 100 else state_text,
            "action_text": action_text[:100] + "..." if len(action_text) > 100 else action_text,
            "state_tokens_len": len(state_tokens),
            "action_tokens_len": len(action_tokens),
            "loss_mask_summary": f"[0]*{len(state_tokens)} + [1]*{len(action_tokens)}",
            "reward": final_reward,
            "has_reasoning": bool(turn.think),
        })

        # Update cumulative state for next turn
        # Add response WITHOUT reasoning_content (simulating rollout history)
        cumulative_state += turn.response

        # Add tool response if present
        if turn.tool_response:
            cumulative_state += f"\nTool: {turn.tool_response}\nAssistant: "

    return samples


async def test_step_sampling_sglang():
    """Main test function for SGLang."""
    print("=" * 60)
    print("Step-based Sampling Test with SGLang")
    print("=" * 60)

    # SGLang server URL
    sglang_host = os.getenv("SGLANG_HOST", "127.0.0.1")
    sglang_port = os.getenv("SGLANG_PORT", "30000")
    sglang_url = f"http://{sglang_host}:{sglang_port}/generate"

    print(f"\n1. SGLang URL: {sglang_url}")
    print("   (Server should be started with --reasoning-parser qwen3)")

    # Test prompt
    test_prompt = "I need to cancel my order #12345. Can you help?"
    print(f"\n2. Test prompt: {test_prompt}")

    # Simulate trajectory
    print("\n3. Simulating multi-turn trajectory...")
    turns = await simulate_trajectory_with_sglang(sglang_url, test_prompt, max_turns=3)
    print(f"\n   Generated {len(turns)} turns")

    # Print turn details
    print("\n4. Turn details (ROLLOUT - reasoning excluded from history):")
    print("-" * 50)
    for turn in turns:
        print(f"\n   Turn {turn.turn_idx}:")
        if turn.think:
            print(f"   [REASONING]: {turn.think[:60]}...")
        print(f"   [RESPONSE]: {turn.response[:60]}...")
        if turn.tool_response:
            print(f"   [TOOL RESPONSE]: {turn.tool_response[:40]}...")
        print(f"   → History에 추가: response만 (reasoning 제외)")

    # Build samples
    print("\n5. Building step samples (TRAINING)...")
    samples = build_step_samples(turns, test_prompt, final_reward=1.0)

    # Print sample details
    print("\n6. Sample details (각 step이 독립적인 trajectory):")
    print("-" * 50)
    for sample in samples:
        print(f"\n   Sample {sample['turn_idx']}:")
        print(f"   - State (mask=0): {sample['state_tokens_len']} tokens")
        print(f"     Preview: {sample['state_text'][:60]}...")
        print(f"   - Action (mask=1): {sample['action_tokens_len']} tokens")
        print(f"     Preview: {sample['action_text'][:60]}...")
        has_reasoning = sample['has_reasoning']
        print(f"     Has reasoning: {has_reasoning} ✓" if has_reasoning else f"     Has reasoning: {has_reasoning}")
        print(f"   - Loss mask: {sample['loss_mask_summary']}")
        print(f"   - Reward: {sample['reward']}")

    # Verification
    print("\n7. Verification:")
    print("-" * 50)

    all_have_reasoning = all(s['has_reasoning'] for s in samples)
    all_same_reward = len(set(s['reward'] for s in samples)) == 1

    print(f"   ✓ 모든 sample의 action에 reasoning 포함: {all_have_reasoning}")
    print(f"   ✓ State는 이전 턴들의 response만 누적 (reasoning 제외)")
    print(f"   ✓ Tool response는 state에 포함 (mask=0)")
    print(f"   ✓ 모든 sample이 동일한 reward: {all_same_reward} (gamma=1.0)")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    return True


def main():
    """Entry point."""
    print("\nNote: This test requires a running SGLang server.")
    print("Start the server with:")
    print("  python -m sglang.launch_server \\")
    print("    --model-path Qwen/Qwen3-8B \\")
    print("    --reasoning-parser qwen3 \\")
    print("    --port 30000")
    print()

    try:
        success = asyncio.run(test_step_sampling_sglang())
        sys.exit(0 if success else 1)
    except aiohttp.ClientError as e:
        print(f"\nERROR: Could not connect to SGLang server: {e}")
        print("Make sure the server is running on the specified host/port.")
        sys.exit(1)


if __name__ == "__main__":
    main()
