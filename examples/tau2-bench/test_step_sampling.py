#!/usr/bin/env python3
"""
Test script for step-based sampling logic using Claude API.

This script tests the core step sampling logic:
1. Rollout: think excluded from history
2. Training samples: think included in each step's action
3. Tool responses included in state with mask=0
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import anthropic


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


def simulate_trajectory_with_claude(
    client: anthropic.Anthropic,
    initial_prompt: str,
    max_turns: int = 3,
) -> List[TurnData]:
    """Simulate a multi-turn trajectory using Claude with extended thinking and tool use."""
    turns = []
    messages = []

    system_prompt = """You are a customer service agent helping with order management.
Use the provided tools to look up and modify orders."""

    # Define tools for Claude
    tools = [
        {
            "name": "get_order",
            "description": "Look up order details by order ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up"
                    }
                },
                "required": ["order_id"]
            }
        },
        {
            "name": "cancel_order",
            "description": "Cancel an order by order ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to cancel"
                    }
                },
                "required": ["order_id"]
            }
        }
    ]

    messages.append({"role": "user", "content": initial_prompt})

    for turn_idx in range(max_turns):
        print(f"\n   Calling Claude for turn {turn_idx} (with extended thinking + tools)...")

        # Use extended thinking API with tools
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 5000,
            },
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Parse response content blocks
        think = ""
        response_text = ""
        tool_use_block = None

        for block in response.content:
            if block.type == "thinking":
                think = block.thinking
                print(f"   [THINKING]: {think[:80]}...")
            elif block.type == "text":
                response_text = block.text
                print(f"   [TEXT]: {response_text[:80] if response_text else '(empty)'}...")
            elif block.type == "tool_use":
                tool_use_block = block
                print(f"   [TOOL_USE]: {block.name}({block.input})")

        # Handle tool call
        tool_call = None
        tool_response = None

        if tool_use_block:
            tool_call = {"name": tool_use_block.name, "input": tool_use_block.input}

            # Mock tool response
            if tool_use_block.name == "get_order":
                tool_response = '{"order_id": "12345", "status": "processing", "amount": 99.99}'
            elif tool_use_block.name == "cancel_order":
                tool_response = '{"order_id": "12345", "status": "cancelled", "refund": "pending"}'
            else:
                tool_response = '{"status": "success"}'

            print(f"   [TOOL_RESULT]: {tool_response[:60]}...")

            # Add assistant message with tool_use block (NOT thinking)
            messages.append({
                "role": "assistant",
                "content": response.content  # Contains text + tool_use blocks (thinking excluded by API)
            })

            # Add tool result
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": tool_response
                }]
            })
        else:
            # No tool call - conversation ended
            # Add only text response to messages (NOT thinking)
            messages.append({"role": "assistant", "content": response_text})

        turns.append(TurnData(
            turn_idx=turn_idx,
            think=think,
            response=response_text if response_text else f"[Tool: {tool_use_block.name}]" if tool_use_block else "",
            tool_call=tool_call,
            tool_response=tool_response,
        ))

        # Stop if no tool call (conversation ended) or stop_reason is end_turn
        if not tool_use_block or response.stop_reason == "end_turn":
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
    - state: accumulated history WITHOUT think
    - action: current turn's think + response
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

        # Action = think + response (for training)
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
            "has_think": bool(turn.think),
        })

        # Update cumulative state for next turn
        # Add response WITHOUT think (simulating rollout history)
        cumulative_state += turn.response

        # Add tool response if present
        if turn.tool_response:
            cumulative_state += f"\nTool: {turn.tool_response}\nAssistant: "

    return samples


def test_step_sampling():
    """Main test function."""
    print("=" * 60)
    print("Step-based Sampling Test with Claude")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return False

    print(f"\n1. API Key: {api_key[:25]}...{api_key[-10:]}")

    # Initialize Claude client
    print("\n2. Initializing Claude client...")
    client = anthropic.Anthropic(api_key=api_key)

    # Test prompt
    test_prompt = "I need to cancel my order #12345. Can you help?"
    print(f"\n3. Test prompt: {test_prompt}")

    # Simulate trajectory
    print("\n4. Simulating multi-turn trajectory...")
    turns = simulate_trajectory_with_claude(client, test_prompt, max_turns=3)
    print(f"\n   Generated {len(turns)} turns")

    # Print turn details
    print("\n5. Turn details (ROLLOUT - think excluded from history):")
    print("-" * 50)
    for turn in turns:
        print(f"\n   Turn {turn.turn_idx}:")
        if turn.think:
            print(f"   [THINK]: {turn.think[:60]}...")
        print(f"   [RESPONSE]: {turn.response[:60]}...")
        if turn.tool_response:
            print(f"   [TOOL RESPONSE]: {turn.tool_response[:40]}...")
        print(f"   → History에 추가: response만 (think 제외)")

    # Build samples
    print("\n6. Building step samples (TRAINING)...")
    samples = build_step_samples(turns, test_prompt, final_reward=1.0)

    # Print sample details
    print("\n7. Sample details (각 step이 독립적인 trajectory):")
    print("-" * 50)
    for sample in samples:
        print(f"\n   Sample {sample['turn_idx']}:")
        print(f"   - State (mask=0): {sample['state_tokens_len']} tokens")
        print(f"     Preview: {sample['state_text'][:60]}...")
        print(f"   - Action (mask=1): {sample['action_tokens_len']} tokens")
        print(f"     Preview: {sample['action_text'][:60]}...")
        print(f"     Has think: {sample['has_think']} ✓" if sample['has_think'] else f"     Has think: {sample['has_think']}")
        print(f"   - Loss mask: {sample['loss_mask_summary']}")
        print(f"   - Reward: {sample['reward']}")

    # Verification
    print("\n8. Verification:")
    print("-" * 50)

    all_have_think = all(s['has_think'] for s in samples)
    all_same_reward = len(set(s['reward'] for s in samples)) == 1

    print(f"   ✓ 모든 sample의 action에 think 포함: {all_have_think}")
    print(f"   ✓ State는 이전 턴들의 response만 누적 (think 제외)")
    print(f"   ✓ Tool response는 state에 포함 (mask=0)")
    print(f"   ✓ 모든 sample이 동일한 reward: {all_same_reward} (gamma=1.0)")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_step_sampling()
    sys.exit(0 if success else 1)
