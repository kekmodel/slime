"""
Live Episode Test - Uses Claude API as agent (instead of sglang).

Tests the full pipeline including tool parsing (generate_with_gym.py flow):
1. Environment setup
2. Claude API call → raw text format conversion
3. tool_adapter.parse() for tool call parsing (same as sglang flow)
4. Token/loss_mask calculation (with dual-message tracking)
5. Sample data structure validation

Key difference from direct Claude API usage:
- Converts Claude's structured tool_calls to raw text (Qwen format)
- Parses with tool_adapter.parse() to test the actual parsing logic
- This mirrors generate_with_gym.py's sglang flow
"""

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from message_utils import (
    build_tool_specs,
    clean_user_content,
    get_token_delta,
    prepare_initial_messages,
    prepare_prompt_tokens,
    strip_think_from_previous_turns,
)
from slime_gym_env import SlimeAgentGymEnv
from tool_parser import create_tool_adapter
from tau2.config import DEFAULT_LLM_USER
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.gym import register_gym_agent
from tau2.registry import registry
from tau2.utils.llm_utils import generate as llm_generate


@dataclass
class AgentResponse:
    """Response from agent LLM."""
    content: str  # Plain text content
    tool_calls: list[dict[str, Any]]  # Structured tool calls from Claude
    raw_text: str  # Raw text format (Qwen style) for tool parsing test


def convert_to_raw_text(content: str, tool_calls: list[dict[str, Any]]) -> str:
    """
    Convert Claude's structured response to raw text format (Qwen style).

    This simulates what sglang would return, allowing us to test tool_adapter.parse().

    Qwen format:
    <tool_call>{"name": "func", "arguments": {"arg": "value"}}</tool_call>
    """
    parts = []

    # Add content if present
    if content:
        parts.append(content)

    # Convert tool calls to Qwen format
    for tc in tool_calls:
        tool_call_json = json.dumps({
            "name": tc["name"],
            "arguments": tc.get("parameters", {}),
        }, ensure_ascii=False)
        parts.append(f"<tool_call>{tool_call_json}</tool_call>")

    return "".join(parts)


# Test tokenizer
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"


@dataclass
class MockSample:
    """Mock SLIME Sample for testing."""

    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float = 0.0
    status: str = "pending"
    metadata: dict = field(default_factory=dict)
    rollout_log_probs: list[float] = field(default_factory=list)


def get_env_settings():
    """Get settings from environment."""
    return {
        "domain": os.environ.get("TAU2_DOMAIN", "telecom"),
        "max_steps": int(os.environ.get("TAU2_MAX_STEPS", "30")),  # sglang calls limit
        "max_turns": int(os.environ.get("TAU2_MAX_TURNS", "30")),  # Turn limit for testing
        "user_llm": os.environ.get("TAU2_USER_LLM", DEFAULT_LLM_USER),
        "user_temp": float(os.environ.get("TAU2_USER_TEMP", "0.0")),
        "agent_llm": os.environ.get("TAU2_USER_LLM", "claude-haiku-4-5"),  # Use same as user
        "tool_parser": os.environ.get("TAU2_TOOL_PARSER", "qwen"),
    }


def call_agent_llm(
    messages: list[dict[str, Any]],
    tools: list,
    agent_llm: str,
) -> AgentResponse:
    """
    Call Claude API as agent (replacement for sglang).

    Returns both structured tool_calls AND raw_text format for testing
    tool_adapter.parse() which is used in generate_with_gym.py.
    """
    # Convert dict messages to tau2 Message objects
    tau2_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            tau2_messages.append(SystemMessage(role="system", content=content))
        elif role == "user":
            tau2_messages.append(UserMessage(role="user", content=content))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(role="assistant", content=content))
        elif role == "tool":
            tool_content = f"[Tool Result: {msg.get('name', 'tool')}]\n{content}"
            tau2_messages.append(UserMessage(role="user", content=tool_content))

    # Call LLM
    response = llm_generate(
        model=agent_llm,
        messages=tau2_messages,
        tools=tools,
        temperature=0.7,
    )

    # Convert tool_calls to dict format
    tool_calls = []
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append({"name": tc.name, "parameters": tc.arguments})

    content = response.content or ""

    # Generate raw text format (Qwen style) for tool parsing test
    raw_text = convert_to_raw_text(content, tool_calls)

    return AgentResponse(content=content, tool_calls=tool_calls, raw_text=raw_text)


def run_episode(
    task_id: str,
    settings: dict,
    tokenizer: AutoTokenizer,
    max_steps: int = 100,
) -> MockSample:
    """
    Run a single episode using Claude as agent.

    Turn definition: User message → Assistant's final text response (after tool chain)
    Tool calls are intermediate steps within a turn, not separate turns.

    Returns a MockSample with the same structure as SLIME Sample.
    """
    sample = MockSample(prompt=task_id, metadata={"task_id": task_id})

    # Register gym
    register_gym_agent()

    # Create environment (SlimeAgentGymEnv provides new_messages in info)
    env = SlimeAgentGymEnv(
        domain=settings["domain"],
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=settings["user_llm"],
        user_llm_args={"temperature": settings["user_temp"]},
    )

    # Reset
    obs, info = env.reset()
    policy = info.get("policy", "")
    tools = info.get("tools", [])
    tool_specs = build_tool_specs(tools)

    logger.info(f"Policy length: {len(policy)}")
    logger.info(f"Tools: {[t['function']['name'] for t in tool_specs]}")
    logger.info(f"Initial observation: {obs[:100]}...")

    # Build initial messages (SlimeAgentGymEnv already strips role prefix)
    initial_messages: list[dict[str, Any]] = prepare_initial_messages(policy, obs)
    context_messages: list[dict[str, Any]] = copy.deepcopy(initial_messages)

    # Create tool adapter for parsing (same as generate_with_gym.py)
    tool_adapter = create_tool_adapter(tool_specs, settings["tool_parser"])
    logger.info(f"Tool adapter created with parser: {settings['tool_parser']}")

    # Turn-based samples: each turn ends with assistant's final text response
    turn_samples: list[dict[str, Any]] = []

    # Legacy prompt tokens
    _, prompt_token_ids = prepare_prompt_tokens(tokenizer, context_messages, tool_specs, reformulate=False)

    logger.info(f"Prompt tokens: {len(prompt_token_ids)}")

    # Trajectory tracking
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    tool_call_idx = 0

    terminated = False
    step_count = 0
    turn_count = 0
    total_reward = 0.0

    # Turn tracking: accumulate until final answer
    turn_prompt_tokens: list[int] = []
    turn_response_tokens: list[int] = []
    turn_loss_mask: list[int] = []
    turn_response_text: str = ""

    # Get max_turns from settings
    max_turns = settings.get("max_turns", 10)

    # Multi-turn loop
    # tau2 step() runs until user responds, so each step may include:
    # - Tool responses (if tool calls were made)
    # - User response (always, unless terminated)
    while not terminated and step_count < max_steps and turn_count < max_turns:
        step_count += 1
        logger.info(f"\n=== Step {step_count} ===")

        # Save prompt tokens at turn start (before any generation in this turn)
        if not turn_prompt_tokens:
            turn_prompt = tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tool_specs,
            )
            turn_prompt_tokens = tokenizer.encode(turn_prompt, add_special_tokens=False)

        # Call agent LLM using CONTEXT messages
        agent_response = call_agent_llm(context_messages, tools, settings["agent_llm"])

        logger.info(f"Agent raw_text: {agent_response.raw_text[:200] if agent_response.raw_text else '[empty]'}...")

        # Parse tool calls using tool_adapter (same as generate_with_gym.py)
        # This tests the actual parsing logic used in production
        parsed = tool_adapter.parse(agent_response.raw_text)

        if not parsed.success:
            logger.warning(f"Tool parsing failed: {parsed.error}")
            logger.debug(f"Raw text: {agent_response.raw_text[:300]}...")

        if parsed.calls:
            logger.info(f"Parsed tool calls: {[c['name'] for c in parsed.calls]}")

        # Build assistant message from PARSED result (not direct tool_calls)
        # This mirrors generate_with_gym.py flow
        if parsed.success and parsed.calls:
            tool_calls = []
            for call in parsed.calls:
                tool_call_id = f"functions.{call['name']}:{tool_call_idx}"
                tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call.get("parameters", {}),
                    },
                })
                tool_call_idx += 1
            assistant_msg = {
                "role": "assistant",
                "content": parsed.normal_text,  # Use parsed normal_text
                "tool_calls": tool_calls,
            }
        else:
            assistant_msg = {"role": "assistant", "content": agent_response.raw_text}

        # Add assistant message to context
        context_messages.append(copy.deepcopy(assistant_msg))

        # Calculate assistant response tokens (mask=1, trainable)
        token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
        turn_response_tokens.extend(token_ids)
        turn_loss_mask.extend(mask)
        response_token_ids.extend(token_ids)
        loss_masks.extend(mask)

        # Save raw LLM output for turn sample (always save raw_text)
        turn_response_text = agent_response.raw_text

        # Format action for tau2 environment (using parsed result)
        # NOTE: tau2's parse_action_string only handles SINGLE tool calls
        # For multi-tool calls, we need to step through each one separately
        if parsed.success and parsed.calls:
            # Build list of action strings (one per tool call)
            action_list = []
            for i, call in enumerate(parsed.calls):
                name = call["name"]
                params = call.get("parameters", {})
                if params:
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
                    action_list.append({
                        "action_str": f"{name}({args_str})",
                        "name": name,
                        "id": f"functions.{name}:{tool_call_idx - len(parsed.calls) + i}",
                    })
                else:
                    action_list.append({
                        "action_str": f"{name}()",
                        "name": name,
                        "id": f"functions.{name}:{tool_call_idx - len(parsed.calls) + i}",
                    })
        else:
            # Plain text action (no tool calls)
            action_list = [{
                "action_str": parsed.normal_text or agent_response.raw_text,
                "name": None,
                "id": None,
            }]

        # Step through each action (handles multi-tool calls)
        user_msg_content = None
        turn_complete = False

        for action_info in action_list:
            action_str = action_info["action_str"]
            logger.info(f"Action: {action_str[:100]}...")

            # Set pending tool call for ID mapping before step
            if action_info["name"]:
                env.set_pending_tool_calls([{
                    "name": action_info["name"],
                    "id": action_info["id"],
                }])

            # Step environment
            obs, reward, terminated, _, info = env.step(action_str)
            total_reward = reward
            new_messages = info.get("new_messages", [])
            turn_complete = info.get("turn_complete", False)

            logger.info(f"Reward: {reward:.3f}, Terminated: {terminated}, Turn complete: {turn_complete}")

            # Process new_messages from tau2 (tool responses + user message)
            for new_msg in new_messages:
                role = new_msg.get("role")

                if role == "tool":
                    # Tool response - add to context (mask=0, not trainable)
                    context_messages.append(copy.deepcopy(new_msg))
                    token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
                    turn_response_tokens.extend(token_ids)
                    turn_loss_mask.extend(mask)
                    response_token_ids.extend(token_ids)
                    loss_masks.extend(mask)
                    logger.info(f"  Tool [{new_msg.get('name')}]: {new_msg.get('content', '')[:80]}...")

                elif role == "user":
                    # User message - save for turn boundary processing
                    user_msg_content = clean_user_content(new_msg.get("content", ""))
                    logger.info(f"  User: {user_msg_content[:80]}...")

            # If turn is complete (user responded) or terminated, stop processing more tools
            if turn_complete or terminated:
                break

        # Check turn boundary (defined by environment)
        if turn_complete:
            # Save current turn
            turn_count += 1
            turn_samples.append({
                "turn": turn_count,
                "prompt_tokens": turn_prompt_tokens,
                "response_tokens": turn_response_tokens,
                "loss_mask": turn_loss_mask,
                "response_text": turn_response_text,
            })
            logger.info(f"Turn {turn_count} saved: prompt={len(turn_prompt_tokens)}, response={len(turn_response_tokens)}")

            if terminated:
                break

            # Reset turn tracking for next turn
            turn_prompt_tokens = []
            turn_response_tokens = []
            turn_loss_mask = []
            turn_response_text = ""

            # Add user message to context for next turn
            if user_msg_content:
                context_messages = strip_think_from_previous_turns(context_messages)
                user_msg = {"role": "user", "content": user_msg_content}
                context_messages.append(copy.deepcopy(user_msg))

                token_ids, mask = get_token_delta(tokenizer, context_messages, tool_specs)
                response_token_ids.extend(token_ids)
                loss_masks.extend(mask)

    # Build final sample
    sample.tokens = prompt_token_ids + response_token_ids
    sample.loss_mask = loss_masks
    sample.response = "".join([msg.get("content", "") for msg in context_messages if msg.get("role") == "assistant"])
    sample.response_length = len(response_token_ids)
    sample.reward = total_reward

    if terminated:
        sample.status = MockSample.Status.COMPLETED
    else:
        sample.status = MockSample.Status.TRUNCATED

    sample.metadata.update(
        {
            "domain": settings["domain"],
            "steps": step_count,
            "total_messages": len(context_messages),
            # PRIMARY: Turn-based samples for training
            "turn_samples": turn_samples,
            "num_turns": len(turn_samples),
            "context_messages": context_messages,
        }
    )

    return sample


def validate_sample(sample: MockSample) -> dict[str, Any]:
    """Validate sample data structure."""
    results = {
        "valid": True,
        "errors": [],
        "stats": {},
    }

    # Check tokens
    if not sample.tokens:
        results["errors"].append("No tokens")
        results["valid"] = False
    else:
        results["stats"]["total_tokens"] = len(sample.tokens)

    # Check loss_mask
    if not sample.loss_mask:
        results["errors"].append("No loss_mask")
        results["valid"] = False
    else:
        results["stats"]["loss_mask_len"] = len(sample.loss_mask)
        results["stats"]["trainable_tokens"] = sum(sample.loss_mask)
        results["stats"]["non_trainable_tokens"] = len(sample.loss_mask) - sum(sample.loss_mask)

    # Check alignment
    if sample.loss_mask and len(sample.loss_mask) != sample.response_length:
        results["errors"].append(
            f"loss_mask length ({len(sample.loss_mask)}) != response_length ({sample.response_length})"
        )
        # This might be okay depending on implementation

    # Check reward
    results["stats"]["reward"] = sample.reward
    results["stats"]["status"] = sample.status
    results["stats"]["steps"] = sample.metadata.get("steps", 0)

    # Validate turn-based samples
    turn_samples = sample.metadata.get("turn_samples", [])
    if turn_samples:
        results["stats"]["num_turns"] = len(turn_samples)

        # Validate each turn
        for i, turn in enumerate(turn_samples):
            turn_key = f"turn_{i+1}"
            prompt_len = len(turn.get("prompt_tokens", []))
            response_len = len(turn.get("response_tokens", []))
            loss_mask_len = len(turn.get("loss_mask", []))

            results["stats"][f"{turn_key}_prompt_tokens"] = prompt_len
            results["stats"][f"{turn_key}_response_tokens"] = response_len

            # Check response tokens match loss_mask
            if response_len != loss_mask_len:
                results["errors"].append(
                    f"Turn {i+1}: response_tokens ({response_len}) != loss_mask ({loss_mask_len})"
                )
                results["valid"] = False

            # Check all loss_mask are 1 (trainable)
            trainable = sum(turn.get("loss_mask", []))
            if trainable != response_len:
                results["errors"].append(
                    f"Turn {i+1}: not all response tokens are trainable ({trainable}/{response_len})"
                )

        # Total trainable across all turns
        total_trainable = sum(sum(t.get("loss_mask", [])) for t in turn_samples)
        results["stats"]["total_trainable_tokens"] = total_trainable

    return results


def main():
    """Run live episode test."""
    logger.info("=== Live Episode Test ===")

    # Load settings
    settings = get_env_settings()
    logger.info(f"Settings: {json.dumps(settings, indent=2)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Get a task
    domain = settings["domain"]
    tasks = registry.get_tasks_loader(domain)("train")
    task_id = tasks[0].id
    logger.info(f"Testing with task: {task_id}")

    # Run episode until completion
    sample = run_episode(
        task_id=task_id,
        settings=settings,
        tokenizer=tokenizer,
    )

    # Validate
    logger.info("\n=== Validation ===")
    validation = validate_sample(sample)

    logger.info(f"Valid: {validation['valid']}")
    if validation["errors"]:
        for err in validation["errors"]:
            logger.error(f"  - {err}")

    logger.info("Stats:")
    for key, value in validation["stats"].items():
        logger.info(f"  {key}: {value}")

    # Print sample summary
    logger.info("\n=== Sample Summary ===")
    logger.info(f"Status: {sample.status}")
    logger.info(f"Reward: {sample.reward:.3f}")
    logger.info(f"Steps: {sample.metadata.get('steps', 0)}")
    logger.info(f"Total tokens: {len(sample.tokens)}")
    logger.info(f"Response tokens: {sample.response_length}")
    logger.info(f"Trainable tokens: {sum(sample.loss_mask)}")
    logger.info(f"Response preview: {sample.response[:200]}...")

    # Print turn-based samples info
    if "turn_samples" in sample.metadata:
        logger.info("\n=== Turn-Based Samples ===")
        logger.info(f"Number of turns: {validation['stats'].get('num_turns', 0)}")
        logger.info(f"Total trainable tokens: {validation['stats'].get('total_trainable_tokens', 0)}")
        for i in range(validation["stats"].get("num_turns", 0)):
            turn_key = f"turn_{i+1}"
            prompt = validation["stats"].get(f"{turn_key}_prompt_tokens", 0)
            response = validation["stats"].get(f"{turn_key}_response_tokens", 0)
            logger.info(f"  Turn {i+1}: prompt={prompt}, response={response}")
        logger.info(f"Final reward (shared): {sample.reward:.3f}")

    # Save sample to file
    output_path = os.path.join(os.path.dirname(__file__), "test_sample_output.json")
    turn_samples_clean = []
    for turn in sample.metadata.get("turn_samples", []):
        turn_samples_clean.append({
            "turn": turn["turn"],
            "prompt_tokens_len": len(turn["prompt_tokens"]),
            "response_tokens_len": len(turn["response_tokens"]),
            "loss_mask_sum": sum(turn["loss_mask"]),
            "response_text": turn["response_text"],
        })

    sample_data = {
        "prompt": sample.prompt,
        "tokens_len": len(sample.tokens),
        "loss_mask_len": len(sample.loss_mask),
        "trainable_tokens": sum(sample.loss_mask),
        "response": sample.response,
        "response_length": sample.response_length,
        "reward": sample.reward,
        "status": sample.status,
        "metadata": {
            "task_id": sample.metadata.get("task_id"),
            "domain": sample.metadata.get("domain"),
            "steps": sample.metadata.get("steps"),
            "num_turns": sample.metadata.get("num_turns"),
            "turn_samples": turn_samples_clean,
            "context_messages": sample.metadata.get("context_messages"),
        },
    }
    with open(output_path, "w") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSample saved to: {output_path}")

    return sample, validation


if __name__ == "__main__":
    sample, validation = main()

    if not validation["valid"]:
        sys.exit(1)
