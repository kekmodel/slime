"""
Live Episode Test V2 - Uses new SlimeGymEnv (no AgentGymEnv inheritance).

Key improvements over v1:
- ToolCallAction/TextAction instead of string parsing
- Structured observations (chat template compatible)
- Native multi-tool call support
- No workarounds needed

Tests:
1. Environment setup with new SlimeGymEnv
2. Claude API as agent
3. Tool parsing and execution
4. Token/loss_mask calculation
5. Sample data structure validation
"""

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

from message_utils import get_token_delta
from slime_env import SlimeGymEnv, TextAction, ToolCallAction
from tau2.config import DEFAULT_LLM_USER
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import generate as llm_generate
from tool_parser import create_tool_adapter


@dataclass
class AgentResponse:
    """Response from agent LLM."""

    content: str  # Plain text content
    tool_calls: list[dict[str, Any]]  # Structured tool calls
    raw_text: str  # Raw text format for tool parsing test
    thinking_blocks: list[dict] | None = None  # Thinking blocks from Claude


def _extract_thinking_text(thinking_blocks: list[dict]) -> str:
    """
    Extract text from Claude's thinking_blocks for Qwen3 chat template.

    Handles multiple possible formats:
        - {"type": "thinking", "thinking": "..."}
        - {"type": "thinking", "text": "..."}
        - {"type": "thinking", "content": "..."}
        - {"thinking": "..."}  (without type)
        - String directly in list

    Qwen3's chat template expects:
        {"reasoning_content": "reasoning text..."}
    """
    if not thinking_blocks:
        return ""

    parts = []
    for block in thinking_blocks:
        thinking_text = ""

        if isinstance(block, str):
            # Direct string
            thinking_text = block
        elif isinstance(block, dict):
            # Try multiple possible keys
            thinking_text = (
                block.get("thinking")
                or block.get("text")
                or block.get("content")
                or ""
            )

        if thinking_text:
            parts.append(thinking_text)

    result = "\n".join(parts)
    if not result and thinking_blocks:
        # Debug: if we have blocks but couldn't extract, log the structure
        logger.warning(f"Could not extract thinking from blocks: {thinking_blocks[:2]}...")

    return result


def convert_to_raw_text(content: str, tool_calls: list[dict[str, Any]]) -> str:
    """Convert to Qwen-style raw text format for tool parsing test."""
    parts = []
    if content:
        parts.append(content)
    for tc in tool_calls:
        tool_call_json = json.dumps(
            {
                "name": tc["name"],
                "arguments": tc.get("parameters", {}),
            },
            ensure_ascii=False,
        )
        parts.append(f"<tool_call>{tool_call_json}</tool_call>")
    return "".join(parts)


TOKENIZER_NAME = "openai/gpt-oss-20b"

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "trajectory_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_thinking_field_name(tokenizer_name: str) -> str:
    """
    Get the correct field name for reasoning/thinking based on tokenizer.

    - Qwen3: uses 'reasoning_content' → rendered as <think>...</think>
    - gpt-oss: uses 'thinking' → rendered as separate <|channel|>analysis message
    """
    if "qwen" in tokenizer_name.lower():
        return "reasoning_content"
    elif "gpt-oss" in tokenizer_name.lower():
        return "thinking"
    else:
        # Default to reasoning_content for unknown tokenizers
        return "reasoning_content"


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


def _transform_messages_for_tokenizer(
    messages: list[dict[str, Any]],
    thinking_field: str,
) -> list[dict[str, Any]]:
    """
    Transform messages for tokenizer rendering.

    For gpt-oss: Split assistant messages with thinking + tool_calls into separate messages.
    For Qwen: Convert thinking_blocks to reasoning_content field.
    """
    if thinking_field != "thinking":
        # Qwen and others: just convert thinking_blocks to reasoning_content
        result = []
        for msg in messages:
            new_msg = dict(msg)
            if msg.get("role") == "assistant" and msg.get("thinking_blocks"):
                thinking_text = _extract_thinking_text(msg["thinking_blocks"])
                if thinking_text:
                    new_msg["reasoning_content"] = thinking_text
                # Remove thinking_blocks (not needed for Qwen template)
                new_msg.pop("thinking_blocks", None)
            result.append(new_msg)
        return result

    # gpt-oss: For tool_calls messages, use thinking field (not content) for analysis
    # Template logic: if tool_calls present, renders thinking OR content as analysis channel
    # Cannot have BOTH content and thinking with tool_calls
    result = []
    for msg in messages:
        if msg.get("role") != "assistant":
            result.append(msg)
            continue

        thinking_blocks = msg.get("thinking_blocks")
        tool_calls = msg.get("tool_calls")
        content = msg.get("content", "")

        if thinking_blocks and tool_calls:
            # gpt-oss: Put thinking in 'thinking' field, content must be empty
            # Template will render thinking as analysis channel, then tool call
            # If there's content, prepend it to thinking (since we can only use one)
            thinking_text = _extract_thinking_text(thinking_blocks)
            if content and thinking_text:
                # Combine: content first, then thinking
                thinking_text = f"{content}\n\n{thinking_text}"
            elif content:
                thinking_text = content

            new_msg = {
                "role": "assistant",
                "content": "",  # Must be empty when using thinking with tool_calls
                "tool_calls": tool_calls,
            }
            if thinking_text:
                new_msg["thinking"] = thinking_text
            result.append(new_msg)
        elif thinking_blocks:
            # Only thinking, no tool calls
            thinking_text = _extract_thinking_text(thinking_blocks)
            new_msg = {
                "role": "assistant",
                "content": content,
            }
            if thinking_text:
                new_msg["thinking"] = thinking_text
            result.append(new_msg)
        else:
            # No thinking blocks, keep as is
            result.append(msg)

    return result


def save_trajectory(
    tokenizer: AutoTokenizer,
    context_messages: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]],
    turn_count: int,
    reward: float,
    terminated: bool,
    thinking_field: str = "reasoning_content",
):
    """
    Save decoded trajectory and context messages to files.

    Each turn is saved to a separate file with turn number.
    Files: test_live_trajectory_turn{N}.txt, test_live_context_turn{N}.json
    """
    # File paths with turn number
    trajectory_file = os.path.join(
        OUTPUT_DIR, f"test_live_trajectory_turn{turn_count:02d}.txt"
    )
    context_file = os.path.join(
        OUTPUT_DIR, f"test_live_context_turn{turn_count:02d}.json"
    )

    # Transform messages for tokenizer (gpt-oss needs special handling)
    transformed_messages = _transform_messages_for_tokenizer(
        context_messages, thinking_field
    )

    # Use apply_chat_template to test actual template behavior
    full_text = tokenizer.apply_chat_template(
        transformed_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tool_specs,
    )

    # Save decoded trajectory
    with open(trajectory_file, "w", encoding="utf-8") as f:
        f.write(f"# Turn: {turn_count}, Reward: {reward}, Terminated: {terminated}\n")
        f.write("=" * 60 + "\n\n")
        f.write(full_text)

    # Save context messages as JSON
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump(context_messages, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved turn {turn_count} to {trajectory_file}")


def get_env_settings():
    """Get settings from environment."""
    return {
        "domain": os.environ.get("TAU2_DOMAIN", "telecom"),
        "max_steps": int(os.environ.get("TAU2_MAX_STEPS", "30")),
        "max_turns": int(os.environ.get("TAU2_MAX_TURNS", "30")),
        "user_llm": os.environ.get("TAU2_USER_LLM", DEFAULT_LLM_USER),
        "user_temp": float(os.environ.get("TAU2_USER_TEMP", "0.0")),
        "agent_llm": os.environ.get("TAU2_USER_LLM", "claude-haiku-4-5"),
        "tool_parser": os.environ.get("TAU2_TOOL_PARSER", "qwen"),
    }


def call_agent_llm(
    messages: list[dict[str, Any]],
    tools: list,
    agent_llm: str,
) -> AgentResponse:
    """Call Claude API as agent."""
    # Convert dict messages to tau2 Message objects
    tau2_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        tool_calls_data = msg.get("tool_calls")

        if role == "system":
            # Skip empty system messages
            if content:
                tau2_messages.append(
                    SystemMessage(role="system", content=content)
                )
        elif role == "user":
            # Skip empty user messages
            if content:
                tau2_messages.append(UserMessage(role="user", content=content))
        elif role == "assistant":
            # Handle assistant messages with tool_calls
            thinking_blocks = msg.get("thinking_blocks")
            raw_data = (
                {"thinking_blocks": thinking_blocks}
                if thinking_blocks
                else None
            )

            if tool_calls_data:
                # Convert tool_calls to ToolCall objects
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=json.loads(
                            tc.get("function", {}).get("arguments", "{}")
                        ),
                        requestor="assistant",
                    )
                    for tc in tool_calls_data
                ]
                tau2_messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=content if content else None,
                        tool_calls=tool_calls,
                        raw_data=raw_data,
                    )
                )
            elif content or thinking_blocks:
                # Add if has content or thinking_blocks
                tau2_messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=content if content else None,
                        raw_data=raw_data,
                    )
                )
        elif role == "tool":
            # Tool results as ToolMessage
            tau2_messages.append(
                ToolMessage(
                    id=msg.get("tool_call_id", ""),
                    content=content,
                    role="tool",
                )
            )

    # Call LLM (thinking enabled - will return thinking_blocks)
    response = llm_generate(
        model=agent_llm,
        messages=tau2_messages,
        tools=tools,
        temperature=1.0,  # Required for thinking
    )

    # Convert tool_calls to dict format
    tool_calls = []
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append({"name": tc.name, "parameters": tc.arguments})

    content = response.content or ""
    raw_text = convert_to_raw_text(content, tool_calls)

    # Extract thinking from raw_data (supports both formats)
    thinking_blocks = None
    if response.raw_data:
        # Format 1: thinking_blocks (list of blocks)
        if "thinking_blocks" in response.raw_data:
            thinking_blocks = response.raw_data["thinking_blocks"]
        # Format 2: reasoning_content (string) - convert to blocks format
        elif "reasoning_content" in response.raw_data:
            reasoning_text = response.raw_data["reasoning_content"]
            if reasoning_text:
                thinking_blocks = [{"type": "thinking", "thinking": reasoning_text}]

        # Debug: log raw_data keys
        logger.debug(f"raw_data keys: {list(response.raw_data.keys())}")

    return AgentResponse(
        content=content,
        tool_calls=tool_calls,
        raw_text=raw_text,
        thinking_blocks=thinking_blocks,
    )


def run_episode(
    task_id: str,
    settings: dict,
    tokenizer: AutoTokenizer,
    tokenizer_name: str,
    max_steps: int = 100,
) -> MockSample:
    """
    Run a single episode using new SlimeGymEnv.

    Key difference from v1:
    - Uses ToolCallAction/TextAction instead of string parsing
    - obs["messages"] is already chat template compatible
    """
    sample = MockSample(prompt=task_id, metadata={"task_id": task_id})

    # Create environment (NEW: SlimeGymEnv instead of SlimeAgentGymEnv)
    env = SlimeGymEnv(
        domain=settings["domain"],
        task_id=task_id,
        max_steps=max_steps,
        user_llm=settings["user_llm"],
        user_llm_args={"temperature": settings["user_temp"]},
    )

    # Reset - returns structured observation
    obs, info = env.reset()

    policy = obs.get("policy", "")
    tool_specs = obs.get("tools", [])  # Dict format for chat template
    initial_messages = obs.get("messages", [])

    # Get tau2 Tool objects for llm_generate (requires openai_schema)
    # Include 'done' tool (same as GymAgent does)
    from tau2.environment.tool import as_tool

    def done() -> str:
        """Call this function when you have completed the task and the customer is satisfied."""
        return "###STOP###"

    tau2_tools = list(env._env.get_tools()) + [as_tool(done)]

    logger.info(f"Policy length: {len(policy)}")
    logger.info(f"Tools: {[t['function']['name'] for t in tool_specs]}")
    logger.info(f"Initial messages: {len(initial_messages)}")

    # Build context messages for agent (system + user's first request)
    # Agent's first greeting is NOT included - it's system-provided, not agent-generated
    # Agent's context starts with user's request
    context_messages: list[dict[str, Any]] = [
        {"role": "system", "content": policy},
    ]
    context_messages.extend(initial_messages)  # User's first response

    # Create tool adapter for parsing test
    tool_adapter = create_tool_adapter(tool_specs, settings["tool_parser"])

    # Get thinking field name based on tokenizer
    thinking_field = get_thinking_field_name(tokenizer_name)
    logger.info(f"Using thinking field: {thinking_field} (tokenizer: {tokenizer_name})")

    # Tracking
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    tool_call_idx = 0

    terminated = False
    step_count = 0
    turn_count = 0
    max_turns = settings.get("max_turns", 10)

    logger.info("=" * 60)
    logger.info("Starting episode")
    logger.info("=" * 60)

    import time

    try:
        while not terminated and turn_count < max_turns:
            turn_count += 1
            logger.info(f"\n--- Turn {turn_count} ---")

            # Rate limit delay (50k input tokens/min)
            # Add delay between turns to avoid hitting rate limit
            if turn_count > 1:
                time.sleep(10)  # 10 seconds between turns

            # Call agent LLM (use tau2_tools for openai_schema support)
            agent_response = call_agent_llm(
                messages=context_messages,
                tools=tau2_tools,
                agent_llm=settings["agent_llm"],
            )

            logger.info(
                f"Agent content: {agent_response.content[:100] if agent_response.content else '(none)'}..."
            )
            logger.info(f"Agent tool_calls: {len(agent_response.tool_calls)}")

            # Debug: Check thinking_blocks
            if agent_response.thinking_blocks:
                logger.info(f"Agent thinking_blocks: {len(agent_response.thinking_blocks)} blocks")
                for i, block in enumerate(agent_response.thinking_blocks):
                    logger.info(f"  Block {i} keys: {list(block.keys())}")
                    logger.info(f"  Block {i} raw: {str(block)[:200]}...")
            else:
                logger.warning("Agent thinking_blocks: None or empty!")

            # Test tool parsing (same as sglang flow)
            parsed = tool_adapter.parse(agent_response.raw_text)
            logger.info(
                f"Parsed tool calls: {len(parsed.calls) if parsed.success else 0}"
            )

            # Process response
            if agent_response.tool_calls:
                # Tool call action - NEW: use ToolCallAction
                actions = []
                for i, tc in enumerate(agent_response.tool_calls):
                    call_id = f"call_{tool_call_idx}"
                    tool_call_idx += 1
                    actions.append(
                        ToolCallAction(
                            name=tc["name"],
                            arguments=tc.get("parameters", {}),
                            id=call_id,
                        )
                    )

                # Add assistant message with tool calls to context
                # Keep original structure with thinking_blocks for Claude API
                # Transformation for tokenizer happens in save_trajectory
                assistant_msg = {
                    "role": "assistant",
                    "content": agent_response.content or "",
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
                # Preserve thinking_blocks for Claude extended thinking
                if agent_response.thinking_blocks:
                    assistant_msg["thinking_blocks"] = agent_response.thinking_blocks
                    logger.info(f"  Added thinking_blocks: {len(agent_response.thinking_blocks)} blocks")

                context_messages.append(assistant_msg)
                logger.debug(f"  assistant_msg keys: {list(assistant_msg.keys())}")

                # Save trajectory RIGHT AFTER assistant message (before adding responses)
                # This ensures the last message is assistant, so chat_template shows thinking
                save_trajectory(
                    tokenizer=tokenizer,
                    context_messages=context_messages,
                    tool_specs=tool_specs,
                    turn_count=turn_count,
                    reward=0.0,  # reward not yet known
                    terminated=False,
                    thinking_field=thinking_field,
                )

                # Execute tool calls
                obs, reward, terminated, truncated, info = env.step(actions)
                step_count += 1

                # Add tool responses to context (from obs["messages"])
                for msg in obs["messages"]:
                    if msg.get("role") == "tool":
                        context_messages.append(msg)
                        logger.info(
                            f"  Tool response ({msg.get('name', 'tool')}): {msg.get('content', '')[:50]}..."
                        )

                # Calculate tokens for assistant + tool responses
                token_ids, mask = get_token_delta(
                    tokenizer, context_messages, tool_specs
                )
                response_token_ids.extend(token_ids)
                loss_masks.extend([1] * len(token_ids))  # Assistant tokens

            else:
                # Text action - NEW: use TextAction
                action = TextAction(content=agent_response.content or "")

                # Add assistant message to context (only if has content or thinking_blocks)
                assistant_content = agent_response.content or ""
                if assistant_content or agent_response.thinking_blocks:
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_content,
                    }
                    # Preserve thinking_blocks for Claude extended thinking
                    if agent_response.thinking_blocks:
                        assistant_msg["thinking_blocks"] = agent_response.thinking_blocks
                        logger.info(f"  Added thinking_blocks: {len(agent_response.thinking_blocks)} blocks")
                    context_messages.append(assistant_msg)
                    logger.debug(f"  assistant_msg keys: {list(assistant_msg.keys())}")

                    # Save trajectory RIGHT AFTER assistant message (before adding responses)
                    save_trajectory(
                        tokenizer=tokenizer,
                        context_messages=context_messages,
                        tool_specs=tool_specs,
                        turn_count=turn_count,
                        reward=0.0,
                        terminated=False,
                        thinking_field=thinking_field,
                    )

                # Execute
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                # Add user response to context (from obs["messages"])
                for msg in obs["messages"]:
                    if msg.get("role") == "user":
                        # Skip empty user messages
                        user_content = msg.get("content", "")
                        if not user_content:
                            logger.warning(
                                f"  Skipping empty user message: {msg}"
                            )
                            continue
                        context_messages.append(msg)
                        logger.info(f"  User response: {user_content[:100]}...")

                # Calculate tokens
                token_ids, mask = get_token_delta(
                    tokenizer, context_messages, tool_specs
                )
                response_token_ids.extend(token_ids)
                loss_masks.extend([1] * len(token_ids))  # Assistant tokens

            logger.info(
                f"  Step {step_count}: terminated={terminated}, reward={reward}"
            )

            if terminated:
                sample.reward = reward

    except Exception as e:
        # Rate limit or other error - save partial results
        logger.error(f"Episode interrupted at turn {turn_count}: {e}")
        sample.metadata["error"] = str(e)
        sample.metadata["interrupted_at_turn"] = turn_count

        # Save trajectory on error
        save_trajectory(
            tokenizer=tokenizer,
            context_messages=context_messages,
            tool_specs=tool_specs,
            turn_count=turn_count,
            reward=sample.reward,
            terminated=False,
            thinking_field=thinking_field,
        )

    # Build response from assistant messages
    response_parts = []
    for msg in context_messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                response_parts.append(content)

    # Build final sample
    sample.tokens = response_token_ids
    sample.loss_mask = loss_masks
    sample.response = "\n\n".join(response_parts)
    sample.response_length = len(response_token_ids)
    sample.metadata["steps"] = step_count
    sample.metadata["num_turns"] = turn_count

    # Extract 3 state views for debugging
    sample.metadata["states"] = _extract_state_views(env)

    if sample.metadata.get("error"):
        sample.status = MockSample.Status.ABORTED
    elif terminated:
        sample.status = MockSample.Status.COMPLETED
    else:
        sample.status = MockSample.Status.TRUNCATED

    return sample


def _extract_state_views(env) -> dict:
    """
    Extract 3 state views from environment for debugging.

    Returns:
        {
            "main_state": [...],    # All messages (ground truth)
            "agent_state": [...],   # Agent view (user tool removed)
            "user_state": [...],    # User view (agent tool removed + flip_roles)
        }

    Information symmetry:
    - Agent sees: agent's (content, tool_calls, ToolMessages) + user's content only
    - User sees: user's (content, tool_calls, ToolMessages) + agent's content only
      BUT with flip_roles() applied (User→Assistant, Agent→User)
    """
    from tau2.data_model.message import (
        AssistantMessage,
        ToolMessage,
        UserMessage,
    )

    def msg_to_dict(msg) -> dict:
        """Convert tau2 Message to dict."""
        result = {"role": msg.role}
        if hasattr(msg, "content") and msg.content:
            result["content"] = msg.content
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        if hasattr(msg, "id") and msg.id:
            result["tool_call_id"] = msg.id
        if hasattr(msg, "requestor") and msg.requestor:
            result["requestor"] = msg.requestor
        return result

    def flip_role(role: str) -> str:
        """Flip role for user simulator perspective."""
        if role == "user":
            return "assistant"
        elif role == "assistant":
            return "user"
        return role  # tool stays the same

    # 1. Main State: all messages
    main_state = []
    if env._state and env._state.messages:
        for msg in env._state.messages:
            main_state.append(msg_to_dict(msg))

    # 2. Agent State: env.get_context_messages() - user tool removed, includes system
    agent_state = env.get_context_messages(include_system=True)

    # 3. User State: 실제 user simulator에게 전달된 messages
    # env._last_user_state_messages에 저장됨 (flip_roles 적용 전)
    # flip_roles 후의 상태를 보여줌:
    # - Original UserMessage → role="assistant" (user sent it)
    # - Original AssistantMessage (text only) → role="user" (agent sent it)
    # - Original ToolMessage (requestor=user) → role="tool" (user's tool result)
    user_state_before_flip = []
    user_state_after_flip = []

    # Add user system message first (what user simulator sees)
    user_system_prompt = env.get_user_system_prompt()
    if user_system_prompt:
        user_state_after_flip.append(
            {"role": "system", "content": user_system_prompt}
        )

    if env._last_user_state_messages:
        # Before flip (raw messages passed to UserState)
        for msg in env._last_user_state_messages:
            user_state_before_flip.append(msg_to_dict(msg))

        # After flip (what user simulator actually sees)
        for msg in env._last_user_state_messages:
            if isinstance(msg, AssistantMessage):
                # Agent's text content → flip to "user" role
                d = msg_to_dict(msg)
                d["role"] = flip_role(d["role"])  # assistant → user
                user_state_after_flip.append(d)
            elif isinstance(msg, ToolMessage):
                # User's tool response - keep as "tool"
                user_state_after_flip.append(msg_to_dict(msg))
            else:
                # UserMessage → flip to "assistant" role
                d = msg_to_dict(msg)
                d["role"] = flip_role(d["role"])  # user → assistant
                user_state_after_flip.append(d)

    return {
        "main_state": main_state,
        "agent_state": agent_state,
        "user_state_before_flip": user_state_before_flip,
        "user_state_after_flip": user_state_after_flip,
    }


def print_context_messages(context_messages: list[dict[str, Any]]):
    """Print context messages for debugging."""
    print("\n" + "=" * 60)
    print("Context Messages")
    print("=" * 60)
    for i, msg in enumerate(context_messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if len(content) > 100:
            content = content[:100] + "..."

        tool_calls = msg.get("tool_calls", [])
        tool_info = f" [+{len(tool_calls)} tool calls]" if tool_calls else ""

        tool_call_id = msg.get("tool_call_id", "")
        tool_id_info = f" (id={tool_call_id[:30]}...)" if tool_call_id else ""

        print(f"[{i}] {role}{tool_info}{tool_id_info}: {content}")


def main():
    """Run live episode test."""
    settings = get_env_settings()

    logger.info("=" * 60)
    logger.info("Live Episode Test V2 (SlimeGymEnv)")
    logger.info("=" * 60)
    logger.info(f"Settings: {json.dumps(settings, indent=2)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Get a task
    from tau2.registry import registry

    tasks = registry.get_tasks_loader(settings["domain"])()
    task = tasks[0]
    logger.info(f"Task: {task.id}")

    # Run episode (handles errors internally, returns partial sample on error)
    sample = run_episode(
        task_id=task.id,
        settings=settings,
        tokenizer=tokenizer,
        tokenizer_name=TOKENIZER_NAME,
        max_steps=settings["max_steps"],
    )

    # Print results
    print("\n" + "=" * 60)
    print("Episode Results")
    print("=" * 60)
    print(f"Status: {sample.status}")
    print(f"Reward: {sample.reward}")
    print(f"Response length: {sample.response_length}")
    print(f"Token count: {len(sample.tokens)}")
    print(f"Loss mask sum: {sum(sample.loss_mask)}")
    if sample.metadata.get("error"):
        print(f"Error: {sample.metadata['error'][:200]}...")
        print(
            f"Interrupted at turn: {sample.metadata.get('interrupted_at_turn', '?')}"
        )

    # Save sample to JSON (v1 format, always save even on error)
    output_path = os.path.join(
        os.path.dirname(__file__), "test_sample_output_v2.json"
    )
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
            "domain": settings.get("domain"),
            "steps": sample.metadata.get("steps", 0),
            "num_turns": sample.metadata.get("num_turns", 0),
            "error": sample.metadata.get("error"),
            "interrupted_at_turn": sample.metadata.get("interrupted_at_turn"),
        },
        # 3 state views for debugging multi-turn history
        "states": sample.metadata.get("states", {}),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    print(f"\nSample saved to: {output_path}")

    if sample.status == MockSample.Status.ABORTED:
        print("\n" + "=" * 60)
        print("PARTIAL: Episode interrupted but data saved!")
        print("=" * 60)
    elif sample.status == MockSample.Status.COMPLETED:
        print("\n" + "=" * 60)
        print("SUCCESS: Episode completed with new SlimeGymEnv!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TRUNCATED: Episode reached max turns")
        print("=" * 60)


if __name__ == "__main__":
    main()
