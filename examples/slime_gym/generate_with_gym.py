"""
SLIME generate/reward interface for slime_gym environments.

Supports per-sample environment and tool customization:
- sample.metadata["env_name"]: Select environment type per sample
- sample.metadata["enabled_tools"]: Filter available tools per sample

Usage:
    python -m slime.train \
        --custom-generate-function-path examples.slime_gym.generate_with_gym.generate \
        --custom-rm-path examples.slime_gym.generate_with_gym.reward_func \
        ...
"""

import os
import re

from jinja2 import Template

# SLIME imports
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Local imports
from .base import BaseEnvironment, parse_tool_calls
from .dynamic_env import DynamicServiceEnvironment
from .retail_env import RetailServiceEnvironment

# ==================== Configuration ====================


CONFIGS = {
    "max_turns": int(os.environ.get("SLIME_GYM_MAX_TURNS", 10)),
    "max_turns_buffer": int(os.environ.get("SLIME_GYM_MAX_TURNS_BUFFER", 0)),
    "dynamic_max_turns": os.environ.get("SLIME_GYM_DYNAMIC_MAX_TURNS", "true").lower() == "true",
}

# Environment registry - add your environments here
ENVIRONMENTS: dict[str, type[BaseEnvironment]] = {
    "retail_service": RetailServiceEnvironment,
    "dynamic_service": DynamicServiceEnvironment,  # Supports dynamic tool loading
    # Add more environments:
    # "banking_service": BankingServiceEnvironment,
    # "tech_support": TechSupportEnvironment,
}


def get_environment(env_name: str) -> BaseEnvironment:
    """
    Get a new environment instance by name.

    Creates a fresh instance each time to avoid state conflicts
    when processing multiple samples concurrently.

    Args:
        env_name: Name of the environment (must be in ENVIRONMENTS registry)

    Returns:
        BaseEnvironment instance

    Raises:
        ValueError: If env_name is not in ENVIRONMENTS registry
    """
    if env_name not in ENVIRONMENTS:
        available = list(ENVIRONMENTS.keys())
        raise ValueError(
            f"Unknown environment: '{env_name}'. "
            f"Available environments: {available}. "
            f"Register new environments in ENVIRONMENTS dict."
        )

    return ENVIRONMENTS[env_name]()


def resolve_env_name(sample: Sample, args) -> str:
    """
    Resolve environment name for a sample.

    Priority:
    1. sample.metadata["env_name"] - per-sample override
    2. args.env_name - command-line default
    3. "retail_service" - fallback default

    Returns:
        Environment name string
    """
    # Per-sample environment selection (highest priority)
    if sample.metadata and "env_name" in sample.metadata:
        return sample.metadata["env_name"]

    # Command-line argument (default for all samples)
    if hasattr(args, "env_name") and args.env_name:
        return args.env_name

    # Fallback default
    return "retail_service"


# ==================== Prompt Template ====================
# NOTE: This template uses ChatML format (<|im_start|>, <|im_end|>).
# Compatible models: Qwen, Yi, and other ChatML-based models.
# For other models (Llama, Mistral, etc.), override format_prompt() or use
# args.apply_chat_template with the model's tokenizer.

TOOL_TEMPLATE = """<|im_start|>system
{%- if system_prompt %}
{{ system_prompt }}
{%- else %}
You are a helpful customer service agent.
{%- endif %}
{%- if tools %}

# Available Tools

You can use the following tools to help customers:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

To use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

After receiving tool results, continue assisting the customer.
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message.role == 'user' %}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == 'assistant' %}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{%- elif message.role == 'tool' %}
<|im_start|>tool
{{ message.content }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_prompt(messages: list[dict], tools: list[dict], system_prompt: str = None) -> str:
    """Format messages with tools using Jinja2 template"""
    template = Template(TOOL_TEMPLATE)

    # Extract system prompt from messages if present
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", system_prompt)
        messages = messages[1:]

    return template.render(system_prompt=system_prompt, messages=messages, tools=tools)


def postprocess_response(text: str) -> str:
    """Ensure response ends at complete tag"""
    if "<tool_call>" in text:
        pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            return text[: matches[-1].end()]
    return text


# ==================== Main Generate Function ====================


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Custom generation function with tool calling support.

    Implements an agentic loop:
    1. Format prompt with tools
    2. Call model
    3. Parse tool calls
    4. Execute tools
    5. Append results and repeat

    SLIME standard: all output stored in sample.response
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout not supported"

    # Initialize
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get environment - supports per-sample selection via metadata["env_name"]
    env_name = resolve_env_name(sample, args)
    env = get_environment(env_name)

    # Initialize environment for this sample (sets state and enabled_tools)
    env.seed(sample.metadata)

    # Get tools (respects enabled_tools filtering from seed())
    tools = env.get_tools()

    # Build initial prompt
    initial_messages = (
        sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": sample.prompt}]
    )
    prompt_text = format_prompt(initial_messages, tools)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Initialize sample fields
    sample.prompt = prompt_text  # Update to formatted text (Search-R1 standard)
    sample.tokens = prompt_token_ids.copy()
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []

    # Store environment for reward_func
    sample.metadata["_env_name"] = env_name

    # Determine max_turns (priority: per-sample > dynamic > global default)
    # 1. Per-sample override via metadata["max_turns"]
    if sample.metadata and "max_turns" in sample.metadata:
        max_turns = sample.metadata["max_turns"]
    # 2. Dynamic mode: len(expected_actions) + buffer
    elif CONFIGS["dynamic_max_turns"]:
        expected_actions = sample.metadata.get("expected_actions", [])
        if expected_actions:
            max_turns = len(expected_actions) + CONFIGS["max_turns_buffer"]
        else:
            max_turns = CONFIGS["max_turns"]
    # 3. Fixed mode: use global default
    else:
        max_turns = CONFIGS["max_turns"]

    for _turn in range(max_turns):
        # Check context length
        max_ctx = getattr(args, "rollout_max_context_len", None)
        if max_ctx and len(sample.tokens) >= max_ctx:
            sample.status = Sample.Status.TRUNCATED
            break

        # Call model
        payload = {
            "input_ids": sample.tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            break

        # Extract response
        if "output_token_logprobs" in output["meta_info"]:
            cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_text = state.tokenizer.decode(cur_token_ids)
            cur_logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_text = output["text"]
            cur_text = postprocess_response(cur_text)
            cur_token_ids = state.tokenizer(cur_text, add_special_tokens=False)["input_ids"]
            cur_logprobs = [0.0] * len(cur_token_ids)

        # Record model output (trainable: loss_mask = 1)
        sample.tokens.extend(cur_token_ids)
        sample.response += cur_text
        sample.response_length += len(cur_token_ids)
        sample.loss_mask.extend([1] * len(cur_token_ids))
        sample.rollout_log_probs.extend(cur_logprobs)

        # Verify alignment after model output
        assert len(sample.loss_mask) == len(sample.rollout_log_probs), (
            f"Token/logp length mismatch after model output: {len(sample.loss_mask)} loss_mask "
            f"vs {len(sample.rollout_log_probs)} log_probs"
        )

        # Check for length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        # Parse tool calls
        tool_calls = parse_tool_calls(cur_text)

        # No tool calls -> done
        if not tool_calls:
            sample.status = Sample.Status.COMPLETED
            break

        # Execute tools
        for tc in tool_calls:
            result = await env.execute_tool(tc.name, tc.arguments)

            # Format tool result with tool name for clarity
            tool_output = f'\n<tool_result name="{tc.name}">\n{result.output}\n</tool_result>\n'
            tool_token_ids = state.tokenizer(tool_output, add_special_tokens=False)["input_ids"]

            # Record tool output (not trainable: loss_mask = 0)
            sample.tokens.extend(tool_token_ids)
            sample.response += tool_output
            sample.response_length += len(tool_token_ids)
            sample.loss_mask.extend([0] * len(tool_token_ids))
            sample.rollout_log_probs.extend([0.0] * len(tool_token_ids))

            # Verify alignment (same as Search-R1)
            assert len(sample.loss_mask) == len(sample.rollout_log_probs), (
                f"Token/logp length mismatch: {len(sample.loss_mask)} loss_mask "
                f"vs {len(sample.rollout_log_probs)} log_probs"
            )

    # If loop completed without break, mark as truncated (max_turns reached)
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.TRUNCATED

    # Calculate reward while env state is still valid
    # Skip verification for aborted samples (incomplete response)
    if sample.status == Sample.Status.ABORTED:
        reward = 0.0
    else:
        reward = await env.verify(sample)

    tool_calls = parse_tool_calls(sample.response)
    sample.reward = {
        "score": reward,
        "num_turns": len(tool_calls),
    }

    # Final alignment verification
    assert sample.response_length == len(sample.loss_mask), (
        f"response_length mismatch: {sample.response_length} " f"vs {len(sample.loss_mask)} loss_mask entries"
    )
    assert sample.response_length == len(sample.rollout_log_probs), (
        f"response_length mismatch: {sample.response_length} " f"vs {len(sample.rollout_log_probs)} log_probs entries"
    )

    return sample


# ==================== Gym-style Generate Function ====================


async def generate_with_episode(args, sample: Sample, sampling_params) -> Sample:
    """
    Alternative generate function using run_episode() for cleaner abstraction.

    Uses BaseEnvironment.run_episode() which handles the agentic loop internally.
    This is useful for:
    - Simpler code structure
    - Testing environments independently
    - Non-SLIME rollout collection

    The main generate() function duplicates some logic for performance,
    but this function demonstrates the Gym-style interface.
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout not supported"

    # Initialize
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get environment
    env_name = resolve_env_name(sample, args)
    env = get_environment(env_name)
    env.seed(sample.metadata)

    # Get tools and format prompt
    tools = env.get_tools()
    initial_messages = (
        sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": sample.prompt}]
    )
    prompt_text = format_prompt(initial_messages, tools)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Create model_fn wrapper for run_episode()
    accumulated_tokens = prompt_token_ids.copy()

    async def model_fn(prev_response: str) -> tuple[str, list[int], list[float]]:
        """Model function compatible with run_episode() interface."""
        nonlocal accumulated_tokens

        payload = {
            "input_ids": accumulated_tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            return "", [], []

        # Extract response
        if "output_token_logprobs" in output["meta_info"]:
            cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_text = state.tokenizer.decode(cur_token_ids)
            cur_logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_text = output["text"]
            cur_text = postprocess_response(cur_text)
            cur_token_ids = state.tokenizer(cur_text, add_special_tokens=False)["input_ids"]
            cur_logprobs = [0.0] * len(cur_token_ids)

        # Update accumulated tokens for next call
        accumulated_tokens.extend(cur_token_ids)

        return cur_text, cur_token_ids, cur_logprobs

    # Tokenizer function for tool outputs
    def tokenizer_fn(text: str) -> list[int]:
        tokens = state.tokenizer(text, add_special_tokens=False)["input_ids"]
        # Also update accumulated_tokens for tool outputs
        accumulated_tokens.extend(tokens)
        return tokens

    # Determine max_turns
    if sample.metadata and "max_turns" in sample.metadata:
        max_turns = sample.metadata["max_turns"]
    elif CONFIGS["dynamic_max_turns"]:
        expected_actions = sample.metadata.get("expected_actions", [])
        if expected_actions:
            max_turns = len(expected_actions) + CONFIGS["max_turns_buffer"]
        else:
            max_turns = CONFIGS["max_turns"]
    else:
        max_turns = CONFIGS["max_turns"]

    # Run episode using Gym-style interface
    result = await env.run_episode(
        model_fn=model_fn,
        initial_prompt=prompt_text,
        max_turns=max_turns,
        tokenizer=tokenizer_fn,
    )

    # Convert EpisodeResult to Sample
    sample.prompt = prompt_text
    sample.tokens = prompt_token_ids + result.tokens
    sample.response = result.response
    sample.response_length = len(result.tokens)
    sample.loss_mask = result.loss_mask
    sample.rollout_log_probs = result.rollout_log_probs
    sample.status = result.status
    sample.reward = {
        "score": result.reward,
        "num_turns": result.num_turns,
    }
    sample.metadata["_env_name"] = env_name

    return sample


# ==================== Reward Function ====================


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    """
    Reward function: returns pre-calculated reward from generate().

    Reward is calculated in generate() while the environment state is valid.
    """
    # Reward already calculated in generate()
    if sample.reward is not None:
        return sample.reward

    # Fallback: recalculate if reward not set
    # WARNING: This fallback creates a new environment and only calls seed(),
    # so state-based verification (e.g., refund_processed) will NOT work correctly.
    # This should only be used for response-parsing-only verification.
    import logging

    logging.warning(
        "reward_func fallback triggered: reward not pre-calculated in generate(). "
        "State-based verification may fail."
    )

    # Use stored env_name from generate(), or resolve from metadata/args
    env_name = sample.metadata.get("_env_name") or resolve_env_name(sample, args)
    env = get_environment(env_name)
    env.seed(sample.metadata)

    reward = await env.verify(sample)
    tool_calls = parse_tool_calls(sample.response)

    return {
        "score": reward,
        "num_turns": len(tool_calls),
    }
