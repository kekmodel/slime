"""
SLIME generate/reward interface for slime_gym environments.

Usage:
    python -m slime.train \
        --custom-generate-function-path examples.slime_gym.generate_with_gym.generate \
        --custom-rm-path examples.slime_gym.generate_with_gym.reward_func \
        ...
"""

from dataclasses import dataclass
from typing import Any, cast

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .base import BaseEnvironment, parse_tool_calls
from .config import resolve_max_turns
from .env_registry import EnvironmentRegistry, resolve_env_name
from .formatters import ChatMLFormatter, get_formatter
from .types import append_to_sample, init_sample_for_generation

# Type alias for post() response
PostResponse = dict[str, Any]


@dataclass
class GenerateContext:
    """Context for generation, encapsulating all initialized state."""

    env: BaseEnvironment
    formatter: ChatMLFormatter
    url: str
    max_turns: int
    tokenizer: Any


def setup_generate(args, sample: Sample) -> GenerateContext:
    """
    Initialize all components needed for generation.

    Sets up:
    - Environment (resolved from sample metadata or args)
    - Prompt formatter (based on model type)
    - Sample fields (prompt, tokens, response, etc.)
    - Max turns (from metadata or config)

    Returns:
        GenerateContext with all initialized components
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get environment
    env_name = resolve_env_name(sample, args)
    env = EnvironmentRegistry.get(env_name)
    env.setup(sample.metadata)

    # Get tools and format prompt
    tools = env.get_tools()
    formatter = get_formatter(getattr(args, "model_type", "chatml"))

    initial_messages = sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": sample.prompt}]
    prompt_text = formatter.format(initial_messages, tools)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    init_sample_for_generation(sample, prompt_text, prompt_token_ids)
    sample.metadata["_env_name"] = env_name

    max_turns = resolve_max_turns(sample.metadata)

    return GenerateContext(
        env=env,
        formatter=formatter,
        url=url,
        max_turns=max_turns,
        tokenizer=state.tokenizer,
    )


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Custom generation function with tool calling support.

    Implements an agentic loop:
    1. Format prompt with tools
    2. Call model
    3. Parse tool calls
    4. Execute tools
    5. Append results and repeat
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout not supported"

    ctx = setup_generate(args, sample)

    for _ in range(ctx.max_turns):
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

        output: PostResponse = cast(PostResponse, await post(ctx.url, payload))

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            break

        # Extract response
        if "output_token_logprobs" in output["meta_info"]:
            cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_text = ctx.tokenizer.decode(cur_token_ids)
            cur_logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_text = output["text"]
            cur_text = ctx.formatter.postprocess_response(cur_text)
            cur_token_ids = ctx.tokenizer(cur_text, add_special_tokens=False)["input_ids"]
            cur_logprobs = [0.0] * len(cur_token_ids)

        # Record model output (trainable)
        append_to_sample(sample, cur_text, cur_token_ids, cur_logprobs, trainable=True)

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
            result = await ctx.env.execute_tool(tc.name, tc.arguments)
            tool_output = ctx.formatter.format_tool_result(tc.name, result.output)
            tool_token_ids = ctx.tokenizer(tool_output, add_special_tokens=False)["input_ids"]

            # Record tool output (not trainable)
            append_to_sample(sample, tool_output, tool_token_ids, [0.0] * len(tool_token_ids), trainable=False)

    # Mark as truncated if loop completed without break
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.TRUNCATED

    # Calculate reward while env state is still valid
    if sample.status == Sample.Status.ABORTED:
        reward = 0.0
    else:
        reward = ctx.env.verify()

    tool_calls = parse_tool_calls(sample.response)
    sample.reward = {
        "score": reward,
        "num_turns": len(tool_calls),
    }

    # Final alignment verification
    if sample.loss_mask is not None:
        assert sample.response_length == len(sample.loss_mask), f"response_length mismatch: {sample.response_length} vs {len(sample.loss_mask)}"
    if sample.rollout_log_probs is not None:
        assert sample.response_length == len(sample.rollout_log_probs), f"response_length mismatch: {sample.response_length} vs {len(sample.rollout_log_probs)}"

    return sample


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    """
    Reward function: returns pre-calculated reward from generate().

    Reward is calculated in generate() while the environment state is valid.
    """
    if sample.reward is None:
        raise RuntimeError("Reward not pre-calculated in generate(). Ensure generate() completed successfully before calling reward_func().")
    return cast(dict[str, Any], sample.reward)
