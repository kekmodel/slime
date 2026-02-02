"""
Multi-hop tool calling generate function for slime.

Uses SGLang's /generate endpoint with return_logprob=True.
Token IDs from logprobs are the source of truth - text is derived by decoding.

Data flow:
    token_ids (from logprobs) → decode → text → parse tool calls
    → execute tools → format observation → encode → observation_token_ids

Usage:
    python train.py \\
        --custom-generate-function-path "examples.tool_calling.generate.generate" \\
        --sglang-tool-call-parser qwen25 \\
        ...
"""

import json
import logging
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .tools import (
    ToolCall,
    ToolRegistry,
    create_default_registry,
    format_observation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ToolCallingConfig:
    max_hops: int = 16
    max_tool_calls_per_hop: int = 4
    tool_parser: str = "qwen25"
    stop_on_no_tool_call: bool = True
    stop_on_error: bool = False


# ============================================================================
# Model Profiles
# ============================================================================
# Centralized model-specific behavior configuration.
# Each profile defines:
# - call_id_format: How to generate call_id for tool responses
# - requires_matching_call_id: Whether model requires matching call_id from tool call
# - formatter_needs: Which kwargs the formatter requires


@dataclass
class ModelProfile:
    """Model-specific configuration for tool calling."""

    call_id_format: str = "call_{index}"  # Format string with {name} and {index} vars
    requires_matching_call_id: bool = False  # Whether call_id must match model-generated ID
    formatter_needs: tuple[str, ...] = ()  # Required kwargs for formatter (e.g., "tool_name", "tool_call_id")


MODEL_PROFILES: dict[str, ModelProfile] = {
    # Qwen family - simple format, no special requirements
    "qwen": ModelProfile(),
    "qwen25": ModelProfile(),
    "qwen3_coder": ModelProfile(),
    "mimo": ModelProfile(),
    "hermes": ModelProfile(),
    # GLM family - simple format
    "glm": ModelProfile(),
    "glm45": ModelProfile(),
    "glm47": ModelProfile(),
    # DeepSeek family - simple format
    "deepseekv3": ModelProfile(),
    "deepseekv31": ModelProfile(),
    "deepseekv32": ModelProfile(),
    # Llama family - simple format
    "llama3": ModelProfile(),
    # MiniMax - simple format
    "minimax-m2": ModelProfile(),
    # GPT-OSS - requires tool_name
    "gpt-oss": ModelProfile(
        call_id_format="functions.{name}",
        formatter_needs=("tool_name",),
    ),
    # Kimi K2 - requires tool_call_id matching
    "kimi_k2": ModelProfile(
        call_id_format="functions.{name}:{index}",
        requires_matching_call_id=True,
        formatter_needs=("tool_call_id",),
    ),
    # Mistral - requires tool_call_id matching (model generates UUID)
    "mistral": ModelProfile(
        requires_matching_call_id=True,
        formatter_needs=("tool_call_id",),
    ),
}


def get_model_profile(parser_name: str) -> ModelProfile:
    """Get model profile for the given parser name."""
    if parser_name not in MODEL_PROFILES:
        logger.warning(f"Unknown parser '{parser_name}', using default profile")
        return ModelProfile()
    return MODEL_PROFILES[parser_name]


# ============================================================================
# Parsing Utilities
# ============================================================================


def parse_tool_calls(
    text: str,
    tools: List[Dict[str, Any]],
    parser_name: str = "qwen25",
) -> Tuple[str, List[ToolCall]]:
    """
    Parse tool calls from model output using SGLang's FunctionCallParser.

    Args:
        text: Model output text
        tools: List of tools in OpenAI format
        parser_name: Name of the parser (qwen25, llama3, deepseekv3, etc.)

    Returns:
        (normal_text, list_of_tool_calls)
    """
    if not tools:
        return text, []

    # Convert to SGLang Tool format
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]

    try:
        parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser_name)
        normal_text, calls = parser.parse_non_stream(text)

        # Convert to our ToolCall format
        # ToolCallItem has: tool_index: int, name: Optional[str], parameters: str (JSON string)
        # Generate call_id based on parser format to match model's expected response format
        tool_calls = [
            ToolCall(
                name=call.name or "",
                arguments=_parse_arguments(call.parameters),
                call_id=_generate_call_id(call, parser_name),
                raw=text,
            )
            for call in calls
        ]

        return normal_text, tool_calls

    except Exception as e:
        logger.warning(f"Failed to parse tool calls: {e}")
        return text, []


def _parse_arguments(args: Union[str, dict]) -> dict:
    """Parse arguments from string to dict if needed."""
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return {"raw": args}


def _generate_call_id(call, parser_name: str) -> str:
    """Generate call_id based on model profile to match model's expected response format."""
    profile = get_model_profile(parser_name)
    name = call.name or "unknown"
    index = getattr(call, "tool_index", 0)

    if profile.requires_matching_call_id and not getattr(call, "call_id", None):
        logger.warning(f"Parser '{parser_name}' requires matching call_id from model, but none provided. Using fallback format. This may cause tool response mismatches.")

    return profile.call_id_format.format(name=name, index=index)


# ============================================================================
# Response Processing
# ============================================================================


def extract_tokens_from_logprobs(output: Dict) -> Tuple[List[int], List[float]]:
    """
    Extract token IDs and log probs from /generate response.

    The logprob entries are: (logprob, token_id, [optional: token_text])
    We only extract token_ids and log_probs - text is derived by decoding token_ids.

    Returns:
        (token_ids, log_probs)

    Raises:
        ValueError: If no logprobs data is available
    """
    meta_info = output.get("meta_info", {})
    output_token_logprobs = meta_info.get("output_token_logprobs", [])

    if not output_token_logprobs:
        raise ValueError("No output_token_logprobs in response - cannot produce valid RL training data")

    valid_entries = [entry for entry in output_token_logprobs if len(entry) >= 2]
    token_ids = [entry[1] for entry in valid_entries]
    log_probs = [entry[0] for entry in valid_entries]

    return token_ids, log_probs


# ============================================================================
# Main Generate Function
# ============================================================================


async def generate(
    args,
    sample: Sample,
    sampling_params: dict,
    registry: Optional[ToolRegistry] = None,
    config: Optional[ToolCallingConfig] = None,
) -> Sample:
    """
    Multi-hop tool calling generate function.

    This function:
    1. Calls /generate with return_text_in_logprobs=True
    2. Parses tool calls using FunctionCallParser
    3. Executes tools and appends observations
    4. Repeats until no tool calls or max_hops reached
    5. Returns Sample with proper tokens, loss_mask, and log_probs for RL

    Args:
        args: Rollout arguments (must have sglang_router_ip, sglang_router_port)
        sample: Input sample with prompt
        sampling_params: Sampling parameters for generation
        registry: Tool registry (uses default if None)
        config: Tool calling configuration (uses default if None)

    Returns:
        Updated Sample with response, tokens, loss_mask, rollout_log_probs
    """
    if getattr(args, "partial_rollout", False):
        logger.error("Partial rollout is not supported for tool calling")
        sample.status = Sample.Status.FAILED
        sample.metadata["tool_calling_error"] = "Partial rollout is not supported for tool calling"
        return sample

    # Setup
    config = config or ToolCallingConfig()
    registry = registry or _get_registry(args, sample)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get tools in OpenAI format for parser
    tools = registry.to_openai_format()

    # Get parser name from args or config
    parser_name = getattr(args, "tool_parser", None) or getattr(args, "sglang_tool_call_parser", None) or config.tool_parser

    # Initialize tracking
    prompt_text = sample.prompt
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Accumulated response data
    all_response_text = ""
    all_response_token_ids: List[int] = []
    all_log_probs: List[float] = []
    all_loss_mask: List[int] = []

    current_input_ids = prompt_token_ids.copy()

    # Multi-hop loop
    hop = 0
    for hop in range(config.max_hops):
        # Check context length
        if args.rollout_max_context_len is not None:
            if len(current_input_ids) >= args.rollout_max_context_len:
                sample.status = Sample.Status.TRUNCATED
                break

        # Build payload
        payload = {
            "input_ids": current_input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            # Note: We don't use return_text_in_logprobs - we decode token_ids ourselves
        }

        # Call /generate
        raw_output = await post(url, payload)
        output: dict = raw_output if isinstance(raw_output, dict) else {}

        # Check for abort
        finish_reason = output.get("meta_info", {}).get("finish_reason", {})
        if finish_reason.get("type") == "abort":
            sample.status = Sample.Status.ABORTED
            break

        # Extract token_ids and log_probs (source of truth for RL training)
        try:
            response_token_ids, response_log_probs = extract_tokens_from_logprobs(output)
        except ValueError as e:
            logger.error(f"Hop {hop}: {e}")
            sample.status = Sample.Status.ABORTED
            break

        # CRITICAL: Verify per-hop data alignment
        if len(response_token_ids) != len(response_log_probs):
            logger.error(f"Hop {hop}: token_ids/log_probs mismatch: {len(response_token_ids)} vs {len(response_log_probs)}")
            sample.status = Sample.Status.FAILED
            sample.metadata["tool_calling_error"] = f"Hop {hop}: token_ids/log_probs mismatch"
            break

        # Decode token_ids to get text for tool parsing
        # This ensures text and token_ids are perfectly aligned
        response_text = state.tokenizer.decode(response_token_ids, skip_special_tokens=False)

        # Parse tool calls from decoded text
        _, tool_calls = parse_tool_calls(response_text, tools, parser_name)

        # Accumulate generation (loss_mask = 1 for generated tokens)
        all_response_text += response_text
        all_response_token_ids.extend(response_token_ids)
        all_log_probs.extend(response_log_probs)
        all_loss_mask.extend([1] * len(response_token_ids))

        # Update input for next hop
        current_input_ids.extend(response_token_ids)

        # Invariant check after generation accumulation
        if not (len(all_response_token_ids) == len(all_loss_mask) == len(all_log_probs)):
            logger.error(f"Hop {hop} post-gen: length mismatch - tokens={len(all_response_token_ids)}, loss_mask={len(all_loss_mask)}, log_probs={len(all_log_probs)}")
            sample.status = Sample.Status.FAILED
            sample.metadata["tool_calling_error"] = f"Hop {hop} post-gen: length mismatch"
            break

        # Check if we should stop
        if not tool_calls:
            if config.stop_on_no_tool_call:
                break
            continue

        # Limit tool calls per hop
        tool_calls = tool_calls[: config.max_tool_calls_per_hop]

        # Execute tools
        results = await registry.execute_batch(tool_calls)

        # Check for errors
        if config.stop_on_error and any(not r.ok for r in results):
            break

        # Format observations and add to context
        for i, result in enumerate(results):
            observation = format_observation(result, parser_name)
            observation_token_ids = state.tokenizer(observation, add_special_tokens=False)["input_ids"]
            num_obs_tokens = len(observation_token_ids)

            # Accumulate observation (loss_mask = 0 for observations - not trained on)
            all_response_text += observation
            all_response_token_ids.extend(observation_token_ids)
            all_log_probs.extend([0.0] * num_obs_tokens)  # Dummy logprobs for observations
            all_loss_mask.extend([0] * num_obs_tokens)

            # Update input for next hop
            current_input_ids.extend(observation_token_ids)

            # Invariant check after each observation
            if not (len(all_response_token_ids) == len(all_loss_mask) == len(all_log_probs)):
                logger.error(f"Hop {hop} obs {i}: length mismatch - tokens={len(all_response_token_ids)}, loss_mask={len(all_loss_mask)}, log_probs={len(all_log_probs)}")
                sample.status = Sample.Status.FAILED
                sample.metadata["tool_calling_error"] = f"Hop {hop} obs {i}: length mismatch"
                break

        # Check finish reason
        if finish_reason.get("type") == "length":
            sample.status = Sample.Status.TRUNCATED
            break

    # =========================================================================
    # Final validation and sample update
    # =========================================================================

    # CRITICAL: Final alignment check before writing to sample
    response_len = len(all_response_token_ids)
    if response_len != len(all_loss_mask):
        logger.error(f"FINAL: token_ids/loss_mask mismatch: {response_len} vs {len(all_loss_mask)}")
        sample.status = Sample.Status.FAILED
        sample.metadata["tool_calling_error"] = f"FINAL: token_ids/loss_mask mismatch"
        return sample

    if response_len != len(all_log_probs):
        logger.error(f"FINAL: token_ids/log_probs mismatch: {response_len} vs {len(all_log_probs)}")
        sample.status = Sample.Status.FAILED
        sample.metadata["tool_calling_error"] = f"FINAL: token_ids/log_probs mismatch"
        return sample

    # Validate loss_mask values (must be 0 or 1)
    invalid_mask_values = [m for m in all_loss_mask if m not in (0, 1)]
    if invalid_mask_values:
        logger.error(f"Invalid loss_mask values: {invalid_mask_values[:10]}")
        sample.status = Sample.Status.FAILED
        sample.metadata["tool_calling_error"] = f"Invalid loss_mask values: {invalid_mask_values[:10]}"
        return sample

    # Update sample with validated data
    sample.tokens = prompt_token_ids + all_response_token_ids
    sample.response_length = response_len
    sample.response = all_response_text
    sample.loss_mask = all_loss_mask
    sample.rollout_log_probs = all_log_probs

    # Final structural validation
    expected_total_len = len(prompt_token_ids) + response_len
    if len(sample.tokens) != expected_total_len:
        logger.error(f"sample.tokens length {len(sample.tokens)} != prompt({len(prompt_token_ids)}) + response({response_len})")
        sample.status = Sample.Status.FAILED
        sample.metadata["tool_calling_error"] = f"sample.tokens length mismatch"
        return sample

    if sample.response_length != len(sample.loss_mask):
        logger.error(f"response_length {sample.response_length} != loss_mask length {len(sample.loss_mask)}")
        sample.status = Sample.Status.FAILED
        sample.metadata["tool_calling_error"] = f"response_length/loss_mask length mismatch"
        return sample

    # Set status if not already set (default is PENDING, not None)
    if sample.status is None or sample.status == Sample.Status.PENDING:
        if hop >= config.max_hops - 1:
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.COMPLETED

    return sample


def _get_registry(args: Namespace, sample: Sample) -> ToolRegistry:
    """Get or create a tool registry.

    Priority:
    1. sample.metadata["tool_names"] - list of tool names from TOOL_BINDINGS
    2. sample.metadata["tools"] - DEPRECATED: list of ToolSchema dicts (schema only, no func)
    3. args.tool_registry if present
    4. Default registry with calculator

    Note: Tool schemas (name/description/parameters) for parsing are separate from
    tool bindings (executable func). Metadata should use "tool_names" to reference
    pre-registered bindings, not try to define full ToolSpec with func.
    """
    from .tools import create_registry_from_names, ToolSchema

    # Check sample metadata for tool names (preferred approach)
    if sample.metadata and "tool_names" in sample.metadata:
        tool_names = sample.metadata["tool_names"]
        if isinstance(tool_names, list):
            try:
                return create_registry_from_names(tool_names)
            except ValueError as e:
                logger.warning(f"Failed to create registry from tool_names: {e}. Falling back to default.")
                return create_default_registry()

    # DEPRECATED: Check for old-style tool definitions (schema only, no func)
    # This path exists for backward compatibility but cannot create executable tools
    if sample.metadata and "tools" in sample.metadata:
        logger.warning("sample.metadata['tools'] is deprecated. Use sample.metadata['tool_names'] with pre-registered bindings instead. Falling back to default registry.")
        # We cannot create executable tools from schema-only definitions
        # Return default registry as fallback
        return create_default_registry()

    # Check args for explicit registry
    if hasattr(args, "tool_registry") and args.tool_registry is not None:
        return args.tool_registry

    # Default
    return create_default_registry()


# ============================================================================
# Reward Function (Optional)
# ============================================================================


async def reward_func(_args, _sample: Sample, **_kwargs) -> dict:
    """
    Placeholder reward function for tool calling.

    This function must be overridden via --custom-rm-path.

    Raises:
        NotImplementedError: Always raised. Provide your own reward function.
    """
    raise NotImplementedError("reward_func is a placeholder. Provide your own via --custom-rm-path. Expected signature: async def reward_func(args, sample: Sample, **kwargs) -> dict")
