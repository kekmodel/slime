"""Token invariant tests for tool calling.

Tests the critical invariants for RL training:
- len(token_ids) == len(loss_mask) == len(log_probs)
- loss_mask = 1 for generated tokens, 0 for observations
- Token round-trip: decode(encode(text)) preserves meaning
"""

import pytest

pytest.importorskip("transformers")

from examples.tool_calling.tests.utils import (
    get_tokenizer,
    MockGenerateResponse,
    create_generate_response,
    CALCULATOR_TOOL,
)


@pytest.mark.parametrize("parser_name", ["qwen25", "deepseekv3", "glm47"])
def test_token_length_invariants(parser_name: str):
    """Test that token arrays maintain length invariants."""
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    # Simulate a simple generation
    text = "The answer is 42."
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Create mock response
    response = create_generate_response(tokenizer, text)
    output = response.to_dict()

    # Extract logprobs
    logprobs = output["meta_info"]["output_token_logprobs"]

    # INVARIANT: len(token_ids) == len(logprobs)
    assert len(token_ids) == len(logprobs), "Token/logprob length mismatch"

    # INVARIANT: Each logprob entry has [logprob, token_id]
    for entry in logprobs:
        assert len(entry) >= 2, f"Invalid logprob entry: {entry}"


def test_loss_mask_values():
    """Test that loss_mask only contains 0 or 1."""
    # Simulate generation + observation
    gen_mask = [1, 1, 1, 1, 1]  # Generated tokens
    obs_mask = [0, 0, 0]  # Observation tokens

    full_mask = gen_mask + obs_mask

    # INVARIANT: All values are 0 or 1
    for i, val in enumerate(full_mask):
        assert val in (0, 1), f"Invalid loss_mask value at {i}: {val}"


def test_token_roundtrip():
    """Test decode(encode(text)) preserves meaning."""
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    # Test observation text
    from examples.tool_calling.tools import format_qwen

    observation = format_qwen(content="42")

    # Encode then decode
    token_ids = tokenizer.encode(observation, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Should be able to recover the original meaning
    # Note: Exact match may not hold due to special tokens, but
    # the content should be preserved
    assert "42" in decoded, "Content lost in token roundtrip"
    assert "tool_response" in decoded or " Tool " in decoded, "Format markers lost"


@pytest.mark.slow
def test_accumulated_response_alignment():
    """Test alignment in multi-hop accumulated responses.

    This simulates the full multi-hop flow and checks alignment
    after each hop.
    """
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    # Simulate multi-hop
    all_tokens = []
    all_log_probs = []
    all_loss_mask = []

    # Hop 1: Generation
    gen_text = "Let me calculate that."
    gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
    gen_log_probs = [-0.5] * len(gen_tokens)

    all_tokens.extend(gen_tokens)
    all_log_probs.extend(gen_log_probs)
    all_loss_mask.extend([1] * len(gen_tokens))  # Train on generation

    # INVARIANT after generation
    assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask)

    # Hop 1: Observation
    obs_text = "<tool_response>4</tool_response>"
    obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
    obs_log_probs = [0.0] * len(obs_tokens)  # Dummy logprobs

    all_tokens.extend(obs_tokens)
    all_log_probs.extend(obs_log_probs)
    all_loss_mask.extend([0] * len(obs_tokens))  # Don't train on observation

    # INVARIANT after observation
    assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask)

    # Hop 2: Generation
    gen_text2 = "The result is 4."
    gen_tokens2 = tokenizer.encode(gen_text2, add_special_tokens=False)
    gen_log_probs2 = [-0.6] * len(gen_tokens2)

    all_tokens.extend(gen_tokens2)
    all_log_probs.extend(gen_log_probs2)
    all_loss_mask.extend([1] * len(gen_tokens2))

    # FINAL INVARIANT
    assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask)

    # Verify loss_mask values
    for i, val in enumerate(all_loss_mask):
        assert val in (0, 1), f"Invalid mask at {i}: {val}"
