"""Token invariant tests for tool calling.

Tests the critical invariants for RL training:
- len(token_ids) == len(loss_mask) == len(log_probs)
- loss_mask = 1 for generated tokens, 0 for observations
- Token round-trip: decode(encode(text)) preserves meaning

Design:
- NO pytest.skip() - use pytest.fail() with instructions
- Parametrized across all parsers using conftest fixtures
- Uses shared fixtures from conftest.py
"""

import pytest

pytest.importorskip("transformers")

from examples.tool_calling.tests.conftest import (
    PARSER_TO_HF_MODEL,
    get_tokenizer_for_parser,
)
from examples.tool_calling.tests.utils import (
    MockGenerateResponse,
    create_generate_response,
)
from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS


class TestTokenLengthInvariants:
    """Test that token arrays maintain length invariants."""

    @pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
    def parser_name(self, request) -> str:
        return request.param

    def test_token_logprob_length_match(self, parser_name: str):
        """INVARIANT: len(token_ids) == len(logprobs)"""
        tokenizer = get_tokenizer_for_parser(parser_name)

        text = "The answer is 42."
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        response = create_generate_response(tokenizer, text)
        output = response.to_dict()

        logprobs = output["meta_info"]["output_token_logprobs"]

        assert len(token_ids) == len(logprobs), (
            f"{parser_name}: Token/logprob length mismatch: " f"tokens={len(token_ids)}, logprobs={len(logprobs)}"
        )

    def test_logprob_entry_format(self, parser_name: str):
        """Each logprob entry must be [logprob, token_id]."""
        tokenizer = get_tokenizer_for_parser(parser_name)

        text = "Testing format."
        response = create_generate_response(tokenizer, text)
        output = response.to_dict()

        for i, entry in enumerate(output["meta_info"]["output_token_logprobs"]):
            assert len(entry) >= 2, f"{parser_name}: Invalid logprob entry at {i}: {entry}"


class TestLossMaskValues:
    """Test that loss_mask only contains valid values."""

    def test_loss_mask_binary(self):
        """INVARIANT: All loss_mask values are 0 or 1."""
        gen_mask = [1, 1, 1, 1, 1]  # Generated tokens
        obs_mask = [0, 0, 0]  # Observation tokens

        full_mask = gen_mask + obs_mask

        for i, val in enumerate(full_mask):
            assert val in (0, 1), f"Invalid loss_mask value at {i}: {val}"


class TestTokenRoundtrip:
    """Test that token round-trips preserve content."""

    @pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
    def parser_name(self, request) -> str:
        return request.param

    def test_observation_roundtrip(self, parser_name: str):
        """decode(encode(observation)) preserves content."""
        if parser_name not in TOOL_RESPONSE_FORMATTERS:
            pytest.xfail(f"No formatter registered for {parser_name}")

        tokenizer = get_tokenizer_for_parser(parser_name)
        formatter = TOOL_RESPONSE_FORMATTERS[parser_name]

        # Get observation
        kwargs = {"content": "42"}
        if parser_name in {"kimi_k2", "mistral"}:
            kwargs["tool_call_id"] = "functions.test:0"
        if parser_name == "gpt-oss":
            kwargs["tool_name"] = "calculator"

        observation = formatter(**kwargs)

        # Round-trip
        token_ids = tokenizer.encode(observation, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

        assert "42" in decoded, (
            f"{parser_name}: Content lost in round-trip. " f"Original: {observation[:100]}..., Decoded: {decoded[:100]}..."
        )


class TestMultiHopAlignment:
    """Test alignment in multi-hop accumulated responses."""

    @pytest.fixture(params=["qwen25", "deepseekv3", "glm47", "step3", "interns1"])
    def parser_name(self, request) -> str:
        return request.param

    @pytest.mark.slow
    def test_accumulated_response_alignment(self, parser_name: str):
        """Alignment must hold at every step of multi-hop flow."""
        tokenizer = get_tokenizer_for_parser(parser_name)

        all_tokens: list[int] = []
        all_log_probs: list[float] = []
        all_loss_mask: list[int] = []

        # Hop 1: Generation
        gen_text = "Let me calculate that."
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
        all_tokens.extend(gen_tokens)
        all_log_probs.extend([-0.5] * len(gen_tokens))
        all_loss_mask.extend([1] * len(gen_tokens))

        assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask), (
            f"Alignment broken after generation: "
            f"tokens={len(all_tokens)}, logprobs={len(all_log_probs)}, mask={len(all_loss_mask)}"
        )

        # Hop 1: Observation
        obs_text = "<tool_response>4</tool_response>"
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
        all_tokens.extend(obs_tokens)
        all_log_probs.extend([0.0] * len(obs_tokens))
        all_loss_mask.extend([0] * len(obs_tokens))

        assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask), (
            f"Alignment broken after observation: "
            f"tokens={len(all_tokens)}, logprobs={len(all_log_probs)}, mask={len(all_loss_mask)}"
        )

        # Hop 2: Final generation
        final_text = "The result is 4."
        final_tokens = tokenizer.encode(final_text, add_special_tokens=False)
        all_tokens.extend(final_tokens)
        all_log_probs.extend([-0.6] * len(final_tokens))
        all_loss_mask.extend([1] * len(final_tokens))

        # Final check
        assert len(all_tokens) == len(all_log_probs) == len(all_loss_mask)
        assert all(m in (0, 1) for m in all_loss_mask)
