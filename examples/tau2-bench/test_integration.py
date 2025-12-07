"""
Integration Tests for SLIME tau2-bench integration.

These tests verify the core functionality without requiring a running sglang server.
Uses .env configuration for API keys.
"""

import os
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from message_utils import (
    build_tool_specs,
    get_token_delta,
    prepare_initial_messages,
    tau2_message_to_dict,
)
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.gym import AgentGymEnv, register_gym_agent
from tau2.registry import registry
from tool_parser import ParseResult, convert_to_tau2_message, parse_tool_calls

# Use Qwen tokenizer (free, no API needed)
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"


@dataclass
class MockSample:
    """Mock SLIME Sample for testing."""

    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    index: int = 0
    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float = 0.0
    loss_mask: list[int] = field(default_factory=list)
    status: str = "pending"
    metadata: dict = field(default_factory=dict)
    rollout_log_probs: list[float] | None = None


class TestToolParser:
    """Test tool call parsing functionality."""

    @pytest.fixture
    def tool_specs(self):
        """Sample tool specifications."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_customer",
                    "description": "Search for a customer by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                        },
                        "required": ["customer_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_account",
                    "description": "Update customer account",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_id": {"type": "string"},
                            "new_value": {"type": "string"},
                        },
                    },
                },
            },
        ]

    def test_parse_plain_text(self, tool_specs):
        """Test parsing plain text (no tool calls)."""
        response = "Hello, how can I help you today?"
        result = parse_tool_calls(response, tool_specs)

        # Should not fail, but may not find tool calls
        assert isinstance(result, ParseResult)
        assert result.normal_text == response or result.normal_text != ""

    def test_convert_to_tau2_message_no_calls(self):
        """Test converting parse result without tool calls."""
        parsed = ParseResult(
            success=True,
            normal_text="Hello, I can help you.",
            calls=[],
        )

        message = convert_to_tau2_message(parsed)

        assert isinstance(message, AssistantMessage)
        assert message.content == "Hello, I can help you."
        assert message.tool_calls is None

    def test_convert_to_tau2_message_with_calls(self):
        """Test converting parse result with tool calls."""
        parsed = ParseResult(
            success=True,
            normal_text="Let me search for that.",
            calls=[
                {
                    "name": "search_customer",
                    "parameters": {"customer_id": "12345"},
                }
            ],
        )

        message = convert_to_tau2_message(parsed, "raw response")

        assert isinstance(message, AssistantMessage)
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "search_customer"


class TestMessageUtils:
    """Test message conversion utilities."""

    @pytest.fixture
    def tokenizer(self):
        """Load test tokenizer."""
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def test_tau2_message_to_dict_user(self):
        """Test converting UserMessage to dict."""
        msg = UserMessage(role="user", content="Hello")
        result = tau2_message_to_dict(msg)

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_tau2_message_to_dict_assistant_with_tools(self):
        """Test converting AssistantMessage with tool calls."""
        tool_call = ToolCall(
            name="search",
            arguments={"query": "test"},
            requestor="assistant",
        )
        msg = AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        )
        result = tau2_message_to_dict(msg)

        assert result["role"] == "assistant"
        assert "search" in result["content"]

    def test_build_tool_specs(self):
        """Test building tool specs from mock tools."""
        # Create mock tools
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.short_desc = "A test tool"
        mock_tool.params = None

        specs = build_tool_specs([mock_tool])

        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "test_tool"

    def test_prepare_initial_messages(self):
        """Test preparing initial messages."""
        messages = prepare_initial_messages(
            policy="Be helpful.",
            initial_observation="Hello, I need help.",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_get_token_delta_assistant(self, tokenizer):
        """Test token delta for assistant message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        token_ids, loss_mask = get_token_delta(tokenizer, messages)

        assert len(token_ids) > 0
        assert len(loss_mask) == len(token_ids)
        assert all(m == 1 for m in loss_mask)  # All trainable

    def test_get_token_delta_user(self, tokenizer):
        """Test token delta for user message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        token_ids, loss_mask = get_token_delta(tokenizer, messages)

        assert len(token_ids) > 0
        assert len(loss_mask) == len(token_ids)
        assert all(m == 0 for m in loss_mask)  # Not trainable


class TestEnvironmentIntegration:
    """Test tau2 environment integration."""

    @pytest.fixture
    def domain(self):
        """Test domain from .env."""
        return os.environ.get("TAU2_DOMAIN", "telecom")

    def test_registry_available(self, domain):
        """Test that registry has the domain."""
        info = registry.get_info()
        # Should have at least some task sets
        assert len(info.task_sets) > 0

    def test_environment_creation(self, domain):
        """Test AgentGymEnv creation."""
        try:
            register_gym_agent()
        except Exception:
            pass

        # Get a task ID
        try:
            tasks = registry.get_tasks_loader(domain)()
            if not tasks:
                pytest.skip(f"No tasks available for domain {domain}")
            task_id = tasks[0].id
        except Exception as e:
            pytest.skip(f"Could not load tasks: {e}")

        user_llm = os.environ.get("TAU2_USER_LLM", "gpt-4.1")
        user_temp = float(os.environ.get("TAU2_USER_TEMP", "0.0"))

        env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            max_steps=10,
            user_llm=user_llm,
            user_llm_args={"temperature": user_temp},
        )

        assert env is not None
        assert env.domain == domain


class TestTokenAlignment:
    """Test token and loss_mask alignment."""

    @pytest.fixture
    def tokenizer(self):
        """Load test tokenizer."""
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def test_multi_turn_alignment(self, tokenizer):
        """Test alignment across multiple turns."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # Tokenize initial prompt
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)[
            "input_ids"
        ]

        response_tokens = []
        loss_masks = []

        # Simulate multi-turn
        messages.append({"role": "assistant", "content": "2+2 equals 4."})
        token_ids, mask = get_token_delta(tokenizer, messages)
        response_tokens.extend(token_ids)
        loss_masks.extend(mask)

        messages.append({"role": "user", "content": "Thanks!"})
        token_ids, mask = get_token_delta(tokenizer, messages)
        response_tokens.extend(token_ids)
        loss_masks.extend(mask)

        # Verify alignment
        assert len(response_tokens) == len(loss_masks)

        # Verify assistant tokens are trainable (sum of 1s in loss_masks)
        trainable_count = sum(loss_masks)
        assert trainable_count > 0  # At least some trainable tokens

    def test_loss_mask_correctness(self, tokenizer):
        """Test that loss_mask correctly marks trainable tokens."""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Hello"},
        ]

        all_response_tokens = []
        all_loss_masks = []

        # Add assistant (should be trainable)
        messages.append({"role": "assistant", "content": "Hi there!"})
        tokens, mask = get_token_delta(tokenizer, messages)
        all_response_tokens.extend(tokens)
        all_loss_masks.extend(mask)
        assistant_count = len(tokens)

        # Add user (should not be trainable)
        messages.append({"role": "user", "content": "Bye"})
        tokens, mask = get_token_delta(tokenizer, messages)
        all_response_tokens.extend(tokens)
        all_loss_masks.extend(mask)

        # Verify counts
        trainable_count = sum(all_loss_masks)
        assert trainable_count == assistant_count


class TestEnvVars:
    """Test environment variable loading."""

    def test_env_loaded(self):
        """Test that .env values are loaded."""
        domain = os.environ.get("TAU2_DOMAIN", "telecom")
        user_llm = os.environ.get("TAU2_USER_LLM", "gpt-4.1")
        max_steps = int(os.environ.get("TAU2_MAX_STEPS", "100"))
        tool_parser = os.environ.get("TAU2_TOOL_PARSER", "qwen")

        logger.info(f"Domain: {domain}")
        logger.info(f"User LLM: {user_llm}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Tool parser: {tool_parser}")

        assert domain == "telecom"  # From .env
        assert max_steps > 0
        assert tool_parser in ["qwen", "llama3", "mistral"]


class TestLiveEnvironment:
    """Live tests that use the actual tau2 environment (no LLM calls)."""

    def test_env_reset(self):
        """Test environment reset returns valid observation."""
        try:
            register_gym_agent()
        except Exception:
            pass

        domain = os.environ.get("TAU2_DOMAIN", "telecom")
        user_llm = os.environ.get("TAU2_USER_LLM", "gpt-4.1")
        user_temp = float(os.environ.get("TAU2_USER_TEMP", "0.0"))

        tasks = registry.get_tasks_loader(domain)()
        task_id = tasks[0].id

        env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            max_steps=5,
            user_llm=user_llm,
            user_llm_args={"temperature": user_temp},
        )

        obs, info = env.reset()

        assert obs is not None
        assert "policy" in info
        assert "tools" in info
        logger.info(f"Initial observation: {obs[:100]}...")
        logger.info(f"Tools count: {len(info['tools'])}")

    def test_tool_specs_from_env(self):
        """Test building tool specs from real environment."""
        try:
            register_gym_agent()
        except Exception:
            pass

        domain = os.environ.get("TAU2_DOMAIN", "telecom")
        user_llm = os.environ.get("TAU2_USER_LLM", "gpt-4.1")
        user_temp = float(os.environ.get("TAU2_USER_TEMP", "0.0"))

        tasks = registry.get_tasks_loader(domain)()
        task_id = tasks[0].id

        env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            max_steps=5,
            user_llm=user_llm,
            user_llm_args={"temperature": user_temp},
        )

        obs, info = env.reset()
        tools = info.get("tools", [])
        tool_specs = build_tool_specs(tools)

        assert len(tool_specs) > 0
        for spec in tool_specs:
            assert "type" in spec
            assert "function" in spec
            assert "name" in spec["function"]
            logger.info(f"Tool: {spec['function']['name']}")


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
