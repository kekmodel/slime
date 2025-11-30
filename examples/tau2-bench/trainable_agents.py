"""
Trainable Agent for tau2-bench RLVR Training

This module provides a trainable agent for tau2-bench that:
- Uses sglang server for LLM inference
- Collects turn-by-turn data for step-based sampling
- Handles think tags (only last turn includes think for learning)
- Distributes final reward to all turns (gamma=1.0)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

# Local imports
from .sglang_tool_parser import parse_tools
from .step_sampler import StepInteractionResult, Status, TurnData

# tau2 imports
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool

logger = logging.getLogger(__name__)


TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


def tools_to_openai_format(tools: List[Tool]) -> List[Dict[str, Any]]:
    """Convert tau2 tools to OpenAI format for chat template."""
    return [tool.openai_schema for tool in tools]


class TrainableTau2Agent:
    """
    A trainable agent for tau2-bench that uses sglang server for LLM inference.

    This agent:
    - Interacts with tau2 environment using tool calls
    - Collects turn-by-turn data for step-based RLVR training
    - Handles think tags: only last turn includes reasoning in action
    - Distributes final reward to all turns (gamma=1.0)
    """

    def __init__(
        self,
        tools: List[Tool],
        policy: str,
        rollout_args: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainable agent.

        Args:
            tools: List of tau2 Tool objects available to the agent
            policy: Domain policy text for the agent
            rollout_args: Rollout configuration (sglang server IP, port, etc.)
            sampling_params: LLM sampling parameters
        """
        self.tools = tools
        self.tools_info = tools_to_openai_format(tools)
        self.policy = policy

        self.rollout_args = rollout_args or {
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "use_http2": False,
        }
        self.sampling_params = sampling_params or {
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
        }

    def _reformulate_tool_call(self, text: str) -> str:
        """Reformulate tool call instruction for tau2 environment."""
        return text.replace(
            "You may call one or more functions to assist with the user query.",
            TOOL_INSTRUCTION,
        )

    async def _call_llm(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make an async LLM call to sglang server."""
        return await post(url, payload)

    def _parse_tool(self, response: str) -> Dict[str, Any]:
        """Parse tool calls from LLM response string."""
        return parse_tools(response, self.tools_info, "qwen25")

    def _build_system_prompt(self) -> str:
        """Build the system prompt with policy."""
        return f"""You are a customer service agent that helps the user according to the policy provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.

<policy>
{self.policy}
</policy>"""

    def _build_initial_messages(self, user_message: str) -> List[Dict[str, Any]]:
        """Build initial conversation messages."""
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

    def _prepare_prompt_tokens(
        self, state: GenerateState, messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[int]]:
        """Prepare prompt text and tokenize it."""
        prompt_text = state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=self.tools_info,
        )
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)[
            "input_ids"
        ]
        return prompt_text, prompt_token_ids

    def _get_token_delta(
        self, tokenizer: AutoTokenizer, messages: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[int]]:
        """
        Calculate token delta for multi-turn conversations.

        Returns:
            Tuple of (token_ids, loss_mask)
        """
        curr = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False, tools=self.tools_info
        )
        curr = self._reformulate_tool_call(curr)
        token_ids = []
        loss_mask = []

        if messages[-1]["role"] == "assistant":
            prev = tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                tokenize=False,
                tools=self.tools_info,
            )
            prev = self._reformulate_tool_call(prev)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [1] * len(new_tokens)
        else:
            prev = tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=False,
                tokenize=False,
                tools=self.tools_info,
            )
            prev = self._reformulate_tool_call(prev)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [0] * len(new_tokens)

        return token_ids, loss_mask

    async def asolve(
        self,
        env: Environment,
        rollout_args: Dict[str, Any],
        sampling_params: Dict[str, Any],
        initial_user_message: str,
        max_num_steps: int = 30,
    ) -> StepInteractionResult:
        """
        Execute async agent-environment interaction for training.

        This method collects turn-by-turn data for step-based RLVR training:
        - Each turn's state (accumulated tokens) and action are stored separately
        - Think/reasoning is only included in the last turn's action
        - Final reward is distributed to all turns after trajectory ends

        Args:
            env: tau2 Environment instance
            rollout_args: Rollout configuration arguments
            sampling_params: LLM sampling parameters
            initial_user_message: Initial user message to start conversation
            max_num_steps: Maximum number of interaction steps

        Returns:
            StepInteractionResult containing turn-level data for step sampling
        """
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"

        # Build initial conversation
        messages = self._build_initial_messages(initial_user_message)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages)

        # Initialize tracking variables
        turns_data: List[TurnData] = []
        cumulative_tokens = prompt_token_ids.copy()
        total_reward = 0.0
        info: Dict[str, Any] = {}

        # Initialize result
        result = StepInteractionResult(
            prompt=prompt_text,
            reward=0.0,
            messages=[],
            info={},
            status=Status.COMPLETED,
        )

        done = False
        env_response_text = ""

        for step in range(max_num_steps):
            # Store current state (tokens accumulated so far)
            current_state = cumulative_tokens.copy()

            # Prepare payload for sglang
            text_input = state.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=self.tools_info,
            )
            text_input = self._reformulate_tool_call(text_input)
            payload = {"text": text_input, "sampling_params": sampling_params}

            # Send request to sglang server
            output = await self._call_llm(url, payload)

            # Check for abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                result.status = Status.ABORTED
                break

            # Get response text (without think if reasoning-parser is used)
            response_text = output["text"]
            reasoning_content = output.get("reasoning_content", "")

            # Remove end of conversation token if present
            if response_text.endswith("<|im_end|>"):
                response_text = response_text[:-10]

            # Parse tool calls
            try:
                parsed = self._parse_tool(response_text)
                normal_text = parsed["normal_text"]
                calls = parsed["calls"]
            except Exception as e:
                logger.warning(f"Tool parsing failed: {e}")
                result.status = Status.ABORTED
                break

            # Execute action in environment
            if calls:
                tool_call_data = calls[0]
                try:
                    params = json.loads(tool_call_data["parameters"])
                except json.JSONDecodeError:
                    params = {}

                tool_call = ToolCall(
                    id=f"call_{step}_{tool_call_data['name']}",
                    name=tool_call_data["name"],
                    arguments=params,
                    requestor="assistant",
                )
                tool_response = env.get_response(tool_call)
                env_response_text = tool_response.content

                # Add assistant message with tool call
                messages.append({"role": "assistant", "content": response_text})

                # Add tool response
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_call.name,
                        "content": env_response_text,
                    }
                )
            else:
                # Direct response to user (conversation might end here)
                messages.append({"role": "assistant", "content": response_text})
                done = True

            # Determine if this is the last turn
            is_last_turn = done or (step == max_num_steps - 1)

            # Build action tokens for training
            # IMPORTANT: Each step's training sample includes think+action
            # (rollout history excludes think, but training treats each step as independent trajectory)
            if reasoning_content:
                action_text = reasoning_content + response_text
            else:
                action_text = response_text

            action_token_ids = state.tokenizer.encode(
                action_text, add_special_tokens=False
            )
            action_loss_mask = [1] * len(action_token_ids)

            # Store turn data (reward will be set later)
            turns_data.append(
                TurnData(
                    turn_idx=step,
                    state_token_ids=current_state,
                    action_token_ids=action_token_ids,
                    action_loss_mask=action_loss_mask,
                    reward=None,
                )
            )

            # Update cumulative tokens for next turn's state
            # State accumulates: assistant response (without think) + tool response (if any)
            #
            # messages structure at this point:
            # - If tool call: [..., assistant_msg, tool_msg]
            # - If no tool call: [..., assistant_msg]

            if calls:
                # Tool call case: add both assistant response and tool response
                # Get assistant token delta (messages[-2] is assistant, messages[-1] is tool)
                assistant_msgs = messages[:-1]  # up to and including assistant
                assistant_delta, _ = self._get_token_delta(state.tokenizer, assistant_msgs)
                cumulative_tokens.extend(assistant_delta)

                # Get tool response token delta
                tool_delta, _ = self._get_token_delta(state.tokenizer, messages)
                cumulative_tokens.extend(tool_delta)
            else:
                # No tool call: just add assistant response
                assistant_delta, _ = self._get_token_delta(state.tokenizer, messages)
                cumulative_tokens.extend(assistant_delta)

            if done:
                result.status = Status.COMPLETED
                break

        # Handle truncation
        if not done:
            result.status = Status.TRUNCATED

        # Distribute final reward to all turns (gamma=1.0)
        # In tau2, reward is determined by evaluation after simulation
        # For now, we set a placeholder; actual reward will be set in generate_with_tau2.py
        for turn in turns_data:
            turn.reward = total_reward

        # Build final result
        result.turns = turns_data
        result.reward = total_reward
        result.info = info
        result.messages = messages
        result.response = "".join(
            [msg.get("content", "") for msg in messages if msg["role"] == "assistant"]
        )

        return result


def create_trainable_agent(
    tools: List[Tool],
    policy: str,
    rollout_args: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
) -> TrainableTau2Agent:
    """
    Factory function to create a trainable tau2 agent.

    Args:
        tools: List of tau2 Tool objects
        policy: Domain policy text
        rollout_args: Rollout configuration
        sampling_params: LLM sampling parameters

    Returns:
        Configured TrainableTau2Agent instance
    """
    return TrainableTau2Agent(
        tools=tools,
        policy=policy,
        rollout_args=rollout_args,
        sampling_params=sampling_params,
    )
