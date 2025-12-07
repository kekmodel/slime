"""
SLIME 학습용 AgentGymEnv 확장.

AgentGymEnv를 상속하여 chat template 호환 형식의 messages를 info에 포함.
문자열 파싱 없이 apply_chat_template()에 직접 사용 가능.
"""

import json
import re
from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.gym import AgentGymEnv
from tau2.utils import llm_utils
from tau2.utils.tools import to_functional_format


def _to_litellm_messages_with_thinking(messages: list[Message]) -> list[dict]:
    """
    Convert tau2 messages to litellm format, preserving thinking blocks.

    Extended thinking 모드에서 Claude는 이전 assistant 메시지에
    thinking blocks가 있어야 합니다. 원본 to_litellm_messages()는
    thinking blocks를 포함하지 않아서 에러가 발생합니다.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]

            msg_dict = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": tool_calls,
            }

            # Preserve thinking blocks from raw_data if available
            if message.raw_data:
                thinking_blocks = message.raw_data.get("thinking_blocks")
                if thinking_blocks:
                    msg_dict["thinking_blocks"] = thinking_blocks

            litellm_messages.append(msg_dict)
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


# Monkey-patch tau2's to_litellm_messages to preserve thinking blocks
llm_utils.to_litellm_messages = _to_litellm_messages_with_thinking


def _flip_roles_with_thinking(self) -> list:
    """
    flip_roles that preserves raw_data (including thinking_blocks).

    원본 flip_roles()는 새 메시지를 만들 때 raw_data를 복사하지 않아서
    thinking blocks가 손실됩니다.
    """
    from tau2.data_model.message import ToolMessage as TM

    flipped_messages = []
    for message in self.messages:
        if isinstance(message, UserMessage):
            flipped_messages.append(
                AssistantMessage(
                    role="assistant",
                    tool_calls=message.tool_calls,
                    content=message.content,
                    raw_data=message.raw_data,  # Preserve raw_data!
                )
            )
        elif isinstance(message, AssistantMessage):
            if not message.is_tool_call():
                flipped_messages.append(
                    UserMessage(
                        role="user",
                        content=message.content,
                    )
                )
            else:
                raise ValueError(
                    f"Tool calls are not supported in the flipped messages: {message}"
                )
        elif isinstance(message, TM):
            if message.requestor == "user":
                flipped_messages.append(
                    TM(
                        id=message.id,
                        role=message.role,
                        content=message.content,
                    )
                )
            else:
                raise ValueError(
                    f"Tool messages should be sent to the user: {message}"
                )
        else:
            raise ValueError(f"Unknown message role: {message.role}")
    return flipped_messages


# Monkey-patch UserState.flip_roles to preserve thinking blocks
from tau2.user.base import UserState
UserState.flip_roles = _flip_roles_with_thinking

# Role prefix pattern for tau2 observation format
_ROLE_PREFIX_PATTERN = re.compile(r"^(user|assistant|tool|system): ", re.MULTILINE)


def _strip_role_prefix(observation: str) -> str:
    """
    tau2 gym observation에서 role prefix 제거.

    tau2 gym은 observation을 "role: content" 형식으로 반환.
    예: "user: Hi there!" -> "Hi there!"

    Args:
        observation: tau2 gym의 observation 문자열

    Returns:
        role prefix가 제거된 순수 content
    """
    if not observation:
        return observation

    # 멀티라인 observation에서 각 줄의 role prefix 제거
    return _ROLE_PREFIX_PATTERN.sub("", observation)


class SlimeAgentGymEnv(AgentGymEnv):
    """
    SLIME 학습에 최적화된 AgentGymEnv.

    변경사항:
    - step()의 info에 'new_messages' 추가 (chat template 호환 형식)
    - apply_chat_template()에 직접 사용 가능
    - 문자열 파싱 불필요
    - set_pending_tool_calls()로 tool_call_id 매핑 지원

    사용 예시:
        env = SlimeAgentGymEnv(domain="telecom", task_id="task_001")
        obs, info = env.reset()

        while not terminated:
            # Tool call이 있는 경우, step 전에 ID 설정
            if tool_calls:
                env.set_pending_tool_calls(tool_calls)

            obs, reward, terminated, truncated, info = env.step(action)

            # Chat template 호환 messages (tool_call_id 포함)
            if "new_messages" in info:
                context_messages.extend(info["new_messages"])
                # 바로 apply_chat_template(context_messages, ...) 사용 가능
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_observation_len = 0
        # Store pending tool calls for ID mapping
        # Each entry: {"name": str, "id": str}
        self._pending_tool_calls: list[dict] = []

    def set_pending_tool_calls(self, tool_calls: list[dict]) -> None:
        """
        Set expected tool calls before calling step().

        tau2의 parse_action_string()은 tool_call_id를 보존하지 않으므로,
        이 메서드로 미리 ID 매핑을 설정해야 합니다.

        Args:
            tool_calls: List of tool call dicts with "name" and "id" keys.
                Example: [{"name": "get_customer", "id": "call_0"}, ...]

        Usage:
            # Before calling step with tool calls
            env.set_pending_tool_calls([
                {"name": "get_customer", "id": "functions.get_customer:0"},
                {"name": "get_order", "id": "functions.get_order:1"},
            ])
            obs, reward, terminated, _, info = env.step(action_str)
            # info["new_messages"] will have correct tool_call_ids
        """
        self._pending_tool_calls = [
            {"name": tc.get("name", ""), "id": tc.get("id", "")}
            for tc in tool_calls
        ]

    def reset(self, seed: Optional[int] = None) -> tuple[str, dict]:
        """Reset and track initial observation length."""
        obs, info = super().reset(seed=seed)

        # 초기 observation 길이 저장
        if self._agent:
            self._prev_observation_len = len(self._agent.observation)
        else:
            self._prev_observation_len = 0

        # 초기 messages도 info에 포함
        info["new_messages"] = self._get_new_messages()

        # Role prefix 제거 (e.g., "user: Hello" -> "Hello")
        clean_obs = _strip_role_prefix(obs)

        return clean_obs, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """
        Execute an action and advance the simulation.

        Returns:
            - observation: 문자열 형식 (role prefix 제거됨)
            - reward: 평가 결과
            - terminated: 종료 여부
            - truncated: False
            - info: dict with:
                - new_messages: 이번 step에서 추가된 메시지들 (chat template 형식)
                    [{"role": "tool", "tool_call_id": "...", "content": "..."}, ...]
                    또는 [{"role": "user", "content": "..."}]
                - turn_complete: bool - True if user responded (turn ended),
                    False if only tool response (turn continues)

        Note:
            Tool call IDs를 정확하게 받으려면 step() 전에
            set_pending_tool_calls()를 호출하세요.
        """
        # 부모 클래스 step 호출
        obs, reward, terminated, truncated, info = super().step(action)

        # 새로 추가된 messages 추출 (pending tool calls 사용)
        new_messages = self._get_new_messages()
        info["new_messages"] = new_messages

        # Clear pending tool calls after processing
        self._pending_tool_calls = []

        # Turn completion check: turn ends when user responds
        # Tool responses mean agent needs to continue (same turn)
        turn_complete = False
        for msg in new_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                turn_complete = True
                break
        info["turn_complete"] = turn_complete or terminated

        # 현재 observation 길이 업데이트
        if self._agent:
            self._prev_observation_len = len(self._agent.observation)

        # Role prefix 제거 (e.g., "user: Hello" -> "Hello")
        clean_obs = _strip_role_prefix(obs)

        return clean_obs, reward, terminated, truncated, info

    def _get_new_messages(self) -> list[dict]:
        """
        이전 step 이후 새로 추가된 메시지들을 chat template 형식으로 반환.

        Returns:
            Chat template 호환 형식의 message 리스트:
            - Tool response: {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
            - User message: {"role": "user", "content": "..."}

        Note:
            Tool call IDs는 set_pending_tool_calls()로 설정된 값을 사용합니다.
            tau2의 parse_action_string()은 tool_call_id를 보존하지 않기 때문입니다.
        """
        if self._agent is None:
            return []

        current_obs = self._agent.observation
        new_messages = current_obs[self._prev_observation_len :]

        # Build tool call info from pending tool calls (set before step)
        # These have the correct IDs from the caller
        pending_tool_info = list(self._pending_tool_calls)  # Copy to consume
        tool_response_idx = 0

        result = []
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                # Get tool_call_id from pending tool calls by position
                if tool_response_idx < len(pending_tool_info):
                    pending = pending_tool_info[tool_response_idx]
                    tool_call_id = pending.get("id", "")
                    tool_name = pending.get("name", "tool")
                    tool_response_idx += 1
                else:
                    # Fallback: use whatever is in the message
                    tool_call_id = msg.id or ""
                    tool_name = "tool"

                result.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": msg.content or "",
                })
            else:
                result.append(self._message_to_dict(msg, {}))

        return result

    def _message_to_dict(self, msg: Message, tool_call_map: dict = None) -> dict:
        """
        tau2 Message를 chat template 호환 dict로 변환.

        Args:
            msg: tau2 Message 객체
            tool_call_map: {tool_call_id: tool_name} 매핑 (ID 기반 매칭)

        Returns:
            Chat template 호환 dict
        """
        if isinstance(msg, ToolMessage):
            # Use msg.id directly (tau2 preserves tool_call.id in ToolMessage.id)
            tool_call_id = msg.id or ""
            tool_name = "tool"

            # ID-based matching: look up tool name from tool_call_map
            if tool_call_map and tool_call_id in tool_call_map:
                tool_name = tool_call_map[tool_call_id]
            elif tool_call_id:
                # Fallback: try to extract name from id format "functions.{name}:{idx}"
                if "." in tool_call_id:
                    parts = tool_call_id.split(".")
                    if len(parts) > 1:
                        tool_name = parts[1].split(":")[0]
                else:
                    tool_name = tool_call_id.split(":")[0]

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": msg.content or "",
            }
        elif isinstance(msg, UserMessage):
            if msg.is_tool_call():
                tool_calls_str = ", ".join(
                    [to_functional_format(t) for t in msg.tool_calls]
                )
                return {"role": "user", "content": tool_calls_str}
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, AssistantMessage):
            if msg.is_tool_call():
                tool_calls = []
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                    )
                return {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }
            return {"role": "assistant", "content": msg.content}
        else:
            return {"role": msg.role, "content": msg.content}
