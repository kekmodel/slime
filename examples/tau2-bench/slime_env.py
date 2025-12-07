"""
SLIME 학습을 위한 새로운 Gymnasium 환경.

AgentGymEnv를 상속하지 않고, tau2의 핵심 컴포넌트만 직접 사용.
Threading/Orchestrator 없이 단순한 동기 실행.

주요 특징:
- 구조화된 Action: ToolCall 객체 또는 텍스트
- 구조화된 Observation: Message 리스트 (chat template 호환)
- tool_call_id 완벽 보존
- multi-tool call 네이티브 지원
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator

# =============================================================================
# Type Definitions
# =============================================================================


@dataclass
class ToolCallAction:
    """Tool call action."""

    name: str
    arguments: dict[str, Any]
    id: str  # tool_call_id - 반드시 제공해야 함


@dataclass
class TextAction:
    """Plain text action (user에게 메시지 전송)."""

    content: str


# Action은 tool calls 또는 text
Action = list[ToolCallAction] | TextAction


@dataclass
class StepResult:
    """step() 결과."""

    messages: list[dict]  # 새로 추가된 메시지들 (chat template 형식)
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class EnvState:
    """환경 상태."""

    messages: list[Message] = field(default_factory=list)
    turn_idx: int = 0
    step_count: int = 0
    terminated: bool = False


# =============================================================================
# SlimeGymEnv
# =============================================================================


class SlimeGymEnv(gym.Env):
    """
    SLIME 학습을 위한 Gymnasium 환경.

    AgentGymEnv와 달리:
    - Threading 없음 (단순 동기 실행)
    - Orchestrator 없음 (직접 Environment/UserSimulator 호출)
    - 구조화된 Action/Observation
    - tool_call_id 완벽 보존
    - multi-tool call 네이티브 지원

    사용 예시:
        env = SlimeGymEnv(domain="telecom", task_id="task_001")
        obs, info = env.reset()

        # obs["messages"]는 chat template 호환 형식
        # tokenizer.apply_chat_template(obs["messages"], ...)

        while not terminated:
            # Tool call action
            action = [
                ToolCallAction(name="get_customer", arguments={"id": "123"}, id="call_0"),
            ]
            # 또는 text action
            action = TextAction(content="How can I help you?")

            result = env.step(action)
            # result.messages: 새로 추가된 메시지들
            # result.terminated: 종료 여부
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        domain: str,
        task_id: str,
        max_steps: int = 100,
        user_llm: Optional[str] = None,
        user_llm_args: Optional[dict] = None,
        solo_mode: bool = False,
    ):
        """
        Args:
            domain: Domain 이름 (e.g., "telecom", "airline", "retail")
            task_id: Task ID
            max_steps: 최대 step 수
            user_llm: User simulator LLM
            user_llm_args: User simulator LLM 인자
            solo_mode: Solo mode (user simulator 없이 실행)
        """
        super().__init__()

        self.domain = domain
        self.task_id = task_id
        self.max_steps = max_steps
        self.user_llm = user_llm or "claude-haiku-4-5"
        self.user_llm_args = user_llm_args or {}
        self.solo_mode = solo_mode

        # 내부 상태 (reset에서 초기화)
        self._env: Optional[Environment] = None
        self._task: Optional[Task] = None
        self._user: Optional[UserSimulator] = None
        self._state: Optional[EnvState] = None

        # 디버깅용: 마지막으로 user simulator에게 전달된 state
        # flip_roles() 적용 전 상태 (agent tool 제거됨)
        self._last_user_state_messages: Optional[list[Message]] = None

        # Gymnasium spaces (문자열 기반으로 단순화)
        self.observation_space = spaces.Dict(
            {
                "messages": spaces.Sequence(spaces.Dict({})),
            }
        )
        self.action_space = spaces.Dict({})

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        """
        환경 초기화.

        Returns:
            observation: {
                "messages": [{"role": "user", "content": "..."}],  # chat template 형식
                "tools": [...],  # tool 정의
                "policy": "...",  # system prompt
            }
            info: 추가 정보
        """
        super().reset(seed=seed)

        # Environment, Task, User 생성
        self._env = self._create_environment()
        self._task = self._get_task()
        self._user = self._create_user()

        # 상태 초기화
        self._state = EnvState()

        # Task의 initial_state 적용
        if self._task.initial_state:
            self._apply_initial_state()

        # tau2 원래 flow:
        # 1. Agent가 먼저 인사 ("Hi! How can I help you today?")
        # 2. User simulator가 응답 (user_scenario에 따라)
        # 3. 이후 agent-user 반복
        #
        # reset()에서는:
        # - Agent의 첫 인사를 state에 추가
        # - User simulator의 첫 응답을 생성
        # - observation으로 user 메시지 반환 (agent가 응답할 차례)
        initial_messages = []
        if not self.solo_mode:
            # Agent's first greeting
            first_agent_msg = AssistantMessage(
                role="assistant",
                content="Hi! How can I help you today?",
            )
            self._state.messages.append(first_agent_msg)

            # User simulator's first response
            first_user_msg, intermediate_msgs = self._get_user_message()

            # Save intermediate messages (if user called tools)
            for msg in intermediate_msgs:
                self._state.messages.append(msg)

            if first_user_msg:
                self._state.messages.append(first_user_msg)
                # Agent only sees user's content
                initial_messages.append(self._to_chat_format(first_user_msg))
            elif intermediate_msgs:
                # Find last user message with content
                for msg in reversed(intermediate_msgs):
                    if isinstance(msg, UserMessage) and msg.content:
                        initial_messages.append(self._to_chat_format(msg))
                        break

        observation = {
            "messages": initial_messages,
            "tools": self._get_tool_specs(),
            "policy": self._env.get_policy(),
        }

        info = self._get_info()

        return observation, info

    def step(self, action: Action) -> tuple[dict, float, bool, bool, dict]:
        """
        Action 실행.

        Args:
            action: ToolCallAction 리스트 또는 TextAction

        Returns:
            observation: {"messages": [...]}  # 새로 추가된 메시지들
            reward: 보상
            terminated: 종료 여부
            truncated: truncation 여부
            info: 추가 정보
        """
        if self._state is None:
            raise RuntimeError("reset()을 먼저 호출하세요.")

        if self._state.terminated:
            return {"messages": []}, 0.0, True, False, self._get_info()

        self._state.step_count += 1
        new_messages: list[dict] = []

        # Action 처리
        if isinstance(action, TextAction):
            # Text action: user에게 메시지 전송
            assistant_msg = AssistantMessage(
                role="assistant",
                content=action.content,
                tool_calls=None,
            )
            self._state.messages.append(assistant_msg)

            # Check for stop
            if self._is_stop_message(assistant_msg):
                self._state.terminated = True
                return (
                    {"messages": []},
                    self._get_reward(),
                    True,
                    False,
                    self._get_info(),
                )

            # User response
            # User's tool calls are processed internally in _get_user_message()
            # but saved to main state for multi-turn context
            # Agent only sees user's final content (information symmetry)
            if not self.solo_mode:
                final_user_msg, intermediate_msgs = self._get_user_message()

                # Save all intermediate messages to main state
                # These include: [UserMessage with tool_calls, ToolMessage, ToolMessage, ...]
                # Order matters for tool_use -> tool_result relationship
                for msg in intermediate_msgs:
                    self._state.messages.append(msg)

                # Save final user message if it wasn't already added as intermediate
                if final_user_msg:
                    self._state.messages.append(final_user_msg)

                # Find the last user message with content for agent observation
                last_user_content = None
                for msg in reversed(intermediate_msgs):
                    if isinstance(msg, UserMessage) and msg.content:
                        last_user_content = msg
                        break
                if not last_user_content and final_user_msg and final_user_msg.content:
                    last_user_content = final_user_msg

                if last_user_content:
                    # Agent only sees user's content (not tool_calls or ToolMessages)
                    new_messages.append(self._to_chat_format(last_user_content))

                    # Check for user stop
                    if self._is_user_stop(last_user_content):
                        self._state.terminated = True

            self._state.turn_idx += 1

        elif isinstance(action, list) and all(
            isinstance(a, ToolCallAction) for a in action
        ):
            # Tool call action
            tool_calls = action

            # Check for special tools (done, transfer) - handle like tau2
            # done() and transfer_to_human_agents are transformed into stop messages
            has_done = any(tc.name == "done" for tc in tool_calls)
            has_transfer = any(
                tc.name == "transfer_to_human_agents" for tc in tool_calls
            )

            if has_done:
                # done() tool: Transform to stop message (no tool call, no tool response)
                # tau2 transforms done() to content="###STOP###" and tool_calls=None
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="###STOP###",
                    tool_calls=None,
                )
                self._state.messages.append(assistant_msg)
                self._state.terminated = True

            elif has_transfer:
                # transfer_to_human_agents: Also a termination signal
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="###TRANSFER###",
                    tool_calls=None,
                )
                self._state.messages.append(assistant_msg)
                self._state.terminated = True

            else:
                # Regular tool calls
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                            requestor="assistant",
                        )
                        for tc in tool_calls
                    ],
                )
                self._state.messages.append(assistant_msg)

                # Execute each tool call
                for tc in tool_calls:
                    tool_call = ToolCall(
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        requestor="assistant",
                    )

                    # Execute tool
                    tool_response = self._env.get_response(tool_call)
                    self._state.messages.append(tool_response)

                    # Add to new_messages
                    new_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": tool_response.content or "",
                        }
                    )
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        # Check truncation
        truncated = self._state.step_count >= self.max_steps
        if truncated:
            self._state.terminated = True

        observation = {"messages": new_messages}
        reward = self._get_reward() if self._state.terminated else 0.0

        return (
            observation,
            reward,
            self._state.terminated,
            truncated,
            self._get_info(),
        )

    def get_context_messages(self, include_system: bool = False) -> list[dict]:
        """
        현재까지의 전체 메시지 히스토리를 chat template 형식으로 반환.

        Agent용 context:
        - 첫 번째 AssistantMessage (시스템 인사) 제외
        - User의 ToolMessages 제외 (정보 대칭)
        - User의 tool_calls만 있는 메시지 제외 (content 없으면 스킵)
        - Agent sees: agent's (content, tool_calls, tool_resp) + user's content
        - User's tool_calls and ToolMessages are NOT visible to agent

        Args:
            include_system: True이면 system message (policy)를 포함

        Returns:
            Chat template 호환 메시지 리스트
        """
        if self._state is None:
            return []

        result = []

        # Optionally include system message (policy)
        if include_system:
            policy = self._env.get_policy()
            if policy:
                result.append({"role": "system", "content": policy})

        skip_first_assistant = True  # Skip system-provided greeting

        for msg in self._state.messages:
            # Skip first assistant message (system greeting)
            if skip_first_assistant and isinstance(msg, AssistantMessage):
                skip_first_assistant = False
                continue

            if isinstance(msg, ToolMessage):
                # Only include agent's tool responses, not user's
                if msg.requestor == "user":
                    continue
            elif isinstance(msg, UserMessage):
                # Skip user messages that ONLY have tool_calls (no content)
                # If user has both tool_calls AND content, include the content
                # (tool_calls will be stripped by _to_chat_format)
                if msg.tool_calls and not msg.content:
                    continue
            result.append(self._to_chat_format(msg))
        return result

    def get_user_system_prompt(self) -> str:
        """
        User simulator의 system prompt 반환.

        Returns:
            User simulator의 전체 system prompt (guidelines + scenario)
        """
        if self._user is None:
            return ""
        return self._user.system_prompt

    def get_agent_system_prompt(self) -> str:
        """
        Agent의 system prompt (policy) 반환.

        Returns:
            Agent의 domain policy
        """
        return self._env.get_policy()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _create_environment(self) -> Environment:
        """Environment 생성."""
        env_constructor = registry.get_env_constructor(self.domain)
        return env_constructor(solo_mode=self.solo_mode)

    def _get_task(self) -> Task:
        """Task 로드."""
        # 먼저 domain의 기본 task loader 시도
        try:
            tasks = registry.get_tasks_loader(self.domain)()
            for task in tasks:
                if task.id == self.task_id:
                    return task
        except Exception:
            pass

        # 관련 task set 검색
        info = registry.get_info()
        related_task_sets = [
            ts for ts in info.task_sets if ts.startswith(self.domain)
        ]
        for task_set in related_task_sets:
            if task_set == self.domain:
                continue
            try:
                tasks = registry.get_tasks_loader(task_set)()
                for task in tasks:
                    if task.id == self.task_id:
                        return task
            except Exception:
                pass

        raise ValueError(
            f"Task not found: {self.task_id} in domain {self.domain}"
        )

    def _create_user(self) -> UserSimulator:
        """User simulator 생성."""
        if self.solo_mode:
            return DummyUser()

        try:
            user_tools = self._env.get_user_tools()
        except ValueError:
            user_tools = None

        return UserSimulator(
            tools=user_tools,
            instructions=self._task.user_scenario,
            llm=self.user_llm,
            llm_args=self.user_llm_args,
        )

    def _apply_initial_state(self):
        """Task의 initial_state 적용."""
        initial_state = self._task.initial_state
        if initial_state is None:
            return

        # Message history
        if initial_state.message_history:
            for msg_dict in initial_state.message_history:
                msg = self._dict_to_message(msg_dict)
                if msg:
                    self._state.messages.append(msg)

        # Initialization data
        if initial_state.initialization_data:
            self._env.set_state(
                initialization_data=initial_state.initialization_data,
                initialization_actions=None,
                message_history=[],
            )

        # Initialization actions
        if initial_state.initialization_actions:
            self._env.run_env_function_calls(
                initial_state.initialization_actions
            )


    def _get_user_message(self) -> tuple[Optional[UserMessage], list[Message]]:
        """
        User simulator로부터 응답 메시지 생성 (대화 중).

        User의 tool calls는 내부적으로 처리되지만, 정보는 main state에 저장됩니다:
        1. User simulator가 tool_calls를 반환하면 tool 실행
        2. Tool 결과를 user simulator에게 다시 전달
        3. User simulator가 최종 content를 생성할 때까지 반복
        4. 정보 대칭 (handled by monkey-patched flip_roles):
           - Agent: user의 content만 봄 (tool_calls, ToolMessages 안 봄)
           - User simulator: user의 이전 tool_calls, ToolMessages 봄 (멀티턴 컨텍스트)

        Returns:
            (UserMessage, list[Message]): 최종 user 메시지와 중간 메시지들
            중간 메시지들 = [UserMessage with tool_calls, ToolMessage, ToolMessage, ...]
            이 순서대로 main state에 저장되어야 함
        """
        if self.solo_mode or self._user is None:
            return None, []

        from tau2.data_model.message import SystemMessage
        from tau2.user.base import UserState

        # 마지막 assistant 메시지 가져오기
        last_assistant_msg = None
        for msg in reversed(self._state.messages):
            if isinstance(msg, AssistantMessage):
                last_assistant_msg = msg
                break

        if last_assistant_msg is None:
            return None, []

        # All intermediate messages to be saved to main state
        # Format: [UserMessage with tool_calls, ToolMessage, ToolMessage, ...]
        # These must be saved in order to maintain tool_use -> tool_result relationship
        intermediate_messages: list[Message] = []

        # Loop until user returns content (not just tool_calls)
        max_tool_rounds = 5  # Safety limit
        for _ in range(max_tool_rounds):
            # Build messages for UserState
            # Information symmetry: user sees only user's tool_calls/ToolMessages + agent's content
            # Filter out agent's tool_calls and ToolMessages BEFORE flip_roles()
            messages = []
            for msg in self._state.messages:
                if isinstance(msg, AssistantMessage):
                    if msg.is_tool_call():
                        # Agent's tool_calls - user doesn't see
                        continue
                    # Agent's text content - user sees
                    messages.append(msg)
                elif isinstance(msg, ToolMessage):
                    if msg.requestor != "user":
                        # Agent's tool response - user doesn't see
                        continue
                    # User's tool response - user sees
                    messages.append(msg)
                else:
                    # UserMessage - user sees (including user's tool_calls)
                    messages.append(msg)

            # Add intermediate messages from previous iterations
            # This ensures ToolMessages have corresponding tool_use in previous message
            for msg in intermediate_messages:
                messages.append(msg)

            # System messages
            system_messages = []
            if self._task and self._task.user_scenario:
                system_messages.append(
                    SystemMessage(
                        role="system", content=str(self._task.user_scenario)
                    )
                )

            user_state = UserState(
                system_messages=system_messages,
                messages=messages,
            )

            # Save for debugging (messages before flip_roles)
            self._last_user_state_messages = deepcopy(messages)

            # Generate user response
            # flip_roles() is called internally by UserSimulator
            user_response, _ = self._user.generate_next_message(
                message=last_assistant_msg,
                state=user_state,
            )

            # Process tool_calls if present (regardless of content)
            if user_response.tool_calls:
                # First, add the UserMessage with tool_calls to intermediate
                # This is needed for tool_use -> tool_result relationship
                intermediate_messages.append(user_response)

                # Then execute tools and add ToolMessages
                for tc in user_response.tool_calls:
                    tool_call = ToolCall(
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        requestor="user",
                    )
                    tool_response = self._env.get_response(tool_call)
                    intermediate_messages.append(tool_response)

            # Check if user has content (final response)
            if user_response.content:
                # User has final content
                # If user_response was already added to intermediate (has tool_calls),
                # return None as final_msg to avoid duplicate
                if user_response.tool_calls:
                    return None, intermediate_messages
                else:
                    return user_response, intermediate_messages

            # No content yet - if we had tool_calls, continue loop to get final response
            if user_response.tool_calls:
                continue

            # No content and no tool_calls - edge case
            return UserMessage(role="user", content=""), intermediate_messages

        # Safety: max rounds exceeded
        logger.warning(f"User tool call loop exceeded {max_tool_rounds} rounds")
        return UserMessage(role="user", content=""), intermediate_messages

    def _execute_user_tool_calls(self, user_msg: UserMessage) -> None:
        """Execute user's tool calls and add ToolMessages to state."""
        if not user_msg.tool_calls:
            return

        for tc in user_msg.tool_calls:
            tool_call = ToolCall(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments,
                requestor="user",
            )
            tool_response = self._env.get_response(tool_call)
            self._state.messages.append(tool_response)

    def _get_tool_specs(self) -> list[dict]:
        """Tool 정의를 chat template 형식으로 변환."""
        tools = self._env.get_tools()
        specs = []
        for tool in tools:
            spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (
                        tool._get_description()
                        if hasattr(tool, "_get_description")
                        else (tool.short_desc or "")
                    ),
                    "parameters": (
                        tool.params.model_json_schema()
                        if hasattr(tool, "params")
                        else {"type": "object", "properties": {}}
                    ),
                },
            }
            specs.append(spec)

        # Add special tools (like GymAgent does)
        # done: Agent calls this when task is complete
        specs.append({
            "type": "function",
            "function": {
                "name": "done",
                "description": "Call this function when you have completed the task and the customer is satisfied.",
                "parameters": {"type": "object", "properties": {}},
            },
        })

        return specs

    def _to_chat_format(self, msg: Message) -> dict:
        """tau2 Message를 chat template 형식으로 변환."""
        if isinstance(msg, UserMessage):
            # Agent should only see user's content, not tool_calls
            # User's tool_calls are processed internally in _get_user_message()
            return {"role": "user", "content": msg.content or ""}
        elif isinstance(msg, AssistantMessage):
            result = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return result
        elif isinstance(msg, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.id or "",
                "name": getattr(msg, "name", "tool"),
                "content": msg.content or "",
            }
        else:
            return {"role": msg.role, "content": msg.content or ""}

    def _dict_to_message(self, msg_dict: dict) -> Optional[Message]:
        """Dict를 tau2 Message로 변환."""
        role = msg_dict.get("role")
        if role == "user":
            return UserMessage(role="user", content=msg_dict.get("content", ""))
        elif role == "assistant":
            tool_calls = None
            if msg_dict.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get(
                            "name", tc.get("function", {}).get("name", "")
                        ),
                        arguments=tc.get(
                            "arguments",
                            tc.get("function", {}).get("arguments", {}),
                        ),
                        requestor="assistant",
                    )
                    for tc in msg_dict["tool_calls"]
                ]
            return AssistantMessage(
                role="assistant",
                content=msg_dict.get("content"),
                tool_calls=tool_calls,
            )
        elif role == "tool":
            return ToolMessage(
                id=msg_dict.get("tool_call_id", msg_dict.get("id", "")),
                content=msg_dict.get("content", ""),
                role="tool",
            )
        return None

    def _is_stop_message(self, msg: AssistantMessage) -> bool:
        """Assistant 메시지가 종료 메시지인지 확인."""
        if msg.content and "###STOP###" in msg.content:
            return True
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name == "done":
                    return True
        return False

    def _is_user_stop(self, msg: UserMessage) -> bool:
        """User 메시지가 종료 신호인지 확인."""
        if msg.content is None:
            return False
        stop_tokens = ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]
        return any(token in msg.content for token in stop_tokens)

    def _get_reward(self) -> float:
        """보상 계산."""
        if self._state is None or self._task is None or self._env is None:
            return 0.0

        from datetime import datetime

        from tau2.data_model.simulation import TerminationReason

        now = datetime.now().isoformat()

        # SimulationRun 생성
        simulation_run = SimulationRun(
            id=f"slime_{self._task.id}_{now}",
            task_id=self._task.id,
            timestamp=now,
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=deepcopy(self._state.messages),
        )

        try:
            result = evaluate_simulation(
                simulation=simulation_run,
                task=self._task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=self.solo_mode,
                domain=self.domain,
            )
            return result.reward
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    def _get_info(self) -> dict:
        """Info dict 반환."""
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "step_count": self._state.step_count if self._state else 0,
            "turn_idx": self._state.turn_idx if self._state else 0,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def make_tool_call_action(
    name: str, arguments: dict[str, Any], call_id: str
) -> ToolCallAction:
    """ToolCallAction 생성 헬퍼."""
    return ToolCallAction(name=name, arguments=arguments, id=call_id)


def make_text_action(content: str) -> TextAction:
    """TextAction 생성 헬퍼."""
    return TextAction(content=content)
