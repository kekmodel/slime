"""
SLIME-tau2 통합을 위한 메시지 유틸리티.

제공 기능:
- tau2 Message 타입 ↔ chat template dict 변환
- OpenAI 형식의 tool specification 생성
- 멀티턴 대화의 토큰 델타 계산
- 턴 간 think 태그 제거
"""

import logging
import re
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.tools import to_functional_format
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Tau-bench 스타일 tool 지시문 (reformulation용)
TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


def tau2_message_to_dict(msg: Message) -> dict[str, Any]:
    """
    tau2 Message를 chat template dict로 변환.

    Args:
        msg: tau2 Message 객체 (UserMessage, AssistantMessage, ToolMessage 등)

    Returns:
        tokenizer.apply_chat_template()에 적합한 dict
    """
    if isinstance(msg, UserMessage):
        if msg.is_tool_call():
            tool_calls_str = ", ".join(
                [to_functional_format(t) for t in msg.tool_calls]
            )
            return {"role": "user", "content": tool_calls_str}
        return {"role": "user", "content": msg.content}

    elif isinstance(msg, AssistantMessage):
        if msg.is_tool_call():
            tool_calls_str = ", ".join(
                [to_functional_format(t) for t in msg.tool_calls]
            )
            if msg.content:
                return {
                    "role": "assistant",
                    "content": f"{msg.content}\n{tool_calls_str}",
                }
            return {"role": "assistant", "content": tool_calls_str}
        return {"role": "assistant", "content": msg.content}

    elif isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "name": msg.name,
            "content": msg.content,
        }

    else:
        return {"role": msg.role, "content": msg.content}


def dict_to_tau2_message(msg_dict: dict[str, Any]) -> Message:
    """
    Chat template dict를 tau2 Message로 변환.

    Args:
        msg_dict: role과 content 키를 가진 dict

    Returns:
        적절한 tau2 Message 서브클래스
    """
    role = msg_dict.get("role", "user")
    content = msg_dict.get("content", "")

    if role == "user":
        return UserMessage(role="user", content=content)
    elif role == "assistant":
        return AssistantMessage(role="assistant", content=content)
    elif role == "tool":
        return ToolMessage(
            role="tool",
            name=msg_dict.get("name", "tool"),
            content=content,
        )
    elif role == "system":
        # System 메시지는 일반적으로 chat template용 dict로 유지
        return UserMessage(role="system", content=content)
    else:
        return UserMessage(role=role, content=content)


def build_tool_specs(tools: list[Tool]) -> list[dict[str, Any]]:
    """
    tau2 Tool 객체를 OpenAI tool spec 형식으로 변환.

    Args:
        tools: tau2 Tool 객체 리스트

    Returns:
        OpenAI 형식의 tool specification 리스트
    """
    tool_specs = []
    for tool in tools:
        # parameters 스키마 생성
        if tool.params is not None:
            parameters = tool.params.model_json_schema()
        else:
            parameters = {"type": "object", "properties": {}}

        spec = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.short_desc or tool.name,
                "parameters": parameters,
            },
        }
        tool_specs.append(spec)

    return tool_specs


def get_token_delta(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[int]]:
    """
    멀티턴 대화의 토큰 델타 계산.

    대화에서 마지막 메시지의 토큰 ID와 loss mask를 계산.
    assistant 응답(학습 대상)과 user/tool 응답(비학습 대상)을 구분 처리.

    참조: tau-bench trainable_agents.py, verl 문서
    https://verl.readthedocs.io/en/v0.4.1/sglang_multiturn/multiturn.html

    Args:
        tokenizer: HuggingFace tokenizer
        messages: 대화 메시지 리스트 (dict 형태)
        tools: tool specification 리스트 (선택)

    Returns:
        (token_ids, loss_mask) 튜플
        - token_ids: 마지막 메시지의 토큰 ID들
        - loss_mask: 학습 대상 토큰은 1 (assistant), 그 외 0
    """
    if not messages:
        return [], []

    # 현재 메시지에 chat template 적용
    curr = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
        tools=tools,
    )

    token_ids = []
    loss_mask = []

    last_role = messages[-1].get("role", "user")

    if last_role == "assistant":
        # 케이스 1: 마지막 메시지가 assistant 응답 (학습 대상)
        prev = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            tokenize=False,
            tools=tools,
        )
        # 새로 추가된 부분 (델타) 추출
        delta_text = curr[len(prev) :]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)
        token_ids = new_tokens
        loss_mask = [1] * len(new_tokens)  # 학습 대상

    else:
        # 케이스 2: 마지막 메시지가 user/tool 응답 (비학습 대상)
        prev = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=False,
            tokenize=False,
            tools=tools,
        )
        # 새로 추가된 부분 (델타) 추출
        delta_text = curr[len(prev) :]
        new_tokens = tokenizer.encode(delta_text, add_special_tokens=False)
        token_ids = new_tokens
        loss_mask = [0] * len(new_tokens)  # 비학습 대상

    return token_ids, loss_mask


def reformulate_tool_instruction(text: str) -> str:
    """
    tau2 환경용 tool call 지시문 재구성.

    일부 모델의 기본 tool 템플릿은 하나 이상의 함수 호출을 가정하지만,
    tau2에서는 최대 하나의 tool call 또는 tool call 없음만 유효.

    Args:
        text: 원본 프롬프트 텍스트

    Returns:
        재구성된 프롬프트 텍스트
    """
    return text.replace(
        "You may call one or more functions to assist with the user query.",
        TOOL_INSTRUCTION,
    )


def format_observation(messages: list[Message], last_only: bool = True) -> str:
    """
    메시지를 관측 문자열로 포맷팅.

    Args:
        messages: tau2 Message 객체 리스트
        last_only: True면 마지막 메시지 내용만 반환

    Returns:
        포맷팅된 관측 문자열
    """
    if not messages:
        return ""

    if last_only:
        last_msg = messages[-1]
        if isinstance(last_msg, AssistantMessage):
            if last_msg.is_tool_call():
                return ", ".join(
                    [to_functional_format(t) for t in last_msg.tool_calls]
                )
            return last_msg.content
        return last_msg.content

    turns = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            if msg.is_tool_call():
                tool_calls = ", ".join(
                    [to_functional_format(t) for t in msg.tool_calls]
                )
                turns.append(f"user: {tool_calls}")
            else:
                turns.append(f"user: {msg.content}")
        elif isinstance(msg, AssistantMessage):
            if msg.is_tool_call():
                tool_calls = ", ".join(
                    [to_functional_format(t) for t in msg.tool_calls]
                )
                turns.append(f"assistant: {tool_calls}")
            else:
                turns.append(f"assistant: {msg.content}")
        else:
            turns.append(f"{msg.role}: {msg.content}")

    return "\n".join(turns)


def prepare_initial_messages(
    policy: str,
    initial_observation: str,
) -> list[dict[str, Any]]:
    """
    초기 대화 메시지 준비.

    Args:
        policy: 시스템 정책/지시문
        initial_observation: 초기 user 메시지 (환경으로부터)

    Returns:
        chat template용 초기 메시지 리스트
    """
    return [
        {"role": "system", "content": policy},
        {"role": "user", "content": initial_observation},
    ]


def prepare_prompt_tokens(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    reformulate: bool = True,
) -> tuple[str, list[int]]:
    """
    프롬프트 텍스트 준비 및 토큰화.

    Args:
        tokenizer: HuggingFace tokenizer
        messages: 대화 메시지들
        tools: tool specification (선택)
        reformulate: tool 지시문 재구성 여부

    Returns:
        (prompt_text, prompt_token_ids) 튜플
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
    )

    # 선택적으로 tool 지시문 재구성
    if reformulate:
        prompt_text = reformulate_tool_instruction(prompt_text)

    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)[
        "input_ids"
    ]

    return prompt_text, prompt_token_ids


def clean_response(response: str, end_token: str = "<|im_end|>") -> str:
    """
    end 토큰 제거로 LLM 응답 정리.

    Args:
        response: 원본 응답 텍스트
        end_token: 제거할 end 토큰

    Returns:
        정리된 응답 텍스트
    """
    if response.endswith(end_token):
        return response[: -len(end_token)]
    return response


# 모델별 think 태그 패턴
THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>\s*", re.DOTALL),  # Qwen3-Thinking
    re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL),  # 대체 형식
]


def strip_think_content(content: str) -> str:
    """
    내용에서 think/reasoning 태그 제거.

    새 user 요청 전에 이전 턴의 thinking을 정리하는 데 사용.
    현재 턴의 thinking은 학습용으로 유지.

    Args:
        content: think 태그가 포함될 수 있는 메시지 내용

    Returns:
        think 태그가 제거된 내용
    """
    if not content:
        return content

    result = content
    for pattern in THINK_PATTERNS:
        result = pattern.sub("", result)

    return result.strip()


def strip_think_from_previous_turns(
    messages: list[dict[str, Any]],
    current_turn_start: int | None = None,
) -> list[dict[str, Any]]:
    """
    이전 턴의 assistant 메시지에서 think 내용 제거.

    RL 학습에서:
    - 현재 턴의 think → 유지 (loss_mask=1, 학습 대상)
    - 이전 턴의 think → 컨텍스트에서 제거

    Args:
        messages: 대화 메시지 리스트
        current_turn_start: 현재 턴 시작 인덱스.
                           None이면 마지막 user 메시지를 찾음.

    Returns:
        이전 턴의 think 내용이 제거된 메시지들
    """
    if not messages:
        return messages

    # 현재 턴 시작점 찾기 (assistant 응답 전 마지막 user 메시지)
    if current_turn_start is None:
        current_turn_start = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                current_turn_start = i
                break

    # 메시지 처리
    result = []
    for i, msg in enumerate(messages):
        if i < current_turn_start and msg.get("role") == "assistant":
            # 이전 턴의 assistant 메시지 - think 제거
            new_msg = msg.copy()
            content = msg.get("content", "")
            new_msg["content"] = strip_think_content(content)
            result.append(new_msg)
        else:
            # 현재 턴 또는 non-assistant - 그대로 유지
            result.append(msg)

    return result


def has_think_content(content: str) -> bool:
    """
    내용에 think 태그가 있는지 확인.

    Args:
        content: 확인할 메시지 내용

    Returns:
        think 태그가 있으면 True
    """
    if not content:
        return False

    for pattern in THINK_PATTERNS:
        if pattern.search(content):
            return True

    return False


def extract_think_content(content: str) -> tuple[str, str]:
    """
    think 내용과 나머지 내용을 분리 추출.

    Args:
        content: think 태그가 있을 수 있는 메시지 내용

    Returns:
        (think_content, remaining_content) 튜플
    """
    if not content:
        return "", ""

    think_parts = []
    remaining = content

    for pattern in THINK_PATTERNS:
        matches = pattern.findall(remaining)
        think_parts.extend(matches)
        remaining = pattern.sub("", remaining)

    think_content = "\n".join(think_parts).strip()
    remaining = remaining.strip()

    return think_content, remaining


# 함수 형식 패턴: function_name(args) 또는 function_name()
# 매칭 예시: search_flights(origin='NYC'), refresh(), get_user(id=123)
TOOL_CALL_PATTERN = re.compile(
    r"\b\w+\s*\([^)]*\)\s*,?\s*",  # function_name(...) 선택적 trailing comma
    re.DOTALL,
)


def strip_tool_call_content(content: str) -> str:
    """
    내용에서 tool call 패턴 제거.

    함수 형식의 tool call 예시:
    - function_name()
    - search_flights(origin='NYC', destination='LAX')
    - get_user(id=123), get_balance()

    Args:
        content: tool call이 포함될 수 있는 메시지 내용

    Returns:
        tool call이 제거된 내용
    """
    if not content:
        return content

    # 함수 형식 tool call 제거
    result = TOOL_CALL_PATTERN.sub("", content)

    return result.strip()


def clean_user_content(content: str) -> str:
    """
    agent 학습 데이터용 user 메시지 내용 정리.

    제거 대상:
    1. Think/reasoning 태그 (user LLM도 thinking 가능)
    2. Tool call 패턴 (user도 tool 사용 가능)

    Agent 학습 데이터에는 user의 자연어만 포함되어야 함.

    Args:
        content: 원본 user 메시지 내용

    Returns:
        자연어만 남은 정리된 내용
    """
    if not content:
        return content

    # 1단계: think 태그 제거
    result = strip_think_content(content)

    # 2단계: tool call 제거
    result = strip_tool_call_content(result)

    return result.strip()


def strip_role_prefix(observation: str) -> str:
    """
    tau2 gym observation에서 role prefix 제거.

    tau2 gym은 observation을 "role: content" 형식으로 반환.
    예: "user: Hi there!" -> "Hi there!"
         "assistant: Hello!" -> "Hello!"

    Args:
        observation: tau2 gym의 observation 문자열

    Returns:
        role prefix가 제거된 순수 content
    """
    if not observation:
        return observation

    # 단일 메시지의 role prefix 제거
    prefixes = ["user: ", "assistant: ", "tool: ", "system: "]
    for prefix in prefixes:
        if observation.startswith(prefix):
            return observation[len(prefix):]

    return observation


def parse_observation_messages(observation: str) -> list[dict[str, str]]:
    """
    tau2 gym observation 문자열을 메시지 리스트로 파싱.

    Multi-line observation 형식:
        user: Hello
        assistant: Hi there!
        user: How are you?

    Args:
        observation: tau2 gym의 observation 문자열

    Returns:
        [{"role": "user", "content": "Hello"}, ...]
    """
    if not observation:
        return []

    messages = []
    current_role = None
    current_content = []

    for line in observation.split("\n"):
        # 새 메시지 시작 감지
        role_found = None
        for role in ["user", "assistant", "tool", "system"]:
            prefix = f"{role}: "
            if line.startswith(prefix):
                # 이전 메시지 저장
                if current_role is not None:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content),
                    })
                # 새 메시지 시작
                current_role = role
                current_content = [line[len(prefix):]]
                role_found = role
                break

        if role_found is None and current_role is not None:
            # 이전 메시지의 연속
            current_content.append(line)

    # 마지막 메시지 저장
    if current_role is not None:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content),
        })

    return messages
