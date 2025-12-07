"""
tau2 Gymnasium 환경을 사용한 SLIME 강화학습 통합.

새로운 SlimeGymEnv를 사용하여 SLIME 학습을 수행함.
test_live_episode_v2.py에서 검증된 로직 기반.

주요 특징:
- ToolCallAction/TextAction 기반의 구조화된 action
- 정확한 토큰 및 loss_mask 계산
- tau2 환경 및 평가 통합
- done tool을 통한 에피소드 종료
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .message_utils import get_token_delta, extract_think_content
from .slime_env import SlimeGymEnv, ToolCallAction, TextAction
from .tool_parser import create_tool_adapter

# 환경 변수 로드
load_dotenv()

# tau2 로그 레벨 조정
logging.getLogger("tau2").setLevel(logging.WARNING)


# =============================================================================
# 환경 변수 헬퍼 함수
# =============================================================================

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _get_thinking_field(tokenizer_name: str) -> str:
    """
    Get the correct field name for reasoning/thinking based on tokenizer.

    - Qwen: uses 'reasoning_content' → rendered as <think>...</think>
    - gpt-oss: uses 'thinking' → rendered as <|channel|>analysis
    """
    name_lower = tokenizer_name.lower()
    if "qwen" in name_lower:
        return "reasoning_content"
    elif "gpt-oss" in name_lower:
        return "thinking"
    else:
        # Default to reasoning_content
        return "reasoning_content"


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default).lower()).lower() == "true"


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


# =============================================================================
# done tool 정의 (GymAgent와 동일)
# =============================================================================

def done() -> str:
    """Call this function when you have completed the task and the customer is satisfied."""
    return "###STOP###"


# =============================================================================
# 메인 generate 함수
# =============================================================================

async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    tau2 SlimeGymEnv를 사용하여 trajectory 생성.

    SLIME rollout 흐름:
    1. SlimeGymEnv 초기화
    2. chat template과 tools로 초기 프롬프트 구성
    3. sglang LLM 호출로 멀티턴 에피소드 실행
    4. tool call 파싱 후 환경 step
    5. 토큰과 loss_mask 정확하게 추적
    6. trajectory를 SLIME Sample 형식으로 변환

    Args:
        args: sglang 연결 정보 포함 SLIME 인자
        sample: prompt 또는 metadata에 task_id 포함된 입력 샘플
        sampling_params: LLM 샘플링 파라미터

    Returns:
        tokens, loss_mask, reward, metadata가 포함된 Sample
    """
    # partial rollout은 tau2 상호작용에서 지원 안 함
    assert (
        not args.partial_rollout
    ), "Partial rollout is not supported for tau2 interactions."

    # sglang 상태 및 URL 초기화
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Tokenizer에 따른 thinking field 결정
    tokenizer_name = getattr(state.tokenizer, "name_or_path", "")
    thinking_field = _get_thinking_field(tokenizer_name)
    is_gpt_oss = "gpt-oss" in tokenizer_name.lower()

    logger.debug(f"Tokenizer: {tokenizer_name}, thinking_field: {thinking_field}")

    # 샘플에서 task ID 추출
    task_id = sample.metadata.get("task_id") or sample.prompt

    # 환경 변수에서 설정 읽기
    domain = _env_str("TAU2_DOMAIN", "telecom")
    max_steps = _env_int("TAU2_MAX_STEPS", 100)
    max_turns = _env_int("TAU2_MAX_TURNS", 30)
    solo_mode = _env_bool("TAU2_SOLO_MODE", False)
    user_llm = _env_str("TAU2_USER_LLM", DEFAULT_LLM_USER)
    user_temperature = _env_float(
        "TAU2_USER_TEMP", DEFAULT_LLM_ARGS_USER.get("temperature", 0.0)
    )
    tool_parser_type = _env_str("TAU2_TOOL_PARSER", "qwen")
    return_logprob = _env_bool("TAU2_RETURN_LOGPROB", False)

    logger.info(f"에피소드 시작: task={task_id}, domain={domain}")

    # =================================================================
    # 환경 초기화
    # =================================================================
    env = SlimeGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        user_llm=user_llm,
        user_llm_args={"temperature": user_temperature},
        solo_mode=solo_mode,
    )

    # 환경 리셋
    obs, _ = env.reset()

    policy = obs.get("policy", "")
    tool_specs = obs.get("tools", [])  # 이미 done tool 포함됨
    initial_messages = obs.get("messages", [])

    logger.info(f"Policy length: {len(policy)}")
    logger.info(f"Tools: {[t['function']['name'] for t in tool_specs]}")

    # =================================================================
    # Context 초기화
    # =================================================================
    context_messages: list[dict[str, Any]] = [
        {"role": "system", "content": policy},
    ]
    context_messages.extend(initial_messages)

    # Tool adapter 생성 (파싱 테스트용)
    tool_adapter = create_tool_adapter(tool_specs, tool_parser_type)

    # =================================================================
    # 추적 변수
    # =================================================================
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    rollout_log_probs: list[float] = [] if return_logprob else None
    tool_call_idx = 0

    terminated = False
    step_count = 0
    turn_count = 0
    total_reward = 0.0

    # =================================================================
    # 멀티턴 상호작용 루프
    # =================================================================
    while not terminated and turn_count < max_turns:
        turn_count += 1

        # 현재 context로 프롬프트 구성
        current_prompt = state.tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tool_specs,
        )

        # sglang 요청 페이로드
        payload = {
            "text": current_prompt,
            "sampling_params": sampling_params,
        }
        if return_logprob:
            payload["return_logprob"] = True

        # sglang 호출
        output = await post(url, payload)

        # 중단 확인
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            logger.warning(f"sglang 중단됨 (turn={turn_count})")
            sample.status = Sample.Status.ABORTED
            sample.reward = total_reward
            return sample

        # Assistant 응답 추출
        assistant_response = output["text"]

        # Tool call 파싱
        parsed = tool_adapter.parse(assistant_response)

        if not parsed.success:
            logger.debug(f"Tool 파싱 실패 (turn={turn_count}): {parsed.error}")

        # =============================================================
        # Action 처리
        # =============================================================
        if parsed.success and parsed.calls:
            # Tool call action
            actions = []
            for call in parsed.calls:
                call_id = f"call_{tool_call_idx}"
                tool_call_idx += 1
                actions.append(
                    ToolCallAction(
                        name=call["name"],
                        arguments=call.get("parameters", {}),
                        id=call_id,
                    )
                )

            # Context에 assistant 메시지 추가
            # Extract thinking from response
            thinking_text, normal_text = extract_think_content(
                parsed.normal_text or ""
            )

            tool_calls_data = [
                {
                    "id": a.id,
                    "type": "function",
                    "function": {
                        "name": a.name,
                        "arguments": json.dumps(a.arguments),
                    },
                }
                for a in actions
            ]

            # gpt-oss 제약: tool_calls + thinking 시 content는 빈 문자열이어야 함
            if is_gpt_oss and thinking_text:
                # thinking에 normal_text 포함 (있다면)
                combined_thinking = thinking_text
                if normal_text:
                    combined_thinking = f"{normal_text}\n\n{thinking_text}"
                assistant_msg = {
                    "role": "assistant",
                    "content": "",  # gpt-oss: must be empty with tool_calls + thinking
                    "tool_calls": tool_calls_data,
                    "thinking": combined_thinking,
                }
            else:
                # Qwen 및 기타: reasoning_content 사용
                assistant_msg = {
                    "role": "assistant",
                    "content": normal_text,
                    "tool_calls": tool_calls_data,
                }
                if thinking_text:
                    assistant_msg[thinking_field] = thinking_text

            context_messages.append(assistant_msg)

            # Assistant 토큰 계산 (loss_mask=1, 학습 대상)
            token_ids, mask = get_token_delta(
                state.tokenizer, context_messages, tool_specs
            )
            response_token_ids.extend(token_ids)
            loss_masks.extend(mask)  # assistant → [1, 1, 1, ...]

            # 환경 step
            obs, reward, terminated, _, _ = env.step(actions)
            step_count += 1
            total_reward = reward

            # Tool response를 context에 추가 (각각 토큰 계산)
            for msg in obs["messages"]:
                if msg.get("role") == "tool":
                    context_messages.append(msg)
                    # Tool response 토큰 (loss_mask=0, 비학습 대상)
                    token_ids, mask = get_token_delta(
                        state.tokenizer, context_messages, tool_specs
                    )
                    response_token_ids.extend(token_ids)
                    loss_masks.extend(mask)  # tool → [0, 0, 0, ...]

        else:
            # Text action
            action = TextAction(content=assistant_response)

            # Context에 assistant 메시지 추가
            if assistant_response:
                # Extract thinking from response
                thinking_text, normal_text = extract_think_content(
                    assistant_response
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": normal_text,
                }
                # Add thinking field (reasoning_content for Qwen, thinking for gpt-oss)
                if thinking_text:
                    assistant_msg[thinking_field] = thinking_text
                context_messages.append(assistant_msg)

                # Assistant 토큰 계산 (loss_mask=1, 학습 대상)
                token_ids, mask = get_token_delta(
                    state.tokenizer, context_messages, tool_specs
                )
                response_token_ids.extend(token_ids)
                loss_masks.extend(mask)  # assistant → [1, 1, 1, ...]

            # 환경 step
            obs, reward, terminated, _, _ = env.step(action)
            step_count += 1
            total_reward = reward

            # User response를 context에 추가
            for msg in obs["messages"]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    if not user_content:
                        continue
                    context_messages.append(msg)
                    # User response 토큰 (loss_mask=0, 비학습 대상)
                    token_ids, mask = get_token_delta(
                        state.tokenizer, context_messages, tool_specs
                    )
                    response_token_ids.extend(token_ids)
                    loss_masks.extend(mask)  # user → [0, 0, 0, ...]

        logger.debug(
            f"Turn {turn_count}: terminated={terminated}, reward={reward}"
        )

    # =================================================================
    # 최종 샘플 구성
    # =================================================================

    # Prompt 토큰 계산
    initial_prompt = state.tokenizer.apply_chat_template(
        [{"role": "system", "content": policy}] + initial_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tool_specs,
    )
    prompt_token_ids = state.tokenizer.encode(initial_prompt, add_special_tokens=False)

    # Response 텍스트 구성
    response_parts = []
    for msg in context_messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                response_parts.append(content)

    # Sample 설정
    sample.tokens = prompt_token_ids + response_token_ids
    sample.loss_mask = [0] * len(prompt_token_ids) + loss_masks
    sample.response = "\n\n".join(response_parts)
    sample.response_length = len(response_token_ids)
    sample.reward = total_reward

    if return_logprob and rollout_log_probs:
        sample.rollout_log_probs = rollout_log_probs

    # 상태 설정
    if terminated:
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    # 메타데이터
    sample.metadata["task_id"] = task_id
    sample.metadata["domain"] = domain
    sample.metadata["steps"] = step_count
    sample.metadata["num_turns"] = turn_count
    sample.metadata["context_messages"] = context_messages

    logger.info(
        f"Task {task_id}: reward={total_reward:.3f}, steps={step_count}, "
        f"turns={turn_count}, status={sample.status.value}"
    )

    return sample


async def generate_eval(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    평가용 generate 함수.

    generate와 동일하지만 평가용 설정 적용:
    - 다른 temperature 사용 가능 (보통 0으로 deterministic)
    - 평가 메타데이터 저장

    Args:
        args: SLIME 인자
        sample: 입력 샘플
        sampling_params: 샘플링 파라미터 (학습과 다를 수 있음)

    Returns:
        평가 결과가 포함된 Sample
    """
    result = await generate(args, sample, sampling_params)
    result.metadata["is_eval"] = True
    return result


async def reward_func(_args, sample: Sample, **_kwargs) -> float:
    """
    tau2 샘플용 reward 함수.

    reward는 generate 중 환경에서 이미 계산되었으므로,
    저장된 reward를 그대로 반환함.

    Args:
        _args: SLIME 인자 (미사용, 인터페이스 호환용)
        sample: reward가 이미 설정된 Sample
        _kwargs: 추가 인자 (미사용, 인터페이스 호환용)

    Returns:
        샘플의 reward 값
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    return sample.reward if sample.reward is not None else 0.0
