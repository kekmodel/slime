"""
Cross-validation test: chat/completions vs completions endpoint.

Validates that our tool parser and formatter work correctly for RL training.

Key principles:
1. chat/completions and completions are INDEPENDENT flows
2. completions uses OUR tool parser and OUR formatter
3. Generation prompt: True for user prompt, False after tool response
4. Compare final results between the two flows

Model Format Reference:
=======================

Tool Call End Tokens (EOS - stripped by provider):
  - gpt-oss: <|call|> (tool), <|return|> (regular) - BOTH stripped
  - glm-4.7: <|endoftext|> (regular), <|observation|> (after tool call), <|user|> (next turn)
  - Others: Standard EOS tokens, not included in tool call content

Tool Response Role Mapping (OpenAI role="tool" →):
  - qwen: <|im_start|>user + <tool_response> wrapper
  - gpt-oss: functions.{name} to=assistant (namespace)
  - kimi-k2: <|im_system|>tool
  - glm-4.7: <|observation|>
  - minimax: ]~b]tool

Thinking/Reasoning Handling:
  - gpt-oss: 'thinking' field → <|channel|>analysis channel
  - kimi-k2-thinking: 'reasoning_content' → <think> tags (history stripped)
  - qwen3/glm/minimax: 'reasoning_content' → <think> tags
"""

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# API Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if API_KEY is None:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is required. Set it before running this test: export OPENROUTER_API_KEY=your_key")
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
COMPLETIONS_URL = "https://openrouter.ai/api/v1/completions"

# Import our parsers and formatters
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from examples.tool_calling.generate import parse_tool_calls
from examples.tool_calling.tools import TOOL_RESPONSE_FORMATTERS

# Models to test: (model_id, sglang_parser, formatter_name, hf_model, providers)
MODELS = [
    ("qwen/qwen3-30b-a3b-thinking-2507", "qwen25", "qwen25", "Qwen/Qwen3-30B-A3B", ["Alibaba"]),
    ("qwen/qwen3-next-80b-a3b-thinking", "qwen25", "qwen25", "Qwen/Qwen3-Next-80B-A3B-Thinking", ["Alibaba"]),
    ("openai/gpt-oss-120b", "gpt-oss", "gpt-oss", "openai/gpt-oss-120b", ["Fireworks"]),
    (
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "qwen3_coder",
        "qwen3_coder",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ["NVIDIA"],
    ),
    ("moonshotai/kimi-k2-thinking", "kimi_k2", "kimi_k2", "moonshotai/Kimi-K2-Thinking", ["DeepInfra"]),
    ("z-ai/glm-4.7-flash", "glm47", "glm47", "zai-org/GLM-4.7-Flash", ["Z.AI"]),
    ("minimax/minimax-m2", "minimax-m2", "minimax-m2", "MiniMaxAI/MiniMax-M2.1", ["Minimax"]),
]

TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}},
            "required": ["expression"],
        },
    },
}

USER_PROMPT = "Use the calculator tool to calculate 15 + 27"

# Tool call patterns by parser (start pattern to detect tool call, end token to add back)
# Note: These are tool call end tokens only. For regular responses:
#   - gpt-oss uses <|return|> instead of <|call|>
#   - glm47 uses <|observation|> after tool call, <|endoftext|> for regular
#   - Other models use their standard EOS tokens
TOOL_CALL_PATTERNS = {
    "qwen25": ("<tool_call>", "</tool_call>"),
    "qwen3_coder": ("<tool_call>", "</tool_call>"),
    "gpt-oss": ("<|channel|>commentary to=", "<|call|>"),
    "minimax-m2": ("<minimax:tool_call>", "</minimax:tool_call>"),
    "kimi_k2": ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
    "glm47": ("<tool_call>", "</tool_call>"),
}

# Regular response end tokens (when NOT making tool calls)
# Only needed for models where EOS token is stripped by provider
REGULAR_END_TOKENS = {
    "gpt-oss": "<|return|>",
    # glm47: EOS tokens (<|endoftext|>, <|user|>, <|observation|>) are stripped
    #        but tool call content </tool_call> is included before EOS
}

# Generation prompt patterns (what gets added with add_generation_prompt=True)
GENERATION_PROMPTS = {
    "qwen25": "<|im_start|>assistant\n",
    "qwen3_coder": "<|im_start|>assistant\n",
    "gpt-oss": "<|start|>assistant",  # Model then outputs <|channel|>analysis or final
    "kimi_k2": "<|im_assistant|>assistant<|im_middle|>",  # Model outputs <think> first
    "glm47": "<|assistant|><think>",  # Thinking model starts with <think>
    "minimax-m2": "]~b]ai\n<think>\n",  # Thinking model starts with <think>
}

# Thinking format patterns
# Maps API response field to raw output format
THINKING_FORMATS = {
    "gpt-oss": {
        "api_field": "reasoning",  # Field name in chat/completions response
        "template_field": "thinking",  # Field name for apply_chat_template
        "output_format": "<|channel|>analysis<|message|>{thinking}<|end|>",
    },
    "kimi_k2": {
        "api_field": "reasoning",
        "template_field": "reasoning_content",
        "output_format": "<think>{thinking}</think>",  # Only in suffix, stripped in history
    },
    "qwen25": {
        "api_field": None,  # Thinking embedded in content
        "template_field": "reasoning_content",
        "output_format": "<think>{thinking}</think>",
    },
    "glm47": {
        "api_field": None,
        "template_field": "reasoning_content",
        "output_format": "<think>{thinking}</think>",
    },
    "minimax-m2": {
        "api_field": None,
        "template_field": "reasoning_content",
        "output_format": "<think>{thinking}</think>",
    },
}


def chat_completions(model_id: str, messages: List[Dict], providers: List[str], tools: Optional[List[Dict]] = None) -> Dict:
    """Call chat/completions endpoint."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "provider": {"order": providers},
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=180)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}


def completions(model_id: str, prompt: str, providers: List[str]) -> Dict:
    """Call completions endpoint with raw text."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "prompt": prompt,
        "provider": {"order": providers},
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024,  # Ensure enough tokens for tool calls
    }

    try:
        response = requests.post(COMPLETIONS_URL, headers=headers, json=payload, timeout=180)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}


def build_raw_prompt_with_hf(messages: List[Dict], hf_model: str, tools: List[Dict], add_generation_prompt: bool) -> str:
    """Build raw prompt using HuggingFace tokenizer.

    Args:
        add_generation_prompt:
            - True: user prompt (모델이 응답 시작)
            - False: tool response 후 (모델이 이어서 생성)
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception as e:
        return f"ERROR: {e}"


def execute_tool(name: str, arguments: Dict) -> str:
    """Execute tool and return result."""
    if name == "calculator":
        expr = arguments.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except Exception:
            return f"Error evaluating: {expr}"
    return "Unknown tool"


def run_cross_validation():
    """Run cross-validation tests."""
    print("=" * 80)
    print("CROSS-VALIDATION TEST: chat/completions vs completions")
    print("=" * 80)

    results = []

    for model_id, sglang_parser, formatter_name, hf_model, providers in MODELS:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_id}")
        print(f"SGLang Parser: {sglang_parser}, Formatter: {formatter_name}")
        print(f"{'=' * 80}")

        model_result: Dict[str, Any] = {
            "model": model_id,
            "sglang_parser": sglang_parser,
            "formatter_name": formatter_name,
            "hf_model": hf_model,
        }

        formatter = TOOL_RESPONSE_FORMATTERS.get(formatter_name)

        if not formatter:
            print(f"  ERROR: Formatter not found for {formatter_name}")
            model_result["error"] = f"Formatter not found: {formatter_name}"
            results.append(model_result)
            continue

        # ==================================================================
        # Flow A: chat/completions (독립적)
        # ==================================================================
        print("\n[Flow A] chat/completions...")

        # Turn 1
        messages_a = [{"role": "user", "content": USER_PROMPT}]
        chat_result1 = chat_completions(model_id, messages_a, providers, tools=[TOOL])

        if "error" in chat_result1:
            print(f"  Turn 1 ERROR: {chat_result1['error']}")
            model_result["flow_a_error"] = str(chat_result1["error"])
            results.append(model_result)
            continue

        chat_msg1 = chat_result1["choices"][0]["message"]
        tool_calls_a = chat_msg1.get("tool_calls", [])

        if not tool_calls_a:
            print(f"  Turn 1: No tool calls")
            model_result["flow_a_error"] = "No tool calls"
            results.append(model_result)
            continue

        tc_a = tool_calls_a[0]
        tc_a_name = tc_a["function"]["name"]
        tc_a_args = tc_a["function"]["arguments"]
        if isinstance(tc_a_args, str):
            try:
                tc_a_args = json.loads(tc_a_args)
            except json.JSONDecodeError:
                tc_a_args = {"raw": tc_a_args}

        print(f"  Turn 1: {tc_a_name}({tc_a_args})")

        # Execute tool
        tool_result_a = execute_tool(tc_a_name, tc_a_args)
        print(f"  Tool result: {tool_result_a}")

        # Turn 2
        messages_a2 = [
            {"role": "user", "content": USER_PROMPT},
            {
                "role": "assistant",
                "content": chat_msg1.get("content"),
                "tool_calls": [
                    {
                        "id": tc_a["id"],
                        "type": "function",
                        "function": {"name": tc_a_name, "arguments": json.dumps(tc_a_args)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tc_a["id"], "content": tool_result_a},
        ]
        chat_result2 = chat_completions(model_id, messages_a2, providers)

        if "error" in chat_result2:
            print(f"  Turn 2 ERROR: {chat_result2['error']}")
            flow_a_answer = None
        else:
            chat_msg2 = chat_result2["choices"][0]["message"]
            flow_a_answer = chat_msg2.get("content", "") + (chat_msg2.get("reasoning") or "")
            print(f"  Turn 2: {flow_a_answer[:100]}...")

        model_result["flow_a"] = {
            "turn1_tool_call": {"name": tc_a_name, "args": tc_a_args},
            "tool_result": tool_result_a,
            "turn2_answer": flow_a_answer,
        }

        # ==================================================================
        # Flow B: completions with OUR parser/formatter (독립적)
        # ==================================================================
        print("\n[Flow B] completions (우리 parser/formatter 사용)...")

        # Turn 1: user prompt with add_generation_prompt=True
        messages_b1 = [{"role": "user", "content": USER_PROMPT}]
        raw_prompt_b1 = build_raw_prompt_with_hf(messages_b1, hf_model, [TOOL], add_generation_prompt=True)

        if raw_prompt_b1.startswith("ERROR"):
            print(f"  HF tokenizer error: {raw_prompt_b1}")
            model_result["flow_b_error"] = raw_prompt_b1
            results.append(model_result)
            continue

        print(f"  Turn 1 prompt (last 80): {repr(raw_prompt_b1[-80:])}")

        comp_result1 = completions(model_id, raw_prompt_b1, providers)

        if "error" in comp_result1:
            print(f"  Turn 1 ERROR: {comp_result1['error']}")
            model_result["flow_b_error"] = str(comp_result1["error"])
            results.append(model_result)
            continue

        model_output_b1 = comp_result1["choices"][0].get("text", "")
        finish_reason_b1 = comp_result1["choices"][0].get("finish_reason", "")

        # Providers use tool call end tokens as stop tokens and remove them from output
        # Add back the end token if: (1) tool call start pattern detected, (2) finish_reason is "stop",
        # (3) end token is missing from output
        if sglang_parser in TOOL_CALL_PATTERNS:
            start_pattern, end_token = TOOL_CALL_PATTERNS[sglang_parser]
            has_tool_pattern = start_pattern in model_output_b1
            if has_tool_pattern and finish_reason_b1 == "stop" and end_token not in model_output_b1:
                model_output_b1 += end_token
                print(f"  Added missing end token: {repr(end_token)}")

        # Kimi-K2-Thinking: HF template expects <think></think> before tool calls
        # Model output starts with <|tool_calls_section_begin|> directly, so add empty think block
        if sglang_parser == "kimi_k2" and "Thinking" in hf_model:
            if "<|tool_calls_section_begin|>" in model_output_b1 and "<think>" not in model_output_b1:
                model_output_b1 = "<think></think>" + model_output_b1
                print(f"  Added <think></think> for Kimi-K2-Thinking template")

        print(f"  Turn 1 output (first 100): {model_output_b1[:100]}...")

        # Parse tool call with OUR parser (SGLang FunctionCallParser)
        try:
            _, parsed_calls = parse_tool_calls(model_output_b1, [TOOL], sglang_parser)
            if not parsed_calls:
                print(f"  Turn 1: Parser returned empty")
                model_result["flow_b_error"] = "Parser returned empty"
                results.append(model_result)
                continue

            tc_b = parsed_calls[0]
            tc_b_name = tc_b.name
            tc_b_args = tc_b.arguments
            tc_b_id = tc_b.call_id or "call_0"
            print(f"  Parsed: {tc_b_name}({tc_b_args})")
        except Exception as e:
            print(f"  Parser error: {e}")
            model_result["flow_b_error"] = f"Parser error: {e}"
            results.append(model_result)
            continue

        # Execute tool
        tool_result_b = execute_tool(tc_b_name, tc_b_args)
        print(f"  Tool result: {tool_result_b}")

        # Format tool response with OUR formatter (NO generation prompt)
        try:
            formatter_kwargs = {"content": tool_result_b}
            if formatter_name in ["kimi_k2", "mistral"]:
                formatter_kwargs["tool_call_id"] = tc_b_id
            if formatter_name == "gpt-oss":
                formatter_kwargs["tool_name"] = tc_b_name

            formatted_response = formatter(**formatter_kwargs)
            print(f"  Formatted: {repr(formatted_response)}")
        except Exception as e:
            print(f"  Formatter error: {e}")
            model_result["flow_b_error"] = f"Formatter error: {e}"
            results.append(model_result)
            continue

        # Turn 2: raw_prompt + model_output + formatted_response + generation_prompt (if needed)
        raw_prompt_b2 = raw_prompt_b1 + model_output_b1 + formatted_response

        # GLM-4.7: Add <|observation|> role marker (stripped by provider as EOS)
        if sglang_parser == "glm47":
            raw_prompt_b2 = raw_prompt_b1 + model_output_b1 + "<|observation|>" + formatted_response
            print(f"  Added <|observation|> role marker for GLM-4.7")

        # Kimi-K2 needs explicit generation prompt after tool response
        # Other models (qwen, glm, etc.) auto-continue without it
        if sglang_parser == "kimi_k2":
            gen_prompt = GENERATION_PROMPTS.get(sglang_parser, "")
            if gen_prompt:
                raw_prompt_b2 += gen_prompt
                print(f"  Added generation prompt: {repr(gen_prompt)}")

        print(f"  Turn 2 prompt (last 150): {repr(raw_prompt_b2[-150:])}")

        comp_result2 = completions(model_id, raw_prompt_b2, providers)

        if "error" in comp_result2:
            print(f"  Turn 2 ERROR: {comp_result2['error']}")
            flow_b_answer = None
        else:
            flow_b_answer = comp_result2["choices"][0].get("text", "")
            print(f"  Turn 2: {flow_b_answer[:100]}...")

        model_result["flow_b"] = {
            "turn1_raw_output": model_output_b1,
            "turn1_parsed": {"name": tc_b_name, "args": tc_b_args},
            "tool_result": tool_result_b,
            "formatted_response": formatted_response,
            "turn2_answer": flow_b_answer,
        }

        # ==================================================================
        # Validation
        # ==================================================================
        print("\n[Validation]")

        flow_a_has_42 = flow_a_answer and "42" in flow_a_answer
        flow_b_has_42 = flow_b_answer and "42" in flow_b_answer

        print(f"  Flow A (chat/completions) has '42': {flow_a_has_42}")
        print(f"  Flow B (completions+parser+formatter) has '42': {flow_b_has_42}")

        model_result["validation"] = {
            "flow_a_correct": flow_a_has_42,
            "flow_b_correct": flow_b_has_42,
            "both_correct": flow_a_has_42 and flow_b_has_42,
        }

        results.append(model_result)

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for r in results:
        model_short = r["model"].split("/")[-1]
        if "error" in r or "flow_a_error" in r or "flow_b_error" in r:
            err = r.get("error") or r.get("flow_a_error") or r.get("flow_b_error")
            print(f"  {model_short}: ERROR - {str(err)[:50]}")
        elif "validation" in r:
            v = r["validation"]
            status = "✓" if v["both_correct"] else "✗"
            print(f"  {model_short}: {status} (A={v['flow_a_correct']}, B={v['flow_b_correct']})")
        else:
            print(f"  {model_short}: No validation")

    # ======================================================================
    # Save results
    # ======================================================================
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    output = {
        "test_timestamp": now.isoformat(),
        "test_type": "cross_validation",
        "description": "Validates chat/completions vs completions with our parser/formatter",
        "results": results,
    }

    output_dir = Path(__file__).parent / "outputs" / "cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_path = output_dir / f"cross_validation_{timestamp_str}.json"
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    latest_path = output_dir / "cross_validation_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  - {ts_path}")
    print(f"  - {latest_path}")

    return results


if __name__ == "__main__":
    run_cross_validation()
