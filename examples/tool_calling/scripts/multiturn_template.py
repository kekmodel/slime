"""
Multi-turn conversation test: Observe apply_chat_template output for each model.

Test scenario (parallel + multi-hop):
1. User: Greeting -> Assistant responds
2. User: "Get prices of apple and banana, calculate average"
3. Assistant: Parallel tool calls (get_price x2)
4. Tool responses received
5. Assistant: Sequential tool call (calculator for average)
6. Tool response received
7. Assistant: Final answer

This tests:
- Regular messages (greeting)
- Parallel tool calls (2 get_price in one turn)
- Tool responses (multiple in one turn)
- Sequential tool call after tool response (calculator)
- Thinking/reasoning content preservation

Model-specific Notes:
=====================

Tool Response Role (OpenAI role="tool" becomes):
  - qwen: role="user" with <tool_response> wrapper
  - gpt-oss: namespace format (functions.X to=assistant)
  - kimi-k2: role="tool" via <|im_system|>tool
  - glm-4.7: <|observation|> role marker
  - minimax: role="tool" via ]~b]tool

Thinking/Reasoning:
  - gpt-oss: 'thinking' field → <|channel|>analysis
  - kimi-k2-thinking: 'reasoning_content' → <think> (history stripped)
  - Others: 'reasoning_content' → <think>

EOS Tokens (stripped by provider):
  - gpt-oss: <|call|> (tool), <|return|> (regular) - BOTH stripped
  - Others: Standard EOS, not stripped
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load .env from project root (examples/tool_calling/.env)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# API Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError(f"OPENROUTER_API_KEY not found. Set it in environment or create {_env_path}")
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test: (model_id, hf_model, providers, thinking_field)
# thinking_field: field name expected by HF template for reasoning content
#   - "reasoning_content": qwen, kimi, glm, minimax
#   - "thinking": gpt-oss
MODELS = [
    ("qwen/qwen3-30b-a3b-thinking-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507", ["Alibaba"], "reasoning_content"),
    ("qwen/qwen3-next-80b-a3b-thinking", "Qwen/Qwen3-Next-80B-A3B-Thinking", ["Alibaba"], "reasoning_content"),
    ("openai/gpt-oss-120b", "openai/gpt-oss-120b", ["Fireworks"], "thinking"),
    ("moonshotai/kimi-k2-thinking", "moonshotai/Kimi-K2-Thinking", ["Moonshot AI"], "reasoning_content"),
    ("z-ai/glm-4.7-flash", "zai-org/GLM-4.7-Flash", ["Z.AI"], "reasoning_content"),
    ("minimax/minimax-m2", "MiniMaxAI/MiniMax-M2.1", ["Minimax"], "reasoning_content"),
    ("nvidia/nemotron-3-nano-30b-a3b:free", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", ["NVIDIA"], "reasoning_content"),
]

# Tools: get_price returns price, calculator does math
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get the current price of a product",
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string", "description": "Product name"}},
                "required": ["product"],
            },
        },
    },
    {
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
    },
]


def chat_completions(
    model_id: str, messages: List[Dict], providers: List[str], tools: Optional[List[Dict]] = None
) -> Dict:
    """Call chat/completions endpoint."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "provider": {"order": providers},
        "temperature": 0.7,
    }
    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=180)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}


def execute_tool(name: str, arguments: Dict) -> str:
    """Execute tool and return result."""
    if name == "calculator":
        expr = arguments.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except Exception:
            return f"Error evaluating: {expr}"
    elif name == "get_price":
        product = arguments.get("product", "").lower()
        # Fake prices for testing
        prices = {"apple": 1500, "banana": 800, "orange": 1200}
        if product in prices:
            return str(prices[product])
        return f"Price not found for: {product}"
    return "Unknown tool"


def build_messages_from_conversation(turns: List[Dict], for_hf_template: bool = False) -> List[Dict]:
    """Build OpenAI-format messages from conversation turns.

    Args:
        turns: Conversation turns
        for_hf_template: If True, format tool_calls for HuggingFace templates
                        (some templates expect different format)
    """
    messages = []
    for turn in turns:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "assistant":
            msg = {"role": "assistant", "content": turn.get("content") or ""}
            # Include reasoning_content for thinking models
            if turn.get("reasoning_content"):
                msg["reasoning_content"] = turn["reasoning_content"]
            if turn.get("thinking"):
                msg["thinking"] = turn["thinking"]
            if "tool_calls" in turn:
                if for_hf_template:
                    # HF templates often expect tool_calls in a specific format
                    tool_calls = []
                    for tc in turn["tool_calls"]:
                        func = tc.get("function", tc)
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            args = json.loads(args)
                        tool_calls.append(
                            {
                                "id": tc.get("id", "call_0"),
                                "type": "function",
                                "function": {"name": func.get("name"), "arguments": args},  # dict, not string
                            }
                        )
                    msg["tool_calls"] = tool_calls
                else:
                    msg["tool_calls"] = turn["tool_calls"]
            messages.append(msg)
        elif turn["role"] == "tool":
            messages.append({"role": "tool", "tool_call_id": turn["tool_call_id"], "content": turn["content"]})
    return messages


def save_results(results: List[Dict[str, Any]], output_dir: Optional[str] = None) -> str:
    """Save test results to JSON file.

    Args:
        results: List of model test results
        output_dir: Directory to save results (defaults to script directory)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "multiturn")
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multiturn_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "timestamp": datetime.now().isoformat(),
        "test_name": "parallel_multihop_tool_call",
        "models_tested": len(results),
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Also save as latest for easy access
    latest_path = os.path.join(output_dir, "multiturn_results_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    print(f"Latest symlink: {latest_path}")
    return filepath


def run_test(models_to_test=None) -> List[Dict[str, Any]]:
    """Run parallel + multi-hop tool call test.

    Scenario:
    1. User: Hello! -> Assistant greeting
    2. User: Get apple and banana prices, calculate average
    3. Assistant: Parallel tool calls (get_price x2)
    4. After both prices -> Calculator for average
    5. Final answer

    Returns:
        List of test results for each model
    """
    print("=" * 80)
    print("PARALLEL + MULTI-HOP TOOL CALL TEST")
    print("=" * 80)

    models = models_to_test if models_to_test is not None else MODELS
    all_results = []

    TOOL_PROMPT = "Get the prices of apple and banana, then calculate the average price using the calculator tool."

    for model_id, hf_model, providers, thinking_field in models:
        model_result = {
            "model_id": model_id,
            "hf_model": hf_model,
            "providers": providers,
            "thinking_field": thinking_field,
            "status": "success",
            "error": None,
            "conversation": [],
            "template_output": None,
            "generation_prompt_suffix": None,
        }
        print(f"\n{'='*80}")
        print(f"MODEL: {model_id}")
        print(f"HF Model: {hf_model}")
        print("=" * 80)

        conversation = []

        # ============================================================
        # Turn 1: Greeting
        # ============================================================
        print("\n[Turn 1] User: Hello!")
        messages = [{"role": "user", "content": "Hello!"}]
        result = chat_completions(model_id, messages, providers)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            model_result["status"] = "error"
            model_result["error"] = result["error"]
            all_results.append(model_result)
            continue

        assistant_msg = result["choices"][0]["message"]
        greeting_response = assistant_msg.get("content", "") or ""
        reasoning = assistant_msg.get("reasoning") or ""
        # If no content but has reasoning, use reasoning as content
        if reasoning and not greeting_response:
            greeting_response = reasoning
        print(f"  Assistant: {greeting_response}")

        conversation.append({"role": "user", "content": "Hello!"})
        turn_msg = {"role": "assistant", "content": greeting_response}  # Full content
        if reasoning:
            turn_msg[thinking_field] = reasoning
        conversation.append(turn_msg)

        # ============================================================
        # Turn 2: Request parallel tool calls
        # ============================================================
        print(f"\n[Turn 2] User: {TOOL_PROMPT}")
        conversation.append({"role": "user", "content": TOOL_PROMPT})
        messages = build_messages_from_conversation(conversation)
        result = chat_completions(model_id, messages, providers, tools=TOOLS)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            model_result["status"] = "error"
            model_result["error"] = result["error"]
            model_result["conversation"] = conversation
            all_results.append(model_result)
            continue

        assistant_msg = result["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls", [])
        reasoning = assistant_msg.get("reasoning") or ""

        if not tool_calls:
            print(f"  No tool calls! Response: {assistant_msg.get('content', '')}")
            model_result["status"] = "no_tool_calls"
            model_result["error"] = f"No tool calls returned: {assistant_msg.get('content', '')}"
            model_result["conversation"] = conversation
            all_results.append(model_result)
            continue

        print(f"  Tool calls: {len(tool_calls)} (parallel: {len(tool_calls) > 1})")

        # Build assistant message with all tool calls
        turn_msg = {"role": "assistant", "content": assistant_msg.get("content") or "", "tool_calls": []}
        if reasoning:
            turn_msg[thinking_field] = reasoning
            print(f"  Reasoning: {reasoning}")

        # Execute all tool calls and collect results
        tool_responses = []
        for i, tc in enumerate(tool_calls):
            tc_name = tc["function"]["name"]
            tc_args = tc["function"]["arguments"]
            if isinstance(tc_args, str):
                tc_args = json.loads(tc_args)
            tc_id = tc["id"]

            print(f"  Tool call {i+1}: {tc_name}({tc_args})")
            tool_result = execute_tool(tc_name, tc_args)
            print(f"  Result {i+1}: {tool_result}")

            turn_msg["tool_calls"].append(
                {"id": tc_id, "type": "function", "function": {"name": tc_name, "arguments": json.dumps(tc_args)}}
            )
            tool_responses.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})

        conversation.append(turn_msg)
        conversation.extend(tool_responses)

        # ============================================================
        # Turn 3: After parallel responses -> Calculator or final
        # ============================================================
        print("\n[Turn 3] After parallel tool responses...")
        messages = build_messages_from_conversation(conversation)
        result = chat_completions(model_id, messages, providers, tools=TOOLS)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            model_result["status"] = "error"
            model_result["error"] = result["error"]
            model_result["conversation"] = conversation
            all_results.append(model_result)
            continue

        assistant_msg = result["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls", [])
        reasoning = assistant_msg.get("reasoning") or ""

        if tool_calls:
            # Calculator for average
            tc = tool_calls[0]
            tc_name = tc["function"]["name"]
            tc_args = tc["function"]["arguments"]
            if isinstance(tc_args, str):
                tc_args = json.loads(tc_args)
            tc_id = tc["id"]

            print(f"  Tool call: {tc_name}({tc_args})")
            if reasoning:
                print(f"  Reasoning: {reasoning}")

            tool_result = execute_tool(tc_name, tc_args)
            print(f"  Result: {tool_result}")

            turn_msg = {
                "role": "assistant",
                "content": assistant_msg.get("content") or "",
                "tool_calls": [
                    {"id": tc_id, "type": "function", "function": {"name": tc_name, "arguments": json.dumps(tc_args)}}
                ],
            }
            if reasoning:
                turn_msg[thinking_field] = reasoning
            conversation.append(turn_msg)
            conversation.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})

            # ============================================================
            # Turn 4: Final answer after calculator
            # ============================================================
            print("\n[Turn 4] After calculator...")
            messages = build_messages_from_conversation(conversation)
            result = chat_completions(model_id, messages, providers)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                model_result["status"] = "error"
                model_result["error"] = result["error"]
                model_result["conversation"] = conversation
                all_results.append(model_result)
                continue

            assistant_msg = result["choices"][0]["message"]
            final_answer = assistant_msg.get("content", "") or ""
            reasoning = assistant_msg.get("reasoning") or ""
            if reasoning and not final_answer:
                final_answer = reasoning

            print(f"  Final: {final_answer}")

            turn_msg = {"role": "assistant", "content": final_answer}
            if reasoning:
                turn_msg[thinking_field] = reasoning
            conversation.append(turn_msg)
        else:
            # Direct answer (model calculated mentally)
            final_answer = assistant_msg.get("content", "") or ""
            if reasoning and not final_answer:
                final_answer = reasoning
            print(f"  Direct answer: {final_answer}")

            turn_msg = {"role": "assistant", "content": final_answer}
            if reasoning:
                turn_msg[thinking_field] = reasoning
            conversation.append(turn_msg)

        # ============================================================
        # Show template output
        # ============================================================
        print("\n" + "-" * 60)
        print("APPLY_CHAT_TEMPLATE OUTPUT")
        print("-" * 60)

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

            template_messages = build_messages_from_conversation(conversation, for_hf_template=True)

            # ============================================================
            # Experiment: Compare generation prompt after tool response
            # ============================================================
            # Find the state right after first tool responses (before calculator)
            tool_response_state = []
            for i, msg in enumerate(template_messages):
                tool_response_state.append(msg)
                # Stop after tool responses (role=tool), before next assistant
                if msg.get("role") == "tool":
                    # Check if next message is assistant (not another tool)
                    if i + 1 < len(template_messages) and template_messages[i + 1].get("role") == "assistant":
                        break

            if tool_response_state and tool_response_state[-1].get("role") == "tool":
                print("\n" + "=" * 40)
                print("EXPERIMENT: After Tool Response")
                print("=" * 40)

                try:
                    # With add_generation_prompt=True
                    with_gen = tokenizer.apply_chat_template(
                        tool_response_state, tools=TOOLS, tokenize=False, add_generation_prompt=True
                    )
                except TypeError:
                    with_gen = tokenizer.apply_chat_template(
                        tool_response_state, tokenize=False, add_generation_prompt=True
                    )

                try:
                    # Without add_generation_prompt (False)
                    without_gen = tokenizer.apply_chat_template(
                        tool_response_state, tools=TOOLS, tokenize=False, add_generation_prompt=False
                    )
                except TypeError:
                    without_gen = tokenizer.apply_chat_template(
                        tool_response_state, tokenize=False, add_generation_prompt=False
                    )

                print(f"\n[add_generation_prompt=False] ends with:")
                print(repr(without_gen[-200:]))
                print(f"\n[add_generation_prompt=True] ends with:")
                print(repr(with_gen[-200:]))

                if len(with_gen) > len(without_gen):
                    diff = with_gen[len(without_gen) :]
                    print(f"\nDifference (suffix added): {repr(diff)}")

                model_result["tool_response_experiment"] = {
                    "with_generation_prompt_suffix": with_gen[-200:],
                    "without_generation_prompt_suffix": without_gen[-200:],
                    "difference": with_gen[len(without_gen) :] if len(with_gen) > len(without_gen) else None,
                }

                print("=" * 40)

            try:
                full_prompt = tokenizer.apply_chat_template(
                    template_messages, tools=TOOLS, tokenize=False, add_generation_prompt=False
                )
            except TypeError:
                full_prompt = tokenizer.apply_chat_template(
                    template_messages, tokenize=False, add_generation_prompt=False
                )

            print(f"\nFull conversation ({len(full_prompt)} chars):")
            print("-" * 40)
            print(full_prompt)
            print("-" * 40)

            # Show generation prompt
            try:
                gen_prompt = tokenizer.apply_chat_template(
                    template_messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
                )
            except TypeError:
                gen_prompt = tokenizer.apply_chat_template(
                    template_messages, tokenize=False, add_generation_prompt=True
                )

            if len(gen_prompt) > len(full_prompt):
                added = gen_prompt[len(full_prompt) :]
                print(f"\nGeneration prompt: {repr(added)}")
                model_result["generation_prompt_suffix"] = added

            model_result["template_output"] = full_prompt

        except Exception as e:
            print(f"Template error: {e}")
            model_result["template_error"] = str(e)

        # Save conversation and add to results
        model_result["conversation"] = conversation
        all_results.append(model_result)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    import sys

    # Usage:
    #   python test_multiturn_template.py [filter1] [filter2] ...
    #
    # Examples:
    #   python test_multiturn_template.py                    # all models
    #   python test_multiturn_template.py qwen3-30b          # filter by name
    #   python test_multiturn_template.py gpt-oss kimi       # multiple filters

    filter_args = sys.argv[1:] if len(sys.argv) > 1 else None

    models_to_run = MODELS
    if filter_args:
        print(f"Filtering: {filter_args}")
        filtered = []
        for m in MODELS:
            model_id, hf_model = m[0], m[1]
            for f in filter_args:
                if f.lower() in model_id.lower() or f.lower() in hf_model.lower():
                    filtered.append(m)
                    break
        if not filtered:
            print(f"No models matched: {filter_args}")
            print("Available:")
            for m in MODELS:
                print(f"  - {m[0]}")
            sys.exit(1)
        models_to_run = filtered

    results = run_test(models_to_run)
    save_results(results)
