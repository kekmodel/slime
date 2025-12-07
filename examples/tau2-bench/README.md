# tau2-bench SLIME Integration

SLIME (Step-Level Integrated Model Enhancement) training integration for tau2-bench.

## Overview

This module enables reinforcement learning training on tau2-bench tasks using SLIME's step-based RLVR (Reinforcement Learning with Verifiable Rewards) framework.

## Prerequisites

```bash
# 1. Install tau2-bench (from project root)
cd /path/to/tau2-bench
pip install -e .

# 2. Install SLIME
cd /path/to/slime
pip install -e .

# 3. Set API keys
cp .env.example .env
# Edit .env with your API keys
```

## File Structure

```
tau2-bench/
├── generate_with_gym.py      # SLIME rollout integration (main entry)
├── slime_env.py              # SlimeGymEnv wrapper for tau2
├── slime_gym_env.py          # Alternative gym env implementation
├── message_utils.py          # Token delta calculation utilities
├── tool_parser.py            # Multi-format tool call parser
├── prepare_data.py           # Task data preparation script
├── run_qwen3_4B.sh           # Training script for Qwen3-4B
├── test_live_episode_v2.py   # Live API test with Claude
├── test_integration.py       # Integration tests
├── test_slime_env.py         # Environment unit tests
└── trajectory_output/        # Test trajectory outputs
```

## Configuration

All settings via environment variables (`.env` in project root):

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_DOMAIN` | `telecom` | Domain (telecom, retail, airline) |
| `TAU2_TASK_SPLIT` | `train` | Task split (train, dev, base) |
| `TAU2_MAX_STEPS` | `100` | Max steps per episode |
| `TAU2_MAX_TURNS` | `30` | Max conversation turns |
| `TAU2_SOLO_MODE` | `false` | No user simulation (solo mode) |

### User Simulator

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_USER_LLM` | `gpt-4.1` | User simulation LLM |
| `TAU2_USER_TEMP` | `0.0` | Temperature for user LLM |

### Claude Extended Thinking

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_ALLOW_THINKING` | `false` | Enable extended thinking for Claude |
| `TAU2_THINKING_BUDGET` | `1024` | Thinking token budget |

When enabled:
- Automatically sets `temperature=1.0` (required for thinking)
- Enables **interleaved thinking** (`anthropic-beta: interleaved-thinking-2025-05-14`)
- Claude generates `thinking_blocks` after tool results (not just at turn start)

### Tool Parsing

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_TOOL_PARSER` | `qwen` | Parser type (qwen, llama3, mistral) |
| `TAU2_MODEL_TYPE` | `qwen3` | Model type for chat template |
| `TAU2_REFORMULATE_TOOL_INSTRUCTION` | `false` | Limit to single tool call |

### Response Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_MAX_RESPONSE_TOKENS` | `1024` | Max response tokens |
| `TAU2_RETURN_LOGPROB` | `false` | Collect log probabilities (for TIS) |

## Usage

### 1. Prepare Data

```bash
python prepare_data.py --domain telecom --task-split train
```

Output (`data/telecom_train_tasks.jsonl`):
```json
{"index": "telecom_task_001", "metadata": {"domain": "telecom", "task_split": "train"}}
```

### 2. Run Training

```bash
# Set environment
export TAU2_DOMAIN=telecom
export TAU2_USER_LLM=gpt-4.1
export TAU2_MAX_STEPS=30

# Run SLIME training with Qwen3-4B
bash run_qwen3_4B.sh
```

### 3. Test with Live API

Test Claude agent with extended thinking:

```bash
export TAU2_ALLOW_THINKING=true
export TAU2_THINKING_BUDGET=2048

python test_live_episode_v2.py
```

Outputs saved to `trajectory_output/`:
- `test_live_context_turn{N}.json` - Message context per turn
- `test_live_trajectory_turn{N}.txt` - Formatted trajectory

### 4. Run Tests

```bash
# Unit tests
pytest test_slime_env.py -v

# Integration tests
pytest test_integration.py -v
```

## Architecture

### Generate Flow

```
generate(args, sample, sampling_params) -> Sample
│
├── Initialize SlimeGymEnv(domain, task_id)
├── env.reset() → policy, tools, initial_messages
│
├── Multi-turn loop:
│   ├── apply_chat_template(messages, tools)
│   ├── sglang LLM call → assistant_response
│   ├── tool_adapter.parse(response)
│   │
│   ├── If tool_calls:
│   │   ├── Create ToolCallAction(name, arguments, id)
│   │   ├── env.step(actions) → tool_results
│   │   └── Track tokens (loss_mask=1 for assistant, 0 for tool)
│   │
│   └── If text_response:
│       ├── Create TextAction(content)
│       ├── env.step(action) → user_response
│       └── Track tokens (loss_mask=1 for assistant, 0 for user)
│
└── Return Sample
    ├── tokens: prompt + response tokens
    ├── loss_mask: [0]*prompt_len + [1,1,0,0,1,1,...]
    ├── reward: task completion score
    └── metadata: task_id, domain, turns, context
```

### Loss Mask Strategy

```
[System Prompt] [User Message] [Assistant] [Tool Result] [Assistant] [User] ...
     0's             0's          1's          0's          1's       0's

- Assistant responses: loss_mask = 1 (trainable)
- Tool results: loss_mask = 0 (environment response)
- User messages: loss_mask = 0 (environment response)
- System/prompt: loss_mask = 0 (context)
```

### Action Types

```python
# Tool call action
ToolCallAction(
    name="get_customer_info",
    arguments={"customer_id": "C123"},
    id="call_0"
)

# Text response action
TextAction(content="How can I help you today?")
```

### Episode Termination

Episode ends when:
1. Agent calls `done()` tool
2. Max turns reached
3. Max steps reached

## Extended Thinking (Claude)

When `TAU2_ALLOW_THINKING=true`:

1. **Budget**: Set via `TAU2_THINKING_BUDGET` (default: 1024 tokens)
2. **Temperature**: Automatically set to 1.0 (required)
3. **Interleaved Thinking**: Enabled via beta header, allows thinking after tool results

### Thinking in Messages

```python
# Assistant message with thinking
{
    "role": "assistant",
    "content": "Let me check your account...",
    "tool_calls": [...],
    "thinking_blocks": [
        {"type": "thinking", "thinking": "The customer needs..."}
    ],
    "reasoning_content": "The customer needs..."
}
```

For Qwen3 models, use `reasoning_content` field which gets wrapped in `<think>...</think>` tags by the chat template.

## Troubleshooting

### Empty Thinking Blocks

If `<think></think>` is empty after tool results:
- Ensure `TAU2_ALLOW_THINKING=true`
- Check that interleaved thinking header is being sent
- Verify `thinking_blocks` are preserved in message history

### Tool Parsing Errors

- Match `TAU2_TOOL_PARSER` to your model (qwen, llama3, mistral)
- Enable debug logging: `export LOGURU_LEVEL=DEBUG`
- Check tool call format in `tool_parser.py`

### Token Mismatch

- Ensure chat template is consistent across turns
- Call `get_token_delta()` after each message addition
- Verify tokenizer matches the training model

### API Rate Limits

- Reduce `TAU2_USER_TEMP` for more deterministic responses
- Use caching: Set `LLM_CACHE_ENABLED=true` in tau2 config
- Consider using `TAU2_SOLO_MODE=true` for testing without user simulation

## References

- [tau2-bench Documentation](https://github.com/sierra-research/tau2-bench)
- [SLIME Framework](https://github.com/kekmodel/slime)
- [Anthropic Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Interleaved Thinking (Beta)](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking)
