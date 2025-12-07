# tau2-bench SLIME Integration

SLIME training integration for tau2-bench using Gymnasium-style `AgentGymEnv`.

## Prerequisites

```bash
# 1. Install tau2-bench (from project root)
cd /path/to/tau2-bench
pip install -e .

# 2. Install SLIME
cd /path/to/slime
pip install -e .
```

Note: This directory (`slime/examples/tau2-bench/`) is a SLIME example module, not a standalone package.

## Features

- **Gymnasium Interface**: Uses tau2's `AgentGymEnv`
- **Multi-Tool Call Support**: Parallel tool calls with proper IDs
- **Extended Thinking**: Supports Claude's reasoning mode
- **Token Tracking**: Accurate delta for multi-turn

## Configuration

All settings via environment variables (`.env` in project root):

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_DOMAIN` | `telecom` | Domain (telecom, retail, airline) |
| `TAU2_TASK_SPLIT` | `train` | Task split |
| `TAU2_MAX_STEPS` | `100` | Max steps per episode |

### User Simulator

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_USER_LLM` | `gpt-4.1` | User simulation LLM |
| `TAU2_USER_TEMP` | `0.0` | Temperature |
| `TAU2_ALLOW_THINKING` | `false` | Enable extended thinking |
| `TAU2_THINKING_BUDGET` | `1024` | Thinking token budget |

### Tool Parsing

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_TOOL_PARSER` | `qwen` | Parser type (qwen, llama3, mistral) |
| `TAU2_REFORMULATE_TOOL_INSTRUCTION` | `false` | Limit to single tool call |

### Response Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_MAX_RESPONSE_TOKENS` | `1024` | Max response tokens |
| `TAU2_RETURN_LOGPROB` | `false` | Collect log probabilities |
| `TAU2_SOLO_MODE` | `false` | No user simulation |

## Usage

### 1. Prepare Data

```bash
# From project root
python slime/examples/tau2-bench/prepare_data.py --domain telecom

# Or from this directory
cd slime/examples/tau2-bench
python prepare_data.py --domain telecom --task-split train
```

Output (JSONL):
```json
{"index": "task_001", "metadata": {"domain": "telecom"}}
```

### 2. Run Training

```bash
# Set environment
export TAU2_DOMAIN=telecom
export TAU2_USER_LLM=claude-haiku-4-5
export TAU2_ALLOW_THINKING=true

# Run
bash run_qwen3_4B.sh
```

Or with SLIME directly:
```bash
python3 train.py \
    --custom-generate-function-path generate_with_gym.generate \
    --prompt-data ./data/telecom_tasks.jsonl \
    --input-key index
```

### 3. Run Tests

```bash
pytest test_integration.py -v
```

## Architecture

### Generate Flow

```
generate(args, sample, sampling_params) -> Sample
│
├── Initialize AgentGymEnv
├── Build prompt with chat_template
├── Multi-turn loop:
│   ├── sglang LLM call
│   ├── Parse tool calls (with IDs)
│   ├── Step environment
│   └── Update tokens/loss_mask
│
└── Return Sample
    ├── tokens
    ├── loss_mask (1=trainable, 0=not)
    ├── reward
    └── metadata
```

### Multi-Tool Call Format

```python
# Assistant message
{
    "role": "assistant",
    "content": "Let me check...",
    "tool_calls": [
        {"id": "functions.get_customer:0", "function": {...}},
        {"id": "functions.get_orders:1", "function": {...}},
    ]
}

# Tool responses (matched by ID)
{"role": "tool", "tool_call_id": "functions.get_customer:0", ...}
{"role": "tool", "tool_call_id": "functions.get_orders:1", ...}
```

## Troubleshooting

### Tool Parsing

- Verify `TAU2_TOOL_PARSER` matches model
- Enable debug: `export LOGURU_LEVEL=DEBUG`
- sglang not available? Falls back to tau2's native parser

### Token Mismatch

- Chat template must be consistent
- Call `get_token_delta()` after each message

### Environment Errors

- Check `TAU2_USER_LLM` API credentials
- Verify domain/task_split
