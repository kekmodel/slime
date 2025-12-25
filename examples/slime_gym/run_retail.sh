#!/bin/bash
# Example script to run retail service training with SLIME
#
# Usage:
#   ./run_retail.sh                    # Run with defaults (Qwen2.5-1.5B, 1 GPU)
#   NUM_GPUS=4 ./run_retail.sh         # Run with 4 GPUs
#   HF_CHECKPOINT=Qwen/Qwen2.5-7B-Instruct MODEL_CONFIG=qwen2.5-7B ./run_retail.sh
#
# Environment Variables:
#   HF_CHECKPOINT  - HuggingFace model path (default: Qwen/Qwen2.5-1.5B-Instruct)
#   MODEL_CONFIG   - Model architecture config name (default: qwen2.5-1.5B)
#                    See scripts/models/ for available configs
#   SAVE_DIR       - Directory to save checkpoints (default: ./outputs/retail_service)
#   NUM_GPUS       - Number of GPUs to use (default: 1)
#   NUM_REPEATS    - Number of times to repeat task set for training data (default: 10)
#   MASTER_ADDR    - Ray master address for multi-node (default: 127.0.0.1)
#
# slime_gym Configuration:
#   MAX_TURNS           - Fixed max turns (default: 10)
#   MAX_TURNS_BUFFER    - Extra turns for dynamic mode (default: 0)
#   DYNAMIC_MAX_TURNS   - "true" for dynamic, "false" for fixed (default: true)

# Cleanup any previous processes
pkill -9 sglang 2>/dev/null
sleep 2
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
sleep 2

set -ex

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Configuration
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen2.5-1.5B-Instruct}"
SAVE_DIR="${SAVE_DIR:-./outputs/retail_service}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_CONFIG="${MODEL_CONFIG:-qwen2.5-1.5B}"

# slime_gym configuration
MAX_TURNS="${MAX_TURNS:-10}"
MAX_TURNS_BUFFER="${MAX_TURNS_BUFFER:-0}"
DYNAMIC_MAX_TURNS="${DYNAMIC_MAX_TURNS:-true}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Load model architecture configuration
MODEL_CONFIG_PATH="${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}.sh"
if [[ -f "${MODEL_CONFIG_PATH}" ]]; then
    source "${MODEL_CONFIG_PATH}"
else
    echo "Warning: Model config not found at ${MODEL_CONFIG_PATH}"
    echo "Available configs:"
    ls "${SLIME_ROOT}/scripts/models/"
    MODEL_ARGS=()
fi

echo "=========================================="
echo "SLIME Gym - Retail Service Training"
echo "=========================================="
echo "Model: $HF_CHECKPOINT"
echo "Model Config: $MODEL_CONFIG"
echo "Save: $SAVE_DIR"
echo "GPUs: $NUM_GPUS"
echo "Data Repeats: ${NUM_REPEATS:-10}"
echo "Max Turns: $MAX_TURNS (buffer: $MAX_TURNS_BUFFER, dynamic: $DYNAMIC_MAX_TURNS)"
echo "=========================================="

# Generate training data
echo "Generating training data..."
mkdir -p "${SCRIPT_DIR}/data"

# Number of times to repeat the task set (default: 10 for 70 samples)
NUM_REPEATS="${NUM_REPEATS:-10}"

python3 << PYTHON_SCRIPT
import json
import sys
import os

# Import tasks module directly (without going through __init__.py which requires torch)
script_dir = "${SCRIPT_DIR}"
sys.path.insert(0, script_dir)

# Import the tasks module content directly
exec(open(os.path.join(script_dir, "tasks.py")).read())

base_samples = generate_training_samples()
num_repeats = ${NUM_REPEATS}

# Repeat samples with unique task_ids for more training data
samples = []
for i in range(num_repeats):
    for sample in base_samples:
        new_sample = {
            "prompt": sample["prompt"],
            "metadata": {**sample["metadata"], "task_id": f"{sample['metadata']['task_id']}_r{i}"}
        }
        samples.append(new_sample)

output_path = os.path.join(script_dir, "data", "retail_train.jsonl")
with open(output_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
print(f"Generated {len(samples)} training samples ({len(base_samples)} tasks x {num_repeats} repeats)")
PYTHON_SCRIPT

# Launch Ray head node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus ${NUM_GPUS} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# Checkpoint arguments
CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --save "${SAVE_DIR}"
    --save-interval 10
)

# Rollout arguments
ROLLOUT_ARGS=(
    --prompt-data "${SCRIPT_DIR}/data/retail_train.jsonl"
    --input-key prompt
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout 100
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --rollout-max-response-len 1024
    --rollout-temperature 0.7
    --global-batch-size 32
)

# Eval arguments (optional)
EVAL_ARGS=(
    # --eval-interval 10
    # --eval-prompt-data retail-eval "${SCRIPT_DIR}/data/retail_eval.jsonl"
)

# Performance arguments
PERF_ARGS=(
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
)

# Algorithm arguments
GRPO_ARGS=(
    --advantage-estimator grpo
    --eps-clip 0.2
)

# Optimizer arguments
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
)

# SGLang arguments
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
)

# Custom arguments for slime_gym
CUSTOM_ARGS=(
    --custom-generate-function-path examples.slime_gym.generate_with_gym.generate
    --custom-rm-path examples.slime_gym.generate_with_gym.reward_func
    --reward-key score
)

# Runtime environment for Ray job
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_ROOT}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SLIME_GYM_MAX_TURNS\": \"${MAX_TURNS}\",
    \"SLIME_GYM_MAX_TURNS_BUFFER\": \"${MAX_TURNS_BUFFER}\",
    \"SLIME_GYM_DYNAMIC_MAX_TURNS\": \"${DYNAMIC_MAX_TURNS}\"
  }
}"

# Submit Ray job
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${NUM_GPUS} \
    --rollout-num-gpus ${NUM_GPUS} \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${CUSTOM_ARGS[@]} \
    "$@"

echo "=========================================="
echo "Training submitted!"
echo "Monitor at: http://127.0.0.1:8265"
echo "=========================================="
