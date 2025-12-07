#!/bin/bash
#
# SLIME Training Script for tau2-bench with Qwen3-4B
#
# This script runs SLIME training using tau2 environment with AgentGymEnv.
# Ensure tau2 and SLIME are properly installed before running.
#

# Clean up any previous processes
pkill -9 sglang 2>/dev/null
sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null
sleep 3

set -ex

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Get script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Source model configuration
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

# ============== tau2 Configuration ==============
# These environment variables configure tau2-bench integration
export TAU2_DOMAIN="${TAU2_DOMAIN:-telecom}"
export TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
export TAU2_USER_LLM="${TAU2_USER_LLM:-gpt-4.1}"
export TAU2_USER_TEMP="${TAU2_USER_TEMP:-0.0}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-30}"
export TAU2_TOOL_PARSER="${TAU2_TOOL_PARSER:-qwen}"
export TAU2_MODEL_TYPE="${TAU2_MODEL_TYPE:-qwen3}"
export TAU2_RETURN_LOGPROB="${TAU2_RETURN_LOGPROB:-false}"
export TAU2_MAX_RESPONSE_TOKENS="${TAU2_MAX_RESPONSE_TOKENS:-1024}"

# ============== Data Preparation ==============
# Generate task JSONL if not exists
DATA_DIR="${SCRIPT_DIR}/data"
TRAIN_DATA="${DATA_DIR}/${TAU2_DOMAIN}_${TAU2_TASK_SPLIT}_tasks.jsonl"
EVAL_DATA="${DATA_DIR}/${TAU2_DOMAIN}_dev_tasks.jsonl"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Preparing training data..."
    mkdir -p "${DATA_DIR}"
    python "${SCRIPT_DIR}/prepare_data.py" \
        --domain "${TAU2_DOMAIN}" \
        --task-split "${TAU2_TASK_SPLIT}" \
        --output-path "${TRAIN_DATA}"
fi

if [ ! -f "${EVAL_DATA}" ]; then
    echo "Preparing evaluation data..."
    python "${SCRIPT_DIR}/prepare_data.py" \
        --domain "${TAU2_DOMAIN}" \
        --task-split "dev" \
        --output-path "${EVAL_DATA}" || true  # dev split may not exist
fi

# ============== Checkpoint Configuration ==============
CKPT_ARGS=(
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507/
    --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/
    --load /root/Qwen3-4B-Instruct-2507_slime/
    --save /root/Qwen3-4B-Instruct-2507_slime/
    --save-interval 20
)

# ============== Rollout Configuration ==============
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key index
    --rollout-shuffle
    --num-rollout 500
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len ${TAU2_MAX_RESPONSE_TOKENS}
    --rollout-temperature 0.8
    --global-batch-size 256
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --balance-data
)

# ============== Evaluation Configuration ==============
EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data ${TAU2_DOMAIN}-dev "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len ${TAU2_MAX_RESPONSE_TOKENS}
    --eval-top-k 1
)

# ============== Performance Configuration ==============
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

# ============== GRPO Configuration ==============
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# ============== Optimizer Configuration ==============
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

# ============== Wandb Configuration ==============
WANDB_ARGS=(
    # Uncomment to enable wandb logging
    # --use-wandb
    # --wandb-project slime-tau2-bench
    # --wandb-group qwen3-4B-${TAU2_DOMAIN}
    # --wandb-key ${WANDB_KEY}
)

# ============== SGLang Configuration ==============
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
    # If user simulation API reports concurrency limit, reduce this
    # --sglang-server-concurrency 32
)

# ============== Misc Configuration ==============
MISC_ARGS=(
    # Default dropout in megatron is 0.1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # Should be good for model performance
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    # Need to comment this when using model with MLA
    --attention-backend flash
)

# ============== Custom Generate Function ==============
CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_gym.generate
)

# ============== Launch Ray ==============
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Adjust GPU count as needed
NUM_GPUS=${NUM_GPUS:-2}

ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus ${NUM_GPUS} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir /root/shared/ray_temp

# ============== Runtime Environment ==============
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TAU2_DOMAIN\": \"${TAU2_DOMAIN}\",
    \"TAU2_TASK_SPLIT\": \"${TAU2_TASK_SPLIT}\",
    \"TAU2_USER_LLM\": \"${TAU2_USER_LLM}\",
    \"TAU2_USER_TEMP\": \"${TAU2_USER_TEMP}\",
    \"TAU2_MAX_STEPS\": \"${TAU2_MAX_STEPS}\",
    \"TAU2_TOOL_PARSER\": \"${TAU2_TOOL_PARSER}\",
    \"TAU2_MODEL_TYPE\": \"${TAU2_MODEL_TYPE}\",
    \"TAU2_RETURN_LOGPROB\": \"${TAU2_RETURN_LOGPROB}\",
    \"TAU2_MAX_RESPONSE_TOKENS\": \"${TAU2_MAX_RESPONSE_TOKENS}\"
  }
}"

# ============== Submit Training Job ==============
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${NUM_GPUS} \
    --rollout-num-gpus ${NUM_GPUS} \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}
