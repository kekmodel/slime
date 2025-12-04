#!/bin/bash
# Tau2-bench RLVR Training Script
#
# This script runs step-based RLVR training on tau2-bench environment.
# Based on the design in docs/rlvr_final_design.md
#
# Key settings:
# - Step-based sampling: Each turn is an independent sample
# - Mean centering advantage: G - mean(G), no std normalization
# - gamma=1.0, lambda=1.0: Monte Carlo return
# - Rejection sampling: Filter all-success or all-fail batches

set -e

# Model configuration
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/tau2_rlvr}"

# Tau2 configuration
TAU2_DOMAIN="${TAU2_DOMAIN:-telecom}"
TAU2_TASK_SET="${TAU2_TASK_SET:-telecom}"
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-}"

# Training configuration
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-30}"

# SGLang server configuration
SGLANG_ROUTER_IP="${SGLANG_ROUTER_IP:-127.0.0.1}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-30000}"

echo "=============================================="
echo "Tau2-bench RLVR Training"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Domain: ${TAU2_DOMAIN}"
echo "Task Set: ${TAU2_TASK_SET}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Run training
python train.py \
    --hf-checkpoint "${MODEL_PATH}" \
    --rollout-function-path "examples.tau2-bench.generate_with_tau2:generate" \
    --eval-function-path "examples.tau2-bench.generate_with_tau2:generate_eval" \
    \
    --advantage-estimator grpo \
    --grpo-std-normalization false \
    --gamma 1.0 \
    --lambd 1.0 \
    --use-critic false \
    \
    --dynamic-sampling-filter-path "slime.rollout.filter_hub.dynamic_sampling_filters:check_reward_nonzero_std" \
    \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --num-rollout "${NUM_ROLLOUT}" \
    --sglang-router-ip "${SGLANG_ROUTER_IP}" \
    --sglang-router-port "${SGLANG_ROUTER_PORT}" \
    \
    --tau2-domain "${TAU2_DOMAIN}" \
    --tau2-task-set "${TAU2_TASK_SET}" \
    --tau2-task-split "${TAU2_TASK_SPLIT}" \
    --max-num-steps "${MAX_NUM_STEPS}" \
    \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
