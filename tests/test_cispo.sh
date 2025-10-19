#!/bin/bash

# CISPO test script for single GPU (H100)
# Based on GSPO test but adapted for CISPO algorithm

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=1

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen2.5-0.5B-Instruct  # Small model for quick testing
)

ROLLOUT_ARGS=(
   --prompt-data dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 2  # Quick test with 2 rollouts
   --rollout-batch-size 2  # Reduced for single GPU
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048  # Reduced for faster testing
   --rollout-temperature 0.8

   --global-batch-size 8  # Reduced for single GPU
)

CISPO_ARGS=(
   --advantage-estimator cispo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip-high 5.0  # CISPO upper truncation (absolute value)
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   --train-backend fsdp \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${CISPO_ARGS[@]} \
   ${SGLANG_ARGS[@]}
