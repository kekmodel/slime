#!/bin/bash

# CISPO test script for single GPU (H100) with Megatron backend

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../scripts/models/qwen3-0.6B.sh"

HF_MODEL="Qwen/Qwen3-0.6B"
HF_CACHE_DIR="${HOME}/.cache/huggingface/hub"
TORCH_DIST_DIR="/root/Qwen3-0.6B_torch_dist"

# Convert HF model to Megatron torch_dist format if not already converted
if [ ! -d "${TORCH_DIST_DIR}" ]; then
    echo "Converting ${HF_MODEL} to torch_dist format..."
    cd /workspace/slime
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${HF_MODEL} \
        --save ${TORCH_DIST_DIR} \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL}
   --ref-load ${TORCH_DIST_DIR}
   --save /root/Qwen3-0.6B_cispo_slime
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data gsm8k/train.parquet
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 2
   --rollout-batch-size 2
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 0.8

   --global-batch-size 8

   --wandb-project slime-cispo-test
   --wandb-group cispo-h100-validation
)

CISPO_ARGS=(
   --advantage-estimator cispo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip-high 5.0
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
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "PYTHONPATH": "/root/Megatron-LM"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   --train-backend megatron \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${CISPO_ARGS[@]} \
   ${SGLANG_ARGS[@]}
