#!/bin/bash

# GSPO test script for single GPU (H100) with Megatron backend

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

# Use 200GB cache volume for tmp directories
export TMPDIR=/root/.cache/ray_tmp
export RAY_TMPDIR=/root/.cache/ray_tmp
mkdir -p "$TMPDIR"

# Get repository root directory (POSIX compatible)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

HF_MODEL="Qwen/Qwen3-0.6B"
HF_CACHE_DIR="${HOME}/.cache/huggingface/hub"
TORCH_DIST_DIR="/root/.cache/huggingface/slime/Qwen3-0.6B_torch_dist"

# Convert HF model to Megatron torch_dist format if not already converted
if [ ! -d "${TORCH_DIST_DIR}" ]; then
    echo "Converting ${HF_MODEL} to torch_dist format..."
    PYTHONPATH=/root/Megatron-LM python "${REPO_ROOT}/tools/convert_hf_to_torch_dist.py" \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${HF_MODEL} \
        --save ${TORCH_DIST_DIR} \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL}
   --ref-load ${TORCH_DIST_DIR}
   # --save /root/.cache/huggingface/slime/checkpoints/gspo-qwen3-0.6B
   # --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data /root/.cache/huggingface/datasets/gsm8k/train.parquet
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 110
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1.0

   --global-batch-size 256
   --micro-batch-size 1

   --use-wandb
   --wandb-project slime-cispo-test
   --wandb-group gspo-h100-mean-std
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k /root/.cache/huggingface/datasets/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-k 1
)

GSPO_ARGS=(
   --advantage-estimator gspo
   # --disable-grpo-std-normalization  # Dr. GRPO: mean-centering만 (binary reward에 최적)
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-3  # GSPO paper ablation: 10^-3 scale optimal for 8B model
   --eps-clip-high 5e-3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.95
   --adam-eps 1e-15
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-log-level warning
   --sglang-log-level-http error
   # Note: FP32 precision is automatically provided by Megatron
   # via --attention-softmax-in-fp32 and --accumulate-allreduce-grads-in-fp32
   # LM head log-probs are automatically upcast to FP32
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
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
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GSPO_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

# Auto-stop pod after training (uncomment to enable)
echo "Training complete. Stopping pod in 60 seconds..."
sleep 60
runpodctl stop pod $RUNPOD_POD_ID
