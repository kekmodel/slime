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

MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 1024
   --ffn-hidden-size 3072
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-0.6B
   --pretrained-checkpoint Qwen/Qwen3-0.6B
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
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
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
