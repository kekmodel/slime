# Project Overview

## Purpose

**slime** is an LLM post-training framework for RL (Reinforcement Learning) scaling. It provides two core capabilities:

1. **High-Performance Training**: Supports efficient training in various modes by connecting Megatron-LM with SGLang
2. **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines

slime is the RL framework behind GLM-4.5 and GLM-4.6, and also supports:
- Qwen3 series (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 series
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1)
- Llama 3

## Core Architecture

The framework follows a three-module design:

1. **Training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training
2. **Rollout (SGLang + router)**: Generates new data (including rewards/verifier outputs) and stores it in the Data Buffer
3. **Data Buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods

## Key Features

- High-performance RL training by combining Megatron-LM (training) with SGLang (inference)
- Support for various RL algorithms: GRPO, GSPO, Reinforce++, PPO
- Flexible parallelism strategies (TP, PP, CP, EP, ETP)
- Dynamic sampling and partial rollout for efficient data generation
- Multi-turn interaction and agentic RL support
- Colocated or disaggregated training/inference modes
- bf16 training with fp8 inference support
