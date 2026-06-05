#!/bin/bash
# ============================================================
# Multi-Task DiT (Diffusion Transformer) Training
# CLIP text + vision encoder → Transformer DiT → actions
# Supports language-conditioned tasks (e.g. "pick green" vs "pick blue")
# ============================================================
set -euo pipefail

DATASET_ROOT="${1:?Usage: $0 <dataset_root> [output_dir]}"
OUTPUT_DIR="${2:-outputs/train/multi_task_dit}"

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-5}"

DATASET_NAME=$(basename "$DATASET_ROOT")
REPO_ID="piper/${DATASET_NAME}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=multi_task_dit \
    --policy.objective=diffusion \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.num_inference_steps=100 \
    --policy.hidden_dim=512 \
    --policy.num_layers=6 \
    --policy.num_heads=8 \
    --policy.vision_encoder_name=/home/huatec/models/clip-vit-base-patch16 \
    --policy.text_encoder_name=/home/huatec/models/clip-vit-base-patch16 \
    --policy.image_crop_shape="[224, 224]" \
    --policy.tokenizer_max_length=77 \
    --policy.optimizer_lr="${LR}" \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=10000 \
    --eval_freq=10000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_multi_task_dit_training
