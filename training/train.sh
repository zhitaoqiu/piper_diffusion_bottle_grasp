#!/bin/bash
# ============================================================
# Diffusion Policy Training for Piper Bottle Grasp
# ============================================================
# Prerequisites:
#   1. Data collected in data/lerobot_dataset (via teleop/data_collector.py)
#   2. conda activate piper_act
#
# Usage:  bash training/train.sh
# ============================================================

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_grasp}"
STEPS="${STEPS:-50000}"

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_grasp \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=diffusion \
    --policy.horizon=8 \
    --policy.n_action_steps=4 \
    --policy.n_obs_steps=2 \
    --policy.num_inference_steps=100 \
    --policy.dim_model=256 \
    --policy.n_heads=4 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=4 \
    --policy.dropout=0.1 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.repo_id=piper/bottle_grasp_diffusion \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --steps="${STEPS}" \
    --save_freq=10000 \
    --eval_freq=10000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_diffusion_training
