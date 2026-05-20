#!/bin/bash
# ============================================================
# Diffusion Policy Training for Piper Bottle Pick & Place Aside
# LeRobot 0.5.2 compatible
# ============================================================
# Prerequisites:
#   1. Data collected in data/lerobot_dataset (via teleop/data_collector.py)
#   2. conda activate piper_act
#
# Usage:  REPO_ID=piper/bottle_pick_place_aside bash training/train.sh
#         STEPS=1000 BATCH_SIZE=16 bash training/train.sh   # smoke test
# ============================================================

set -euo pipefail

REPO_ID="${REPO_ID:-piper/bottle_pick_place_aside}"
DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_pick_place_aside}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=diffusion \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.num_inference_steps=100 \
    --policy.vision_backbone=resnet18 \
    --policy.optimizer_lr=1e-4 \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=10000 \
    --eval_freq=10000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_diffusion_training
