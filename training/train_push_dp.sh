#!/bin/bash
# ============================================================
# Diffusion Policy — push-like task training (global-only)
# ============================================================
# Prerequisites:
#   1. Push-like task data collected via lerobot-record.
#   2. conda activate piper_act.
#   3. Dataset placed under data/push_task_debug (or override
#      DATASET_ROOT).
#
# Usage:
#   DATASET_ROOT=data/push_task_debug bash training/train_push_dp.sh
#   STEPS=5000 BATCH_SIZE=8 bash training/train_push_dp.sh  # quick overfit
#
# This entry uses the standard LeRobot train CLI and loads
# config/train_push_dp.json, which only uses observation.images.global_rgb
# (no wrist camera).  Horizon / n_obs / n_action defaults match the
# project's validated settings.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/push_task_debug}"
REPO_ID="${REPO_ID:-piper/push_task}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train_push_dp}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-4000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
IMAGE_TRANSFORMS="${IMAGE_TRANSFORMS:-true}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/piper_act/bin/python3}"
JOB_NAME="${JOB_NAME:-push_dp}"

if [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
    echo "[ERROR] Dataset is missing: ${DATASET_ROOT}" >&2
    echo "  Make sure you have collected push-like task data before training." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "================================================"
echo "  Diffusion Policy — push-like task training"
echo "================================================"
echo "  Config          : config/train_push_dp.json"
echo "  Dataset         : ${DATASET_ROOT}"
echo "  Repo ID         : ${REPO_ID}"
echo "  Output          : ${OUTPUT_DIR}"
echo "  Steps           : ${STEPS}"
echo "  Save freq       : ${SAVE_FREQ}"
echo "  Eval freq       : ${EVAL_FREQ}"
echo "  Batch size      : ${BATCH_SIZE}"
echo "  Image transforms: ${IMAGE_TRANSFORMS}"
echo "  Camera          : global only (no wrist)"
echo "================================================"

PYTHONPATH= "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
    --config_path="${PROJECT_ROOT}/config/train_push_dp.json" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable="${IMAGE_TRANSFORMS}" \
    --policy.type=diffusion \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=2 \
    --policy.num_inference_steps=100 \
    --policy.vision_backbone=resnet18 \
    --policy.optimizer_lr=1e-4 \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --persistent_workers=false \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq="${EVAL_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}"
