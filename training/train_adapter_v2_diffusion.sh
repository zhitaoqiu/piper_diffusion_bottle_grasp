#!/bin/bash
# Diffusion Policy training on the imported ACT adapter-v2 24-demo dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/lerobot_dataset_piper_bottle_adapter_v2_24demo}"
REPO_ID="${REPO_ID:-piper/adapter_v2_24demo_diffusion}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/diffusion_adapter_v2_24demo_5k}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
IMAGE_TRANSFORMS="${IMAGE_TRANSFORMS:-false}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/piper_act/bin/python3}"
JOB_NAME="${JOB_NAME:-diffusion_adapter_v2_24demo_5k}"

if [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
    echo "[ERROR] Imported adapter-v2 dataset is missing: ${DATASET_ROOT}" >&2
    echo "        Build it first with:" >&2
    echo "        ${PYTHON_BIN} scripts/import_adapter_v2_24demo.py" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "================================================"
echo "  Diffusion adapter-v2 24-demo training"
echo "================================================"
echo "  Dataset          : ${DATASET_ROOT}"
echo "  Repo ID          : ${REPO_ID}"
echo "  Output           : ${OUTPUT_DIR}"
echo "  Steps            : ${STEPS}"
echo "  Save freq        : ${SAVE_FREQ}"
echo "  Eval freq        : ${EVAL_FREQ}"
echo "  Batch size       : ${BATCH_SIZE}"
echo "  Image transforms : ${IMAGE_TRANSFORMS}"
echo "================================================"

PYTHONPATH= "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
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
