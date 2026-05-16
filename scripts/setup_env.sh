#!/bin/bash
# Set up Python virtual environment for Piper Bottle Grasp project.
# Usage:  bash scripts/setup_env.sh

set -e

ENV_DIR="${HOME}/piper_lerobot_env"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Piper Bottle Grasp — Environment Setup ==="
echo ""

# 1. Create venv
if [ ! -d "$ENV_DIR" ]; then
    echo "[1/3] Creating virtual environment at $ENV_DIR ..."
    python3 -m venv "$ENV_DIR"
else
    echo "[1/3] Environment already exists at $ENV_DIR"
fi

# 2. Activate and upgrade pip
echo "[2/3] Activating and upgrading pip ..."
source "$ENV_DIR/bin/activate"
pip install --upgrade pip

# 3. Install dependencies
echo "[3/3] Installing dependencies from requirements.txt ..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source $ENV_DIR/bin/activate"
echo "Then try:       python3 test_hardware.py"
