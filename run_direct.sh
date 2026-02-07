#!/bin/bash
# run_direct.sh - Direct execution to avoid conda run buffering

# Initialize conda for this shell
eval "$(conda shell.bash hook)"
conda activate carla_py37

export USE_CARLA=1

echo "🔍 Starting Headless Python script directly..."
python -u train_headless.py
