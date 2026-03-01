#!/bin/bash
set -e

PYTHON="/venv/carla_py37/bin/python"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/libs

echo "=== Launching CARLA Visualization ==="
./launch_carla.sh

echo "=== Running Curriculum Agent Drive Demo ==="
PYTHONPATH=/workspace $PYTHON /workspace/src/drive_live.py

echo "=== Demo Ready ==="
echo "You can view the video at: /workspace/results/curriculum_driving_demo.mp4"
