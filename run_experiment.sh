#!/bin/bash
set -e

PYTHON="/venv/carla_py37/bin/python"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/libs
export PYTHONPATH=$PYTHONPATH:/workspace

echo "=== PHASE 1: Launching CARLA ==="
./launch_carla.sh

echo "=== PHASE 2: Running Curriculum Training (100k steps) ==="
$PYTHON -m src.train --mode curriculum

echo "=== PHASE 3: Running Baseline Training (100k steps) ==="
# Restart server for clean slate
./launch_carla.sh
$PYTHON -m src.train --mode baseline

echo "=== PHASE 4: Generating Comparison ==="
$PYTHON -m src.compare

echo "=== Experiment Complete ==="
