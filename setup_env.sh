#!/bin/bash
set -e

# Path to the carla_py37 environment python/pip
PYTHON="/venv/carla_py37/bin/python"
PIP="/venv/carla_py37/bin/pip"

echo "=== Installing CARLA 0.9.13 Python API ==="
$PIP install /workspace/carla_0.9.13/PythonAPI/carla/dist/carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl

echo "=== Installing Dependencies ==="
$PIP install opencv-python-headless matplotlib pandas pygame gymnasium==0.26.3 shimmy[gym-anygh] --no-deps || \
$PIP install opencv-python-headless matplotlib pandas pygame gymnasium==0.26.2

# We need a headless-friendly gymnasium/shimmy setup
$PIP install "gymnasium[classic_control,box2d]" || true

echo "=== Configuring Environment ==="
# Symlink cuDNN libs from the nvidia-cudnn-cu11 (or similar) package if present
CUDNN_PATH=$($PYTHON -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "")
if [ -n "$CUDNN_PATH" ]; then
    echo "Found cuDNN at $CUDNN_PATH, setting up symlinks..."
    mkdir -p /workspace/libs
    ln -sf $CUDNN_PATH/lib/*.so* /workspace/libs/
fi

# Set system limits
ulimit -u 4096

echo "=== Setup Complete ==="
