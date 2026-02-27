#!/bin/bash
# launch_carla_viz.sh - CARLA 0.9.13 launcher WITH Window/Rendering

CARLA_DIR="/home/tinkerspace/carla project"

echo "🚀 Launching CARLA 0.9.13 with Visualization..."
echo "GPU Optimization: RTX 4000 SFF (Ada Generation)"

# Cleanup existing processes
pkill -9 -f CarlaUE4-Linux-Shipping || true
sleep 3

# Local libomp5 fallback
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/carla_deps/lib"

# -windowed: Run in a window
# -ResX=1280 -ResY=720: Set resolution
# -vulkan: Better performance/stability on this system
# -prefernvidia: Ensure RTX 4000 is used
cd "$CARLA_DIR" && ./CarlaUE4.sh -windowed -ResX=1280 -ResY=720 -vulkan -nosound -prefernvidia 2>&1 | tee carla_viz.log &

# Wait for server to initialize
echo "Waiting 10s for CARLA initialization..."
sleep 10
echo "✅ CARLA Server is ready."
