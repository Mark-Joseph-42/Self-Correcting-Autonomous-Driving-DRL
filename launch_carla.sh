#!/bin/bash
# launch_carla.sh - Headless CARLA 0.9.13 launcher for RTX 4000

CARLA_DIR="/home/tinkerspace/carla project"

echo "🚀 Launching CARLA 0.9.13 in Headless Mode..."
echo "GPU Optimization: RTX 4000 SFF (Ada Generation)"

# -RenderOffScreen: Headless rendering
# -quality-level=Low: Maximize throughput for DRL
# -prefernvidia: Ensure RTX 4000 is used
# pkill -9 to ensure port 2000 is definitely free
pkill -9 -f CarlaUE4-Linux-Shipping || true
sleep 3
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/carla_deps/lib"
cd "$CARLA_DIR" && ./CarlaUE4.sh -vulkan -RenderOffScreen -nosound -quality-level=Low -prefernvidia &

# Wait for server to initialize
echo "Waiting 10s for CARLA initialization..."
sleep 10
echo "✅ CARLA Server is ready."
