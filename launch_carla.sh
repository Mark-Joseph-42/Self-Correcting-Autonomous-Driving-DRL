#!/bin/bash

CARLA_DIR="/workspace/carla_0.9.13"
CARLA_SH="$CARLA_DIR/CarlaUE4.sh"
PORT=2000

kill_carla() {
    echo "Killing existing CARLA processes..."
    pkill -9 -f "CarlaUE4" || true
    sleep 2
}

if [[ "$1" == "--kill" ]]; then
    kill_carla
    exit 0
fi

kill_carla

echo "Launching CARLA 0.9.13 with -RenderOffScreen -opengl..."
# Remove SDL_VIDEODRIVER=offscreen as it's sometimes problematic with -RenderOffScreen
"$CARLA_SH" -RenderOffScreen -opengl -benchmark -fps=10 -carla-rpc-port=$PORT > /workspace/carla_log.txt 2>&1 &

# Wait for CARLA to start
echo "Waiting for CARLA to respond on port $PORT..."
MAX_ATTEMPTS=60
ATTEMPT=0
while ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$PORT" 2>/dev/null; do
    sleep 2
    ATTEMPT=$((ATTEMPT+1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "Error: CARLA failed to start after $MAX_ATTEMPTS attempts."
        tail -n 20 /workspace/carla_log.txt
        exit 1
    fi
    echo -n "."
done

# Extra wait for RPC server to initialize
sleep 5
echo -e "\nCARLA server ready."
