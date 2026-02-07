#!/bin/bash
# run_curriculum.sh - Robust multi-stage training with server restarts

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate carla_py37
export USE_CARLA=1

# Execute Stages (Support resuming, e.g. ./run_curriculum.sh 3)
START_STAGE=${1:-1}

# 0. Cleanup Stale Models (Crucial for Architecture Changes) - ONLY IF STARTING FRESH
if [ "$START_STAGE" -eq 1 ]; then
    echo "🧹 Cleaning up old model artifacts (Fresh Start)..."
    rm -f outputs/stage_*/ppo_agent_stage_*.zip
    rm -f outputs/stage_*/final_model_stage_*.zip
fi

run_stage() {
    STAGE=$1
    echo ""
    echo "=================================================="
    echo "♻️  PREPARING STAGE $STAGE (Fresh Server Restart)"
    echo "=================================================="
    
    # 1. Kill any existing server to free memory/assets
    pkill -f CarlaUE4 || true
    sleep 2
    
    # 2. Launch Server
    ./launch_carla.sh
    # launch_carla.sh already waits 10s, but we'll wait a bit more to be safe
    sleep 5 
    
    # 3. Run Training Stage
    echo "🚀 Running Training Stage $STAGE..."
    # We use python -u for unbuffered output
    python -u train_headless.py --stage $STAGE
    
    EXIT_CODE=$?
    
    # 3b. Check for Success (Ignore Segfault 139 if model exists)
    FINAL_MODEL="outputs/stage_${STAGE}/final_model_stage_${STAGE}.zip"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Stage $STAGE Success."
    elif [ $EXIT_CODE -eq 139 ] && [ -f "$FINAL_MODEL" ]; then
        echo "⚠️  Stage $STAGE Segfaulted on exit (Code 139), but model saved successfully. Continuing..."
    else
        echo "❌ Stage $STAGE Failed with code $EXIT_CODE. Stopping Curriculum."
        pkill -f CarlaUE4
        exit $EXIT_CODE
    fi
    
    # 4. Kill Server (Clean Slate for next map)
    echo "🧹 Killing Server..."
    pkill -f CarlaUE4
    sleep 5
}

# Execute Stages (Support resuming, e.g. ./run_curriculum.sh 3)


for i in {1..5}; do
    if [ $i -ge $START_STAGE ]; then
        run_stage $i
    fi
done

echo "🏁 Full Curriculum Complete!"
