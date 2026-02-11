#!/bin/bash
# Environment fixes for stability
export OMP_NUM_THREADS=1
export FORCE_CPU=0 # Set to 1 if GPU hangs persist
# run_curriculum.sh - Robust multi-stage training with server restarts

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate carla_py37
export USE_CARLA=1

# Execute Stages (Support resuming, e.g. ./run_curriculum.sh 3 5)
START_STAGE=${1:-1}
END_STAGE=${2:-5}

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
    
    # 3. Run Training Stage (with process lock)
    echo "🚀 Running Training Stage $STAGE..."
    echo "🚀 Running Training Stage $STAGE..."
    # Run with retry on segfaults
    MAX_RETRIES=3
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        python -u train_lock.py --stage $STAGE
        EXIT_CODE=$?
        
        FINAL_MODEL="outputs/stage_${STAGE}/final_model_stage_${STAGE}.zip"
        CHECKPOINT_MODEL="outputs/stage_${STAGE}/ppo_agent_stage_${STAGE}.zip"
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Stage $STAGE Success."
            break
        elif [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 134 ] || [ $EXIT_CODE -eq 11 ]; then
            # Segfault handling
            if [ -f "$FINAL_MODEL" ] || [ -f "$CHECKPOINT_MODEL" ]; then
                echo "⚠️  Stage $STAGE Segfaulted (Code $EXIT_CODE), but model saved. Continuing..."
                break
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                echo "🔄 Segfault detected (Code $EXIT_CODE). Restarting CARLA... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
                pkill -9 -f CarlaUE4 || true
                sleep 5
                ./launch_carla.sh
                sleep 10
            fi
        else
            echo "❌ Stage $STAGE Failed with code $EXIT_CODE. Stopping Curriculum."
            pkill -f CarlaUE4
            exit $EXIT_CODE
        fi
    done
    
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Stage $STAGE failed after $MAX_RETRIES retries. Exiting."
        exit 1
    fi
    
    # 4. Kill Server (Clean Slate for next map)
    echo "🧹 Killing Server..."
    pkill -f CarlaUE4
    sleep 5
}

# Execute Stages (Support resuming, e.g. ./run_curriculum.sh 3)



for i in $(seq $START_STAGE $END_STAGE); do
    run_stage $i
done

echo "🏁 Full Curriculum Complete!"
