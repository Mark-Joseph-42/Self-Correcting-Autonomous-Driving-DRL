import os
import torch
import time
import gc
from agent_logic import get_ppo_agent
from stable_baselines3.common.callbacks import CheckpointCallback
from metrics_logger import TransparencyCallback

def train():
    """
    Main training orchestrator for CARLA 0.9.13 Migration.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    
    # Selection of backend
    USE_CARLA = os.getenv("USE_CARLA", "1") == "1"
    
    if USE_CARLA:
        print("--- Using CARLA 0.9.13 Backend ---")
        from carla_env import make_carla_env as make_env
        from curriculum_manager import get_carla_curriculum_config as get_curriculum_config
        from curriculum_manager import RewardThresholdCallback
        from metrics_logger import TransparencyCallback, save_telemetry_snapshot
    else:
        print("--- Using MetaDrive Backend ---")
        from env_wrapper import make_env
        from curriculum_manager import get_curriculum_config
        from curriculum_manager import RewardThresholdCallback
        from metrics_logger import TransparencyCallback, save_telemetry_snapshot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Training (Device: {device}) ===")
    
    stages = get_curriculum_config()
    model = None

    try:
        for i, stage in enumerate(stages):
            stage_num = i + 1
            print(f"\n🚀 {stage['name']} (Map: {stage['map']})")
            
            # 1. Initialize Stage Output Directory (Phase 1)
            output_dir = f"./outputs/stage_{stage_num}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. Initialize Environment
            env = make_env(stage)
            
            # 3. Get/Update Agent
            if model is None:
                # Initial TB log for stage 1
                model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard")
            else:
                model.set_env(env)
                # Redirect TB log to current stage folder
                model.tensorboard_log = f"{output_dir}/tensorboard"
            
            # 4. Setup Callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=5000, 
                save_path=f"{output_dir}/checkpoints",
                name_prefix=f"stage{stage_num}_model"
            )
            stop_callback = RewardThresholdCallback(
                threshold=stage['threshold'], 
                stage_num=stage_num, 
                output_dir=output_dir,
                verbose=1
            )
            transparency_callback = TransparencyCallback()
            # 5. Train
            print(f"Training Stage {stage_num} (Goal: {stage['threshold']} reward)...")
            model.learn(
                total_timesteps=stage['timesteps'], 
                callback=[checkpoint_callback, stop_callback, transparency_callback], 
                progress_bar=True, # Live bar as requested
                reset_num_timesteps=False
            )
            
            # 6. Save Telemetry Snapshot (Phase 1)
            save_telemetry_snapshot(stage_num, transparency_callback.stats, output_dir)
            
            # 7. Close stage
            env.close()
            del env # Free memory
            gc.collect() 
            print(f"✅ Stage {stage_num} Complete.")
            print("⏳ Cooling down for 5s before next stage...")
            time.sleep(5.0)

        model.save("models/final_model_carla")
        print("\n🏁 Curriculum training complete. Final model saved.")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        if model is not None:
            # Save to current stage if possible
            try:
                save_path = f"./outputs/stage_{stage_num}/interrupted_model"
                model.save(save_path)
                print(f"💾 Saved to {save_path}")
            except:
                model.save("models/interrupted_model")
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    train()
