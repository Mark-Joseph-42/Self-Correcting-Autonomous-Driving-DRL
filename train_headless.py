import os
import torch
import time
import gc
from agent_logic import get_ppo_agent
from stable_baselines3.common.callbacks import CheckpointCallback
from metrics_logger import TransparencyCallback

def train_headless():
    """
    Headless training orchestrator for CARLA 0.9.13.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1-5). 0=Run all.")
    args = parser.parse_args()
    
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("🚀 HEADLESS MODE ACTIVE", flush=True)
    
    # Selection of backend
    USE_CARLA = os.getenv("USE_CARLA", "1") == "1"
    
    if USE_CARLA:
        print("--- Using CARLA 0.9.13 Backend (Headless) ---")
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
        if args.stage > 0:
            # SINGLE STAGE MODE (Robust)
            stage_idx = args.stage - 1
            if stage_idx >= len(stages):
                print(f"❌ Error: Stage {args.stage} out of range (Max: {len(stages)})")
                return

            stage = stages[stage_idx]
            print(f"\n🚀 STARTING SINGLE STAGE: {stage['name']} (Map: {stage['map']})")
            
            # Force display off
            stage["show_display"] = False
            
            output_dir = f"./outputs/stage_{args.stage}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize Environment
            env = make_env(stage)
            
            # Load Model or Create New
            if args.stage == 1:
                print("🆕 Initializing new PPO agent...")
                model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard")
            else:
                # Load from previous stage
                prev_stage = args.stage - 1
                prev_model_path = f"./outputs/stage_{prev_stage}/ppo_agent_stage_{prev_stage}.zip"
                # Fallback to final save if callback save missing
                if not os.path.exists(prev_model_path):
                    prev_model_path = f"./outputs/stage_{prev_stage}/final_model_stage_{prev_stage}.zip"
                
                if os.path.exists(prev_model_path):
                    print(f"🔄 Loading model from {prev_model_path}...")
                    from stable_baselines3 import PPO
                    model = PPO.load(prev_model_path, env=env, device=device, tensorboard_log=f"{output_dir}/tensorboard")
                else:
                    print(f"⚠️ Warning: Previous model not found at {prev_model_path}. Starting fresh.")
                    model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard")

            # Setup Callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=5000, 
                save_path=f"{output_dir}/checkpoints",
                name_prefix=f"stage{args.stage}_model"
            )
            stop_callback = RewardThresholdCallback(
                threshold=stage['threshold'], 
                stage_num=args.stage, 
                output_dir=output_dir,
                verbose=1
            )
            transparency_callback = TransparencyCallback()
            
            # Train
            print(f"Training Stage {args.stage} (Goal: {stage['threshold']} reward)...")
            model.learn(
                total_timesteps=stage['timesteps'], 
                callback=[checkpoint_callback, stop_callback, transparency_callback], 
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            # Save Telemetry
            save_telemetry_snapshot(args.stage, transparency_callback.stats, output_dir)
            
            # EXPLICIT SAVE at end of stage (for next stage to pick up)
            final_save_path = f"{output_dir}/final_model_stage_{args.stage}.zip"
            model.save(final_save_path)
            print(f"💾 Stage {args.stage} complete. Model saved to {final_save_path}")
            
            env.close()

        else:
            # LEGACY MULTI-STAGE LOOP (Original Logic)
            for i, stage in enumerate(stages):
               stage_num = i + 1
               print(f"\n🚀 {stage['name']} (Map: {stage['map']})")
               stage["show_display"] = False
               output_dir = f"./outputs/stage_{stage_num}"
               os.makedirs(output_dir, exist_ok=True)
               env = make_env(stage)
               if model is None:
                   model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard")
               else:
                   model.set_env(env)
                   model.tensorboard_log = f"{output_dir}/tensorboard"
               checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=f"{output_dir}/checkpoints", name_prefix=f"stage{stage_num}_model")
               stop_callback = RewardThresholdCallback(threshold=stage['threshold'], stage_num=stage_num, output_dir=output_dir, verbose=1)
               transparency_callback = TransparencyCallback()
               model.learn(total_timesteps=stage['timesteps'], callback=[checkpoint_callback, stop_callback, transparency_callback], progress_bar=True, reset_num_timesteps=False)
               save_telemetry_snapshot(stage_num, transparency_callback.stats, output_dir)
               env.close()
               del env
               gc.collect()
               time.sleep(5.0)
            model.save("models/final_model_carla")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        if model is not None:
             # Save logic...
             pass
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    train_headless()
