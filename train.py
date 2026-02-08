print("🚀 ROBUST TRAINING MODE ACTIVE", flush=True)
import os
import sys
import torch
import numpy as np
import time
import gc
from agent_logic import get_ppo_agent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from metrics_logger import TransparencyCallback

def train():
    """
    Robust training orchestrator for CARLA 0.9.13.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1-5). 0=Run all.")
    parser.add_argument("--debug", action="store_true", help="Enable ultra-fast debug training mode.")
    parser.add_argument("--steps", type=int, default=None, help="Override total timesteps for this run.")
    args = parser.parse_args()
    
    # Removed sys.stdout.reconfigure for stability
    
    # Selection of backend (Default to CARLA)
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
            
            # Use debug subfolder if requested
            base_output = "./outputs/debug" if args.debug else "./outputs"
            output_dir = f"{base_output}/stage_{args.stage}"
            os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
            
            print(f"\n🚀 STARTING {'DEBUG ' if args.debug else ''}SINGLE STAGE: {stage['name']} (Map: {stage['map']})")
            
            # 1. Prepare Model (if loading from previous stage)
            model = None
            if args.stage > 1:
                # Load from previous stage
                prev_stage = args.stage - 1
                prev_model_path = f"./outputs/stage_{prev_stage}/ppo_agent_stage_{prev_stage}.zip"
                if not os.path.exists(prev_model_path):
                    prev_model_path = f"./outputs/stage_{prev_stage}/final_model_stage_{prev_stage}.zip"
                
                if os.path.exists(prev_model_path):
                    print(f"🔄 Preparing to load model from {prev_model_path}...")
                    # Delay loading until env is ready for stage > 1 to avoid mismatch
                    pass # We'll load it in step 3

            # 2. Initialize CARLA Environment
            print(f"🚀 Initializing Environment...", flush=True)
            env = make_env(stage)
            
            # 3. Finalize Model with Env
            if model is None:
                if args.stage == 1 and os.path.exists("models/ppo_bc_baseline.zip"):
                    print(f"🧠 Loading BC Bootstrap weights from models/ppo_bc_baseline.zip...")
                    try:
                        model = PPO.load("models/ppo_bc_baseline.zip", env=env, device=device)
                        print(f"✅ Model loaded directly to {device} and attached to environment.", flush=True)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to load baseline ({e}). Initializing fresh agent instead.")
                        model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=args.debug)
                else:
                    print(f"🆕 Initializing fresh PPO agent (Debug: {args.debug})...")
                    model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=args.debug)
            else:
                print("✅ Setting environment for loaded model...")
                model.set_env(env)
                # Ensure reset for starting
                env.reset()
                print(f"🛠️  Policy Device: {model.policy.device}", flush=True)

            # Setup Callbacks
            save_freq = 500 if args.debug else 5000
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq, 
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
            # Train loop with TQDM for stability (avoids Rich/sys.meta_path crash)
            from tqdm import tqdm
            print(f"Training Stage {args.stage} (Goal: {stage['threshold']} reward)...", flush=True)
            
            # Get total timesteps (Override for debug)
            total_timesteps = args.steps if args.steps else stage['timesteps']
            if args.debug and not args.steps:
                total_timesteps = 5000 # 5-min snapshot
            
            pbar = tqdm(total=total_timesteps, file=sys.stdout, dynamic_ncols=True)
            current_steps = 0
            
            try:
                while current_steps < total_timesteps:
                     # Train in small chunks
                     chunk_size = 512 if args.debug else 2048
                     model.learn(
                        total_timesteps=chunk_size, 
                        callback=[checkpoint_callback, stop_callback, transparency_callback], 
                        progress_bar=False, 
                        reset_num_timesteps=False
                     )
                     current_steps += chunk_size
                     pbar.update(chunk_size)
                     
                     # Check for stage completion (via callback)
                     # Handle both VecEnv and raw Env
                     is_complete = False
                     if hasattr(env, "get_attr"):
                         is_complete = env.get_attr("_stage_complete")[0]
                     elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_stage_complete"):
                         is_complete = env.unwrapped._stage_complete
                         
                     if is_complete:
                         print(f"✅ Stage {args.stage} graduation criteria met!", flush=True)
                         break
            finally:
                pbar.close()
            
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
    train()
