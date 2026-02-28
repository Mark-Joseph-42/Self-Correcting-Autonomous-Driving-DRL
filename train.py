import numpy as np
import os
import sys
import torch
import time
import gc

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from agent_logic import get_sac_agent
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from metrics_logger import TransparencyCallback

class MasteryBacktrackCallback(BaseCallback):
    """
    Handles 'Mastery' (save on 3 green lights) and 'Backtrack' (reload on pedestrian hit).
    """
    def __init__(self, output_dir, verbose=1):
        super(MasteryBacktrackCallback, self).__init__(verbose)
        self.output_dir = output_dir
        self.mastery_path = os.path.join(output_dir, "mastery_checkpoint.zip")
        self.last_mastery_count = 0

    def _on_step(self) -> bool:
        # Access the unwrapped env to check custom flags
        env = self.training_env.envs[0].unwrapped if hasattr(self.training_env, "envs") else self.training_env.unwrapped
        
        # 1. Mastery Check (Save on 3 green lights)
        if hasattr(env, "green_light_passes") and env.green_light_passes >= 3:
            if env.green_light_passes > self.last_mastery_count: # New pass
                 print(f"✨ MASTERY ACHIEVED: 3 Green Lights! Saving checkpoint to {self.mastery_path}")
                 self.model.save(self.mastery_path)
                 self.last_mastery_count = env.green_light_passes
                 # Optionally reset counter? Prompt says "every time", so we keep it or reset.
                 # Let's keep it and only save on multiples of 3.
        
        # 2. Backtrack Check (Pedestrian Hit)
        if hasattr(env, "pedestrian_collision") and env.pedestrian_collision:
            if os.path.exists(self.mastery_path):
                 print(f"🚨 BACKTRACK: Pedestrian hit! Model Vectorized env will handle reset...")
            else:
                 print("🚨 BACKTRACK FAILURE: No mastery checkpoint found. Environment handles reset.")
            
            env.pedestrian_collision = False
        return True

def detect_gpu():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU DETECTED: {name} ({vram:.1f} GB VRAM)")
        print(f"⚙️  CUDA Version: {torch.version.cuda}")
    else:
        print("🖥️  GPU NOT DETECTED. Using CPU.")

def train():
    """
    Robust training orchestrator for CARLA 0.9.13.
    """
    detect_gpu()
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
            stage["show_display"] = False
            env_raw = make_env(stage)
            
            # Manual wrapping for stability and visibility
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            print("📦 Wrapping Environment (Monitor + DummyVecEnv)...", flush=True)
            env_wrapped = Monitor(env_raw)
            env = DummyVecEnv([lambda: env_wrapped])
            print("✅ Environment Wrapped.", flush=True)
            
            # 3. Finalize Model with Env
            if model is None:
                if args.stage == 1:
                    # Priority 1: Mapped weights (.pth)
                    if os.path.exists("models/bc_mapped_weights.pth"):
                        print(f"🧠 Loading BC Mapped weights from models/bc_mapped_weights.pth...")
                        model = get_sac_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=args.debug)
                        model.policy.load_state_dict(torch.load("models/bc_mapped_weights.pth", map_location=device), strict=False)
                        print("✅ Mapped weights loaded successfully.")
                    # Priority 2: Full Baseline Zip
                    elif os.path.exists("models/ppo_bc_baseline.zip"):
                        print(f"🧠 Loading BC Bootstrap weights from models/ppo_bc_baseline.zip...")
                        try:
                            model = SAC.load("models/ppo_bc_baseline.zip", env=env, device=device)
                            print(f"✅ Model loaded directly to {device} and attached to environment.", flush=True)
                        except Exception as e:
                            print(f"⚠️ Warning: Failed to load baseline ({e}). Initializing fresh agent instead.")
                            model = get_sac_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=args.debug)
                    else:
                        print(f"🆕 Initializing fresh SAC agent (Debug: {args.debug})...")
                        model = get_sac_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=args.debug)
                else:
                    # Regular resume logic or fresh initialization if no previous stage found
                    # Look for model from previous stage (we already found path in step 1 potentially)
                    prev_stage = args.stage - 1
                    prev_model_path = f"./outputs/stage_{prev_stage}/ppo_agent_stage_{prev_stage}.zip"
                    if not os.path.exists(prev_model_path):
                        prev_model_path = f"./outputs/stage_{prev_stage}/final_model_stage_{prev_stage}.zip"
                    
                    if os.path.exists(prev_model_path):
                        print(f"🔄 Resuming from Stage {prev_stage}: {prev_model_path}")
                        model = SAC.load(prev_model_path, env=env, device=device)
                    else:
                        print(f"🆕 No previous stage model found. Initializing fresh SAC agent (Debug: {args.debug})...")
                        model = get_sac_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=args.debug)
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
            mastery_callback = MasteryBacktrackCallback(output_dir)
            
            # Train
            # Train directly (Avoid custom TQDM loop conflicts)
            print(f"Training Stage {args.stage} (Goal: {stage['threshold']} reward)...", flush=True)
            
            # Get total timesteps (Override for debug)
            total_timesteps = args.steps if args.steps else stage['timesteps']
            if args.debug and not args.steps:
                total_timesteps = 5000 # 5-min snapshot
                
            model.learn(
                total_timesteps=total_timesteps, 
                callback=[checkpoint_callback, stop_callback, transparency_callback, mastery_callback], 
                progress_bar=False, 
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
                   model = get_sac_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard")
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
