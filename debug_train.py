import os
import sys
import torch

# Try to find and import CARLA
try:
    import carla
except ImportError:
    import glob
    egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
    if os.path.exists(egg_file):
        sys.path.append(egg_file)
    import carla

from carla_env import CarlaEnv
from agent_logic import get_ppo_agent
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm

def run_debug_train():
    print("🚀 STARTING PHASE 1 DEBUG TRAINING (5000 Steps Limit)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./outputs/debug/phase1"
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "map": "Town01",
        "show_display": False,
        "enable_rewind": True,
        "enable_lookahead_reward": True,
        "ray_scale": True,
        "enable_pedestrian_safety": True,
        "fps": 10
    }
    
    print("📦 Initializing Environment...")
    env = CarlaEnv(config)
    
    print(f"🧠 Initializing PPO Agent (Device: {device})...", flush=True)
    model = get_ppo_agent(env, device=device, tensorboard_log=f"{output_dir}/tensorboard", debug=True)
    print("✅ PPO Agent Initialized.", flush=True)
    
    total_timesteps = 5000
    print(f"🏃 Training for {total_timesteps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, 
        save_path=f"{output_dir}/checkpoints",
        name_prefix="debug_model"
    )
    
    # Simple manual loop with tqdm for visibility
    pbar = tqdm(total=total_timesteps)
    current_steps = 0
    chunk_size = 512
    
    try:
        while current_steps < total_timesteps:
            model.learn(
                total_timesteps=chunk_size, 
                callback=checkpoint_callback,
                reset_num_timesteps=False,
                progress_bar=False
            )
            current_steps += chunk_size
            pbar.update(chunk_size)
            
            # Print periodic info
            # print(f"Step {current_steps}/{total_timesteps}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted.")
    finally:
        pbar.close()
        print(f"💾 Saving final debug model to {output_dir}/debug_final.zip")
        model.save(f"{output_dir}/debug_final.zip")
        env.close()
        print("✅ Debug Training Complete.")

if __name__ == "__main__":
    run_debug_train()
