import os
import torch
from agent_logic import get_ppo_agent
from stable_baselines3.common.callbacks import CheckpointCallback
from metrics_logger import TransparencyCallback, save_telemetry_snapshot

def train_baseline():
    """
    Trains a baseline PPO agent directly on the 'Gauntlet' stage WITHOUT curriculum.
    Used for 100% Submission comparative research.
    """
    os.makedirs("./outputs/baseline", exist_ok=True)
    
    from carla_env import make_carla_env
    from curriculum_manager import get_carla_curriculum_config
    
    # Get Stage 5 (Gauntlet) config
    stages = get_carla_curriculum_config()
    gauntlet_cfg = stages[-1] 
    gauntlet_cfg["name"] = "Baseline: Gauntlet Only"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Baseline Training (Device: {device}) ===")
    
    env = make_carla_env(gauntlet_cfg)
    
    # We use a fresh model with a larger timestep budget to give it a fair chance against the curriculum
    # Total curriculum timesteps ~ 475k, we give baseline 500k
    total_timesteps = 500000
    
    model = get_ppo_agent(env, device=device, tensorboard_log="./outputs/baseline/tensorboard")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path="./outputs/baseline/checkpoints",
        name_prefix="baseline_model"
    )
    transparency_callback = TransparencyCallback()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, transparency_callback],
            progress_bar=True
        )
        model.save("./outputs/baseline/ppo_baseline_final")
        
        # Save final telemetry for baseline
        save_telemetry_snapshot("Baseline", transparency_callback.stats, "./outputs/baseline")
        
    except KeyboardInterrupt:
        print("Baseline training interrupted.")
        model.save("./outputs/baseline/ppo_baseline_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    train_baseline()
