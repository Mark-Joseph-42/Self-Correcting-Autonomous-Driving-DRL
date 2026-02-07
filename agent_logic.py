from stable_baselines3 import PPO
import torch

def get_ppo_agent(env, device="cpu", tensorboard_log="./logs/training"):
    """
    Initializes the PPO agent with a MultiInputPolicy (Sensor Fusion).
    This logic is simulator-agnostic and will remain the same for CARLA.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=5e-5, # Stable for demo
        max_grad_norm=0.5,
        n_steps=512, 
        batch_size=64,
        ent_coef=0.01,
        device=device,
        stats_window_size=1, 
        tensorboard_log=tensorboard_log
    )
    return model

def load_agent(path, env=None, device="cpu"):
    """
    Loads a trained PPO model.
    """
    if env:
        return PPO.load(path, env=env, device=device)
    return PPO.load(path, device=device)
