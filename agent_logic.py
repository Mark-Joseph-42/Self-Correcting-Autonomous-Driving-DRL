from stable_baselines3 import PPO
import torch

def get_ppo_agent(env, device="cpu", tensorboard_log="./logs/training", debug=False):
    """
    Initializes the PPO agent. Debug mode uses smaller buffers for faster iteration.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Linear arch (net_arch=[]) was too weak to coordinate steer/throttle.
    # Increasing complexity to [128, 128] for better behavior learning.
    policy_kwargs = dict(net_arch=[128, 128])
    
    # Debug mode: Update every 512 steps instead of 2048 for faster feedback
    n_steps = 512 if debug else 2048
    learning_rate = 1e-3 if debug else 3e-4  # Aggressive LR for debug
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1, 
        learning_rate=learning_rate, 
        max_grad_norm=0.5,
        n_steps=n_steps, 
        batch_size=256 if not debug else 128,
        n_epochs=10, 
        ent_coef=0.01,
        device=device,
        stats_window_size=1, 
        tensorboard_log=tensorboard_log
    )
    print("✅ PPO Constructor Complete.", flush=True)
    return model

def load_agent(path, env=None, device="cpu"):
    """
    Loads a trained PPO model.
    """
    if env:
        return PPO.load(path, env=env, device=device)
    return PPO.load(path, device=device)
