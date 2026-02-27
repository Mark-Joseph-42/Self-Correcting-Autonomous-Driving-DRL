from stable_baselines3 import SAC
import torch

def get_sac_agent(env, device="cpu", tensorboard_log="./logs/training", debug=False):
    """
    Initializes the SAC agent for off-policy learning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using CnnPolicy since the input is now a 1x64x64 BEV Grid
    learning_rate = 1e-3 if debug else 3e-4
    
    model = SAC(
        "CnnPolicy", 
        env, 
        learning_rate=learning_rate, 
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256 if not debug else 128,
        ent_coef='auto',
        train_freq=1,
        gradient_steps=1,
        device=device,
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    print("✅ SAC Constructor Complete.", flush=True)
    return model

def load_agent(path, env=None, device="cpu"):
    """
    Loads a trained SAC model.
    """
    if env:
        return SAC.load(path, env=env, device=device)
    return SAC.load(path, device=device)
