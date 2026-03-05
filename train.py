"""
CARLA SAC Training — Maximum Speed Configuration with Resume Support
Uses SB3 defaults + robust stage transitions.
"""
import os, argparse, time
import pandas as pd
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from carla_env import CarlaEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        # Load existing rewards if they exist
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                self.ep_rewards = df['reward'].tolist()
                self.ep_steps = df['step'].tolist()
            except:
                self.ep_rewards, self.ep_steps = [], []
        else:
            self.ep_rewards, self.ep_steps = [], []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        dones = self.locals.get('dones', self.locals.get('done', None))
        is_done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        if is_done:
            self.ep_rewards.append(self.current_reward)
            self.ep_steps.append(self.num_timesteps)
            pd.DataFrame({
                'episode': range(1, len(self.ep_rewards)+1),
                'step': self.ep_steps,
                'reward': self.ep_rewards
            }).to_csv(self.log_path, index=False)
            self.current_reward = 0.0
        return True


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.prefix = prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.save_path}/sac_{self.prefix}_step{self.num_timesteps}"
            self.model.save(path)
            print(f"  [Checkpoint] Saved {path}")
        return True


def make_model(env, load_path=None):
    if load_path and os.path.exists(load_path):
        print(f"Resuming model from {load_path}")
        model = SAC.load(load_path, env=env, device="cuda")
    else:
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            learning_starts=500,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cuda",
            verbose=1,
        )
    return model


def train_cl():
    print("=== CL TRAINING (RESUME/SPEED MODE) ===")
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Progress check
    stage1_done = os.path.exists("models/sac_cl_stage1.zip")
    
    # Initialize env
    stage = 2 if stage1_done else 1
    env = Monitor(CarlaEnv(stage=stage))
    reward_cb = RewardLoggerCallback("logs/cl_rewards.csv")
    ckpt_cb = CheckpointCallback(2000, "models", "cl")
    cb = CallbackList([reward_cb, ckpt_cb])

    if not stage1_done:
        print("--- Stage 1: Town01 (10k steps) ---")
        model = make_model(env)
        model.learn(total_timesteps=10_000, callback=cb, reset_num_timesteps=False)
        model.save("models/sac_cl_stage1")
    else:
        print("--- Resuming for Stage 2: Town03 (10k steps) ---")
        model = make_model(env, "models/sac_cl_stage1.zip")
    
    if env.unwrapped.stage != 2:
        print("Switching to Stage 2...")
        env.unwrapped.set_stage(2)
        time.sleep(5) # Safety buffer for map load
        
    model.learn(total_timesteps=10_000, callback=cb, reset_num_timesteps=False)
    model.save("models/sac_cl_final")
    env.close()
    print("=== CL DONE ===")


def train_drl():
    print("=== DRL TRAINING (SPEED MODE) ===")
    env = Monitor(CarlaEnv(stage=3))
    reward_cb = RewardLoggerCallback("logs/drl_rewards.csv")
    ckpt_cb = CheckpointCallback(2000, "models", "drl")
    cb = CallbackList([reward_cb, ckpt_cb])

    # Check for existing final cl model to ensure we are truly starting fresh/baseline
    model = make_model(env)

    print("--- Full map (20k steps) ---")
    model.learn(total_timesteps=20_000, callback=cb)
    model.save("models/sac_drl_final")
    env.close()
    print("=== DRL DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cl", choices=["cl", "drl", "both"])
    args = parser.parse_args()
    if args.mode in ("cl", "both"):
        train_cl()
    if args.mode in ("drl", "both"):
        train_drl()
