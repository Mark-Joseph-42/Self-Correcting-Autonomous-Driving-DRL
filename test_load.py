import sys
import os
sys.path.append('/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')

import torch
print(f"1. PyTorch imported. CUDA: {torch.cuda.is_available()}", flush=True)

from stable_baselines3 import PPO
print("2. SB3 imported", flush=True)

import time
start = time.time()
print("3. Loading model on CPU...", flush=True)
model = PPO.load("models/ppo_bc_baseline.zip", device="cpu")
print(f"4. Model loaded in {time.time()-start:.2f}s", flush=True)
print(f"5. Policy device: {model.policy.device}", flush=True)
