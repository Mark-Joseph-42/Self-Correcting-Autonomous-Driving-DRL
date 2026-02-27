import sys
import os
import time

print("Checking imports...")
start = time.time()

import numpy as np
import torch
import gym
from stable_baselines3 import PPO

# CARLA Egg
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)

import carla
print(f"Imports done in {time.time() - start:.2f}s")

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
print(f"Connecting to CARLA...")
world = client.get_world()
print(f"Connected to world: {world.get_map().name}")
