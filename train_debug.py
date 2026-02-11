
import os
import sys
import glob

# Add CARLA to path
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)

print("1. Importing carla")
import carla
print("2. Importing torch")
import torch
print("3. Importing SB3")
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_env import CarlaEnv

def run():
    print("4. Creating Env")
    env = CarlaEnv({'map': 'Town01'})
    print("5. Wrapping Monitor")
    env = Monitor(env)
    print("6. Wrapping DummyVecEnv")
    env = DummyVecEnv([lambda: env])
    print("7. Initializing PPO (CUDA)")
    model = PPO('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=None)
    print("8. Starting learn(10)")
    model.learn(total_timesteps=10)
    print("9. Success!")

if __name__ == "__main__":
    run()
