import os
import glob
import sys
import numpy as np

# Add egg
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)

from carla_env import CarlaEnv

def test_env_class():
    config = {
        "host": "127.0.0.1",
        "port": 2000,
        "map": "Town01",
        "fps": 20,
        "width": 64,
        "height": 64
    }
    
    print("Initializing CarlaEnv...")
    env = CarlaEnv(config)
    
    print("Resetting env...")
    obs, info = env.reset()
    print("✅ Reset success!")
    
    print("Stepping env...")
    action = np.array([0.0, 0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Step success! Reward: {reward}")
    
    env.close()
    print("Cleanup complete.")

if __name__ == "__main__":
    test_env_class()
