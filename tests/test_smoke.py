import pytest
import time
from src.carla_env import CarlaEnv

def test_carla_env_smoke():
    """Requires CARLA server to be running."""
    env = None
    try:
        env = CarlaEnv()
        obs = env.reset(town="Town01")
        assert obs.shape == (64, 64, 1)
        assert obs.dtype == "uint8"
        
        for _ in range(5):
            obs, reward, done, info = env.step([0.0, 0.5])
            assert isinstance(reward, float)
            if done:
                env.reset()
        
    finally:
        if env:
            env.close()
