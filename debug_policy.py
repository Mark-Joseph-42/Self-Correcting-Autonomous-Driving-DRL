import torch
from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np

class MockMultiInputEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "semantic": spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    def reset(self): return {"semantic": np.zeros((1, 64, 64), dtype=np.float32), "vector": np.zeros(44, dtype=np.float32)}
    def step(self, action): return self.reset(), 0, False, {}

env = MockMultiInputEnv()
policy_kwargs = dict(
    net_arch=dict(pi=[512], vf=[512])
)
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
print(model.policy)
print("Keys:", model.policy.state_dict().keys())
