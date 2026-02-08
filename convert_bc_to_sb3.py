import torch
from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np
from bc_trainer import BCNetwork

class MockMultiInputEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "semantic": spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    def reset(self): return {"semantic": np.zeros((1, 64, 64), dtype=np.uint8), "vector": np.zeros(44, dtype=np.float32)}
    def step(self, action): return self.reset(), 0, False, {}

def convert():
    bc_model = BCNetwork()
    bc_model.load_state_dict(torch.load("models/bc_best.pth", map_location="cpu"))
    bc_model.eval()
    
    env = MockMultiInputEnv()
    # net_arch=[] means the action_net takes the concatenated features directly.
    policy_kwargs = dict(net_arch=[]) 
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    
    sb_state = model.policy.state_dict()
    bc_state = bc_model.state_dict()
    
    print("\n--- SB3 Policy Keys ---")
    for k in sb_state.keys():
        if "features_extractor" in k:
            print(k)
    
    # Mapping for MultiInputPolicy with CombinedExtractor
    # CNN part
    mapping = {
        "conv.0.weight": "features_extractor.extractors.semantic.cnn.0.weight",
        "conv.0.bias": "features_extractor.extractors.semantic.cnn.0.bias",
        "conv.2.weight": "features_extractor.extractors.semantic.cnn.2.weight",
        "conv.2.bias": "features_extractor.extractors.semantic.cnn.2.bias",
        "conv.4.weight": "features_extractor.extractors.semantic.cnn.4.weight",
        "conv.4.bias": "features_extractor.extractors.semantic.cnn.4.bias",
        "fc.0.weight": "features_extractor.extractors.semantic.linear.0.weight",
        "fc.0.bias": "features_extractor.extractors.semantic.linear.0.bias",
    }
    
    new_state = {}
    for bc_key, sb_key in mapping.items():
        if bc_key in bc_state:
            new_state[sb_key] = bc_state[bc_key]
        else:
            print(f"⚠️ Warning: {bc_key} not found in BC model!")
            
    # Load mapped weights for features
    model.policy.load_state_dict(new_state, strict=False)
    
    # Manually handle the Action Net weight initialization
    with torch.no_grad():
        # Action Net Weight: [2, 300]
        # First 256 columns are semantic features
        model.policy.action_net.weight[:, :256] = bc_state["fc.2.weight"]
        # Last 44 columns are vector features (init to 0)
        model.policy.action_net.weight[:, 256:] = 0.0
        # Bias: [2]
        model.policy.action_net.bias.copy_(bc_state["fc.2.bias"])
        
    # Force everything to CPU before saving to avoid device mismatch on load
    model.policy.to("cpu")
    model.save("models/ppo_bc_baseline.zip")
    print("\n✅ Optimized Porting complete. Saved to models/ppo_bc_baseline.zip")

if __name__ == "__main__":
    convert()
