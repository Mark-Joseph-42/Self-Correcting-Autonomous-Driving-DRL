import os
import numpy as np

def debug():
    print("--- CARLA Environment Debugger ---")
    os.environ["USE_CARLA"] = "1"
    
    try:
        from carla_env import make_carla_env
        from curriculum_manager import get_carla_curriculum_config
        
        stages = get_carla_curriculum_config()
        env = make_carla_env(stages[0])
        
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        
        print("Resetting...")
        obs, info = env.reset()
        
        print("Observation keys and shapes:")
        for k, v in obs.items():
            print(f"  - {k}: shape {v.shape}, dtype {v.dtype}")
            if "segmentation" in k:
                unique_labels = np.unique(v)
                print(f"    Unique labels detected: {unique_labels}")
        
        print("\nTesting single step...")
        action = np.array([0.0, 0.5]) # No steer, half throttle
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step Reward: {reward}")
        print(f"Info: {info}")
        
        env.close()
        print("\n✅ Debug session successful.")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        print("Ensure CARLA server is running.")

if __name__ == "__main__":
    debug()
