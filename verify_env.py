import os
import numpy as np

def verify():
    print("--- CARLA Environment Verification ---")
    
    # Force CARLA backend
    os.environ["USE_CARLA"] = "1"
    
    try:
        from carla_env import make_carla_env
        from curriculum_manager import get_carla_curriculum_config
        
        stages = get_carla_curriculum_config()
        stage1 = stages[0]
        
        print(f"Creating env for {stage1['name']}...")
        env = make_carla_env(stage1)
        
        print("Resetting environment...")
        obs, info = env.reset()
        
        print(f"Observation Keys: {obs.keys()}")
        if "semantic_segmentation" in obs:
            shape = obs["semantic_segmentation"].shape
            print(f"Semantic Segmentation Shape: {shape}")
            if shape == (64, 64, 1):
                print("✅ Observation space matches requirements.")
        
        print("Stepping...")
        # Action: [steer, throttle/brake]
        action = np.array([0.0, 0.5]) 
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful. Reward: {reward}")
        
        env.close()
        print("✅ Verification Complete: CARLA integration is working.")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        print("Ensure CARLA server is running (use ./launch_carla.sh)")

if __name__ == "__main__":
    verify()
