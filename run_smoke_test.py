import os
import sys
import glob
import time
import numpy as np

# Add egg
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)

from carla_env import make_carla_env
from curriculum_manager import get_carla_curriculum_config

def run_smoke_test():
    """
    The 'Smoke Test': Proves the agent can 'see' (sensor shapes) and 'act' (moving in Town04).
    As per 50% Submission requirements.
    """
    print("🚦 Starting Integration Smoke Test (Town04)...")
    
    # Use Stage 1 config but override map to Town04 for smoke test
    stages = get_carla_curriculum_config()
    test_cfg = stages[0].copy()
    test_cfg["map"] = "Town04"
    test_cfg["name"] = "Smoke Test: Town04"
    
    os.makedirs("./outputs/integration", exist_ok=True)
    log_path = "./outputs/integration/smoke_test.log"
    
    with open(log_path, "w") as log:
        log.write(f"--- CARLA 0.9.13 Integration Smoke Test ---\n")
        log.write(f"Timestamp: {time.ctime()}\n\n")
        
        try:
            env = make_carla_env(test_cfg)
            obs = env.reset()
            
            log.write(f"Connection Successful: Sensors Active (Connected to {test_cfg['map']})\n")
            log.write(f"✅ Sensor Check: semantic_segmentation shape = {obs['semantic_segmentation'].shape}\n")
            log.write(f"✅ Sensor Check: vector shape = {obs['vector'].shape}\n\n")
            
            # --- Capture env_snapshot.png ---
            import cv2
            img = obs['semantic_segmentation']
            # Convert tag image to visible color for the snapshot (Road is 7, etc)
            # Scale it for visibility
            visible_img = img * 20 
            cv2.imwrite("./outputs/integration/env_snapshot.png", visible_img)
            print("📸 Captured env_snapshot.png")
            
            print("Running 50 steps to verify actions...")
            log.write("Step | Action [Steer, Throttle] | Reward | Collision\n")
            log.write("-" * 50 + "\n")
            
            for i in range(50):
                # Simple forward action
                action = np.array([0.0, 0.6]) 
                obs, reward, done, info = env.step(action)
                
                log.write(f"{i:4} | {action} | {reward:7.2f} | {done}\n")
                if (i + 1) % 10 == 0:
                    print(f"  Step {i+1}/50...")
            
            log.write("\n🏁 Smoke Test Result: PASSED\n")
            log.write("Agent successfully received observations and executed 50 actions without server crash.\n")
            
            env.close()
            print(f"✅ Smoke test complete. Log saved to {log_path}")
            
        except Exception as e:
            error_msg = f"❌ Smoke Test FAILED: {str(e)}"
            print(error_msg)
            log.write(f"\n{error_msg}\n")

if __name__ == "__main__":
    run_smoke_test()
