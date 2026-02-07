import os
import numpy as np
import torch
from agent_logic import load_agent

def test():
    """
    Inference script for CARLA 0.9.13.
    """
    os.environ["USE_CARLA"] = "1"
    
    # Priority: Environment variable then search
    model_path = os.environ.get("TEST_MODEL_PATH")
    
    if not model_path:
        potential_models = [
            "outputs/stage_5/ppo_agent_stage_5.zip",
            "outputs/stage_1/ppo_agent_stage_1.zip",
            "models/final_model_carla.zip"
        ]
        for path in potential_models:
            if os.path.exists(path):
                model_path = path
                break

    if not model_path:
        print("❌ No model found! Searching outputs...")
        import glob
        zips = glob.glob("outputs/**/*.zip", recursive=True)
        if zips:
            model_path = max(zips, key=os.path.getmtime)
            print(f"Using found model: {model_path}")
        else:
            print("❌ No trained CARLA model found.")
            return

    print(f"📡 Loading agent from {model_path}...")
    try:
        from carla_env import make_carla_env
        from curriculum_manager import get_carla_curriculum_config
        
        stages = get_carla_curriculum_config()
        env = make_carla_env(stages[0])
        model = load_agent(model_path, env=env)
    except Exception as e:
        print(f"❌ Load failed: {e}")
        print("Note: If you see pickle protocol errors, it means the model was saved with a newer Python version.")
        return
    
    print("▶️ Starting inference...")
    obs = env.reset()
    try:
        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                print("🔄 Collision! Resetting.")
                obs = env.reset()
            if i % 100 == 0:
                print(f"  Step {i}...")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    test()
