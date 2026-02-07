import os
import numpy as np
import torch
from agent_logic import load_agent

import argparse
import gym
from stable_baselines3 import PPO

def test():
    """
    Inference script for CARLA 0.9.13.
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained agent.")
    parser.add_argument("--model", type=str, help="Path to the .zip model file")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage to simulate (1-5)")
    args = parser.parse_args()

    os.environ["USE_CARLA"] = "1"
    
    # Priority: Arg > Env Var > Default Search
    model_path = args.model or os.environ.get("TEST_MODEL_PATH")
    
    if not model_path:
        # Search defaults based on stage
        default_path = f"outputs/stage_{args.stage}/final_model_stage_{args.stage}.zip"
        if os.path.exists(default_path):
            model_path = default_path
        else:
            # Fallback search
            print(f"⚠️  {default_path} not found. Searching generally...")
            import glob
            zips = glob.glob("outputs/**/*.zip", recursive=True)
            if zips:
                model_path = max(zips, key=os.path.getmtime)
            else:
                print("❌ No trained CARLA model found.")
                return

    print(f"📡 Loading agent from {model_path}...")
    
    if not os.path.exists(model_path):
         print(f"❌ Error: Model file '{model_path}' does not exist.")
         return

    try:
        from carla_env import make_carla_env
        from curriculum_manager import get_carla_curriculum_config
        
        stages = get_carla_curriculum_config()
        # Stage is 1-indexed in args, 0-indexed in list
        stage_config = stages[args.stage - 1]
        
        print(f"🌍 Loading Environment for Stage {args.stage}: {stage_config['name']} (Map: {stage_config['map']})")
        
        env = make_carla_env(stage_config)
        
        # Load agent
        # We need to manually load because we might have different internal structures
        # handled by SB3's load
        model = PPO.load(model_path, env=env)
        
    except Exception as e:
        print(f"❌ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("▶️ Starting inference (Press Ctrl+C to stop)...")
    obs = env.reset()
    total_reward = 0
    try:
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                print(f"🔄 Episode Finished. Total Reward: {total_reward:.2f}")
                total_reward = 0
                obs = env.reset()
                
            if i % 50 == 0:
                print(f"  Step {i} | Speed: {info.get('speed', 0):.1f} km/h | Light: {info.get('light_state', 'Unknown')}")
                
    except KeyboardInterrupt:
        print("\n🛑 Test Validation Stopped by User.")
    finally:
        env.close()

if __name__ == "__main__":
    test()
