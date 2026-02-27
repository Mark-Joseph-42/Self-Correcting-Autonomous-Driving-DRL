import os
import sys
import glob
import time
import pandas as pd
import numpy as np

# Add egg
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)

from carla_env import make_carla_env
from agent_logic import load_agent
from curriculum_manager import get_carla_curriculum_config

def evaluate_agent(name, model_path, env, episodes=20):
    print(f"🧐 Evaluating {name}...")
    model = load_agent(model_path, env=env)
    
    results = {
        "reward": [],
        "steps": [],
        "collisions": [],
        "distance": []
    }
    
    for i in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        ep_collisions = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if bool(env.collision_hist):
                ep_collisions += 1
            
            if ep_steps > 1000: # Timeout
                break
                
        results["reward"].append(ep_reward)
        results["steps"].append(ep_steps)
        results["collisions"].append(ep_collisions)
        results["distance"].append(info.get("reward_route", 0.0))
        
        print(f"  Episode {i+1}/{episodes}: Reward={ep_reward:.2f}, Collisions={ep_collisions}")

    return results

def run_comparison():
    """
    Generates the Research Proof comparison results between Curriculum and Baseline agents.
    """
    os.makedirs("./outputs/research", exist_ok=True)
    
    stages = get_carla_curriculum_config()
    gauntlet_cfg = stages[-1]
    env = make_carla_env(gauntlet_cfg)
    
    curriculum_model = "./outputs/stage_5/ppo_agent_stage_5.zip"
    baseline_model = "./outputs/baseline/ppo_baseline_final.zip"
    
    if not os.path.exists(curriculum_model) or not os.path.exists(baseline_model):
        print("❌ Error: One or both models missing. Ensure training is complete.")
        env.close()
        return

    curr_results = evaluate_agent("Curriculum Agent", curriculum_model, env)
    base_results = evaluate_agent("Baseline Agent", baseline_model, env)
    
    # Compile stats
    df_results = pd.DataFrame({
        "Metric": ["Mean Reward", "Mean Steps", "Total Collisions", "Success Rate (%)"],
        "Curriculum": [
            np.mean(curr_results["reward"]),
            np.mean(curr_results["steps"]),
            sum(curr_results["collisions"]),
            (np.mean(curr_results["distance"]) > 0.8) * 100
        ],
        "Baseline": [
            np.mean(base_results["reward"]),
            np.mean(base_results["steps"]),
            sum(base_results["collisions"]),
            (np.mean(base_results["distance"]) > 0.8) * 100
        ]
    })
    
    report_path = "./outputs/research/comparison_results.csv"
    df_results.to_csv(report_path, index=False)
    print(f"\n✅ Comparison Complete! Research metrics saved to {report_path}")
    print("\n--- PERFORMANCE SUMMARY ---")
    print(df_results.to_string())
    
    env.close()

if __name__ == "__main__":
    run_comparison()
