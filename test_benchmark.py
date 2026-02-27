#!/usr/bin/env python
"""
Benchmark Test Script: Runs episodes in headless mode, then replays the best in visual real-time.
"""
import os
import time
import numpy as np
import argparse
from stable_baselines3 import PPO

def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark agent and visualize best episode.")
    parser.add_argument("--model", type=str, default=None, help="Path to the .zip model file")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage (1-5)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to benchmark")
    args = parser.parse_args()

    os.environ["USE_CARLA"] = "1"
    
    # Find model
    model_path = args.model or f"outputs/stage_{args.stage}/final_model_stage_{args.stage}.zip"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"🚀 BENCHMARK MODE: {args.episodes} episodes, then visualize best")
    print(f"📦 Model: {model_path}")
    
    # === PHASE 1: Headless Benchmark ===
    print("\n" + "="*50)
    print("📊 PHASE 1: HEADLESS BENCHMARK")
    print("="*50)
    
    from carla_env import make_carla_env
    from curriculum_manager import get_carla_curriculum_config
    
    stages = get_carla_curriculum_config()
    stage_config = stages[args.stage - 1].copy()
    stage_config['show_display'] = False  # Force headless
    
    env = make_carla_env(stage_config)
    model = PPO.load(model_path, env=env)
    
    episode_results = []
    
    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        spawn_location = env.vehicle.get_location() if env.vehicle else None
        
        done = False
        while not done and steps < 500:  # Max 500 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        result = {
            'episode': ep + 1,
            'reward': total_reward,
            'steps': steps,
            'spawn_x': spawn_location.x if spawn_location else 0,
            'spawn_y': spawn_location.y if spawn_location else 0,
        }
        episode_results.append(result)
        print(f"  Episode {ep+1:2d} | Reward: {total_reward:7.2f} | Steps: {steps:3d}")
    
    env.close()
    
    # Find best episode
    best = max(episode_results, key=lambda x: x['reward'])
    print(f"\n🏆 BEST EPISODE: #{best['episode']} with reward {best['reward']:.2f}")
    
    # === PHASE 2: Visual Replay at Real-Time ===
    print("\n" + "="*50)
    print("🎬 PHASE 2: REAL-TIME VISUALIZATION OF BEST EPISODE")
    print("="*50)
    print(f"Spawning at: ({best['spawn_x']:.1f}, {best['spawn_y']:.1f})")
    
    # Kill existing CARLA and restart in headless (we still can't do windowed)
    # But we can show OpenCV window for the RGB camera feed
    import subprocess
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # Restart with visualization enabled
    stage_config['show_display'] = True  # Enable OpenCV visualization
    
    # Restart CARLA
    print("🔄 Restarting CARLA server...")
    subprocess.run(["./launch_carla.sh"], cwd=os.path.dirname(os.path.abspath(__file__)))
    time.sleep(15)  # Wait for CARLA to start
    
    env = make_carla_env(stage_config)
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    total_reward = 0
    
    print("▶️ Running best episode at real-time speed (Press Ctrl+C to stop)...")
    try:
        for step in range(500):
            start_time = time.time()
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Real-time: wait to match 20 FPS (50ms per frame)
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.05 - elapsed)  # 20 FPS = 50ms per frame
            time.sleep(sleep_time)
            
            if step % 20 == 0:
                v = env.vehicle.get_velocity() if env.vehicle else None
                speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2) if v else 0
                print(f"  Step {step:3d} | Speed: {speed:.1f} km/h | Reward: {total_reward:.2f}")
            
            if done:
                print(f"\n🏁 Episode finished! Total Reward: {total_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    run_benchmark()
