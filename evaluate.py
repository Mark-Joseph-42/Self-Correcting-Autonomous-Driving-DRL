import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from carla_env import CarlaEnv
import carla

def record_demo_episode(env, model, output_path, max_frames=1000):
    """
    Records a single episode for demonstration/intermittent checkpointing.
    """
    # Setup 3rd person RGB camera for recording
    blueprint_library = env.world.get_blueprint_library()
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1280')
    cam_bp.set_attribute('image_size_y', '720')
    
    # Position: behind the car
    cam_transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))
    
    obs, _ = env.reset()
    # Spawn recording camera
    rec_cam = env.world.spawn_actor(cam_bp, cam_transform, attach_to=env.vehicle)
    
    frames = []
    def _process_frame(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        frames.append(array)
        
    rec_cam.listen(lambda data: _process_frame(data))
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if len(frames) > max_frames: break
        
    rec_cam.stop()
    rec_cam.destroy()
    
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10.0, (1280, 720))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Recorded video to {output_path}")

def record_evaluation(model_path, output_video, stage=3, num_episodes=3):
    print(f"Evaluating {model_path}...")
    env = CarlaEnv(stage=stage, no_render=False)
    model = SAC.load(model_path)
    
    for ep in range(num_episodes):
        output_path = f"{output_video}_ep{ep}.mp4"
        record_demo_episode(env, model, output_path)

    env.close()

def plot_reward_curves():
    print("Plotting Reward Growth Comparison...")
    cl_log = "logs/cl_rewards.csv"
    drl_log = "logs/drl_rewards.csv"
    
    plt.figure(figsize=(12, 7))
    
    # Plot CL if available
    if os.path.exists(cl_log):
        df_cl = pd.read_csv(cl_log)
        window = max(1, len(df_cl) // 20)
        df_cl['rolling_reward'] = df_cl['reward'].rolling(window=window, min_periods=1).mean()
        plt.plot(df_cl['step'], df_cl['rolling_reward'], label='Curriculum Learning (CL)', color='green', linewidth=2.5)
        
        # Add curriculum stage markers
        plt.axvline(x=10000, color='g', linestyle=':', alpha=0.5, label='CL Stage 2 (Curves)')
        plt.axvline(x=20000, color='g', linestyle='-.', alpha=0.5, label='CL Stage 3 (Obstacles)')
    
    # Plot DRL if available
    if os.path.exists(drl_log):
        df_drl = pd.read_csv(drl_log)
        window_drl = max(1, len(df_drl) // 20)
        df_drl['rolling_reward'] = df_drl['reward'].rolling(window=window_drl, min_periods=1).mean()
        plt.plot(df_drl['step'], df_drl['rolling_reward'], label='Standard DRL', color='red', linewidth=2.5, alpha=0.8)
    
    if not os.path.exists(cl_log) and not os.path.exists(drl_log):
        print("No reward logs found to plot.")
        plt.close()
        return
        
    plt.title('Reward Growth: Curriculum Learning vs Standard DRL', fontsize=16, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel(f'Average Episode Reward (Rolling Avg)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('results/reward_growth_comparison.png', dpi=300)
    print("Saved results/reward_growth_comparison.png")

def plot_success_metrics():
    print("Plotting Verification Metrics...")
    cl_log = "logs/cl_verification.csv"
    drl_log = "logs/drl_verification.csv"
    
    if not os.path.exists(cl_log) or not os.path.exists(drl_log):
        print("Verification logs missing. Skipping plot.")
        return
        
    df_cl = pd.read_csv(cl_log)
    df_drl = pd.read_csv(drl_log)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Distance Plot
    ax1.plot(df_cl['step'], df_cl['avg_distance'], 'g-o', label='CL: Avg Distance (m)', linewidth=2)
    ax1.plot(df_drl['step'], df_drl['avg_distance'], 'r-s', label='DRL: Avg Distance (m)', linewidth=2)
    ax1.axhline(y=200, color='b', linestyle='--', alpha=0.5, label='Success Threshold (200m)')
    ax1.set_title('Evaluation Distance over Training Steps', fontsize=14)
    ax1.set_ylabel('Distance (meters)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Collision Rate Plot
    ax2.plot(df_cl['step'], df_cl['collision_rate'] * 100, 'g-o', label='CL: Collision Rate %', linewidth=2)
    ax2.plot(df_drl['step'], df_drl['collision_rate'] * 100, 'r-s', label='DRL: Collision Rate %', linewidth=2)
    ax2.set_title('Collision Rate over Training Steps', fontsize=14)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Collision Rate (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/verification_metrics.png', dpi=300)
    print("Saved results/verification_metrics.png")

if __name__ == "__main__":
    if not os.path.exists("results"): os.makedirs("results")
    
    # Record CL Final Eval
    if os.path.exists("models/sac_cl_final.zip"):
        record_evaluation("models/sac_cl_final", "results/cl_evaluation", stage=3, num_episodes=3)
    
    # Record DRL Final Eval
    if os.path.exists("models/sac_drl_final.zip"):
        record_evaluation("models/sac_drl_final", "results/drl_evaluation", stage=3, num_episodes=3)
        
    # Plotting
    plot_reward_curves()
    plot_success_metrics()
