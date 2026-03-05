import os
import sys
import time
import argparse
import carla
from carla_env import CarlaEnv
from stable_baselines3 import SAC

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sac_cl_stage1", help="Model name in models/ (without .zip)")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage to run (1-3)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--map", type=str, default=None, help="Force a specific CARLA map (e.g. Town01)")
    args = parser.parse_args()

    # Build model path correctly
    if args.model.startswith("models/"):
        model_path = args.model
    else:
        model_path = os.path.join("models", args.model)
        
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model {model_path}.zip not found. Please train the agent first.")
        return

    # Initialize environment in visual mode with custom ports
    tm_port = args.port + 1000  # e.g. 3000 -> 4000
    env = CarlaEnv(port=args.port, stage=args.stage, no_render=False, tm_port=tm_port, map_name=args.map)
    
    # Load the SAC model
    try:
        model = SAC.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    
    print(f"Starting demonstration (Stage {args.stage}). Press Ctrl+C to stop.")
    try:
        while True:
            obs, _ = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                # Move spectator to follow the vehicle
                spectator = env.world.get_spectator()
                transform = env.vehicle.get_transform()
                # 3rd person camera
                forward_vec = transform.get_forward_vector()
                cam_loc = transform.location - carla.Location(x=forward_vec.x * 6, y=forward_vec.y * 6, z=-3)
                spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)))
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                speed = info.get('speed', 0)
                dist = info.get('dist_traveled', 0)
                print(f"\rSpeed: {speed:4.1f} km/h | Distance: {dist:5.1f} m", end="")
                
            print(f"\nEpisode finished. Reason: {info.get('reason', 'unknown')}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping demo.")
    finally:
        env.close()

if __name__ == "__main__":
    run_demo()
