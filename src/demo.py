import os
import time
import numpy as np
import cv2
import pandas as pd
from stable_baselines3 import SAC
from src.carla_env import CarlaEnv
from src.config import CHECKPOINT_DIR, LOG_DIR

# Custom Env for Visualization (with RGB camera)
class DemoEnv(CarlaEnv):
    def __init__(self, **kwargs):
        super(DemoEnv, self).__init__(**kwargs)
        self.output_dir = os.path.join(LOG_DIR, "demo_baseline")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _setup_sensors(self):
        super(DemoEnv, self)._setup_sensors()
        import carla
        # Add RGB front-facing camera for the user
        bp_lib = self.world.get_blueprint_library()
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '800')
        rgb_bp.set_attribute('image_size_y', '600')
        rgb_bp.set_attribute('fov', '100')
        
        rgb_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.rgb_camera = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.ego_vehicle)
        self.rgb_camera.listen(self._create_queue('rgb').put)
        self.sensors.append(self.rgb_camera)

    def run_demo(self, model_path, max_steps=100):
        model = SAC.load(model_path)
        obs = self.reset(town="Town03", weather="HardRainNoon", traffic_pct=20)
        
        telemetry = []
        print(f"Starting demo run for {model_path}...")
        
        for i in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.step(action)
            
            # Save RGB frame every 10 steps
            if i % 10 == 0:
                try:
                    rgb_img = self.sensor_queues['rgb'].get(timeout=2.0)
                    array = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (rgb_img.height, rgb_img.width, 4))
                    array = array[:, :, :3] # BGR
                    cv2.imwrite(os.path.join(self.output_dir, f"frame_{i:03d}.jpg"), array)
                    print(f"Saved frame {i}")
                except:
                    pass
            
            telemetry.append({
                'step': i,
                'steer': action[0],
                'throttle': action[1],
                'reward': reward,
                'speed': info['speed'],
                'd_lat': info['d_lat'],
                'collision': info['collision']
            })
            
            if done:
                print(f"Episode ended at step {i}")
                # Save final frame
                try:
                    rgb_img = self.sensor_queues['rgb'].get(timeout=2.0)
                    array = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (rgb_img.height, rgb_img.width, 4))
                    array = array[:, :, :3]
                    cv2.imwrite(os.path.join(self.output_dir, "frame_final.jpg"), array)
                except:
                    pass
                break
                
        pd.DataFrame(telemetry).to_csv(os.path.join(self.output_dir, "telemetry.csv"), index=False)
        print("Demo run complete.")

if __name__ == "__main__":
    baseline_path = os.path.join(CHECKPOINT_DIR, "baseline_final.zip")
    if os.path.exists(baseline_path):
        env = DemoEnv()
        env.run_demo(baseline_path, max_steps=200)
        env.close()
    else:
        print(f"Baseline model not found at {baseline_path}")
