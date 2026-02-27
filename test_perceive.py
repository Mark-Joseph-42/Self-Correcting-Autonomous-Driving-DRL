import os
import time
import numpy as np
import matplotlib.pyplot as plt
from carla_env import CarlaEnv

def main():
    print("Testing Local Semantic BEV...")
    
    # Configure environment
    config = {
        "host": "127.0.0.1",
        "port": 2000,
        "map": "Town01",
        "fps": 10,
        "show_display": False,  # Running headless for this test
        "enable_rewind": False,
        "enable_pedestrian_safety": False,
    }
    
    env = None
    try:
        env = CarlaEnv(config)
        print("Environment initialized.")
        
        obs = env.reset()
        print("Environment reset complete.")
        
        # Take a few steps to let things settle
        for _ in range(5):
            # Throttle a bit
            obs, reward, done, info = env.step([0.0, 0.5])
            time.sleep(0.1)
            
        print("Taking snapshot of Local BEV...")
        
        # Extract BEV grid (1, 64, 64)
        bev_grid = obs[0]
        
        # Plot and save
        plt.figure(figsize=(6, 6))
        plt.imshow(bev_grid, cmap='magma', vmin=0, vmax=1.0)
        plt.colorbar(label='Semantic Value')
        plt.title('Local Semantic BEV (10m Radius Mask)')
        
        # Draw 10m radius circle (32 pixels in 64x64 grid) for visual reference
        circle = plt.Circle((31.5, 31.5), 32, color='white', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
        
        out_file = 'bev_validation.png'
        plt.savefig(out_file)
        print(f"✅ BEV image saved to {out_file}")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
    finally:
        if env is not None:
            # Need to manually clean up to avoid memory leaks/segfaults
            env._cleanup_actors()
            print("Cleanup complete.")

if __name__ == '__main__':
    main()
