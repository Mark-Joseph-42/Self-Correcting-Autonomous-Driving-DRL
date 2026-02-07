import os
import glob
import sys
import random
import time
import queue

# Hardcoded path for stability
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)

import carla

def test_mixed_sensors():
    print("Connecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    sensors = []
    vehicle = None
    
    try:
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(bp, spawn_point)
        
        # 1. Async Sensor (Collision) - Manual listener
        col_bp = blueprint_library.find('sensor.other.collision')
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        col_sensor.listen(lambda event: print(f"  Collision event detected at frame {event.frame}"))
        sensors.append(col_sensor)
        
        # 2. Sync Sensor (Camera) - Queue listener
        sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '64')
        sem_bp.set_attribute('image_size_y', '64')
        sem_sensor = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)
        
        q = queue.Queue()
        sem_sensor.listen(q.put)
        sensors.append(sem_sensor)
        
        print("Ticking 5 times...")
        for i in range(5):
            frame = world.tick()
            print(f"Tick {i}, frame {frame}")
            data = q.get(timeout=2.0)
            while data.frame < frame:
                data = q.get(timeout=2.0)
            print(f"  Received camera data for frame {data.frame}")
                    
        print("✅ Success! Mixed sensor pipeline is stable.")
        
    finally:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        for s in sensors:
            if s.is_alive: s.destroy()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        print("Cleanup complete.")

if __name__ == "__main__":
    test_mixed_sensors()
