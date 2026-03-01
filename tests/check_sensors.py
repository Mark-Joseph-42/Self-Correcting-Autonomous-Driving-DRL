import carla
import time
import queue

def check():
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        # Spawn vehicle
        bp = bp_lib.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        ego = world.spawn_actor(bp, spawn_point)
        print("Vehicle spawned.")
        
        # Spawn camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=2.0)), attach_to=ego)
        
        q = queue.Queue()
        cam.listen(q.put)
        print("Camera listening...")
        
        # Tick world
        settings = world.get_settings()
        settings.synchronous_mode = True
        world.apply_settings(settings)
        
        world.tick()
        print("Ticked world.")
        
        try:
            image = q.get(timeout=2.0)
            print(f"Captured image: {image}")
            return True
        except queue.Empty:
            print("FAILED: No image received (sensor likely non-functional in -nullrhi)")
            return False
        finally:
            ego.destroy()
            cam.destroy()
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    res = check()
    exit(0 if res else 1)
