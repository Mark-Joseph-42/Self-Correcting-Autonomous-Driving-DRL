import sys
import os
import glob
egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if os.path.exists(egg_file):
    sys.path.append(egg_file)
else:
    print(f"Egg file not found: {egg_file}")

import carla
try:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print(f"✅ Connection successful!")
    print(f"Current Map: {world.get_map().name}")
    print(f"Number of Actors: {len(world.get_actors())}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
