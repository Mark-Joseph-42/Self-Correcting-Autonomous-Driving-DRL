import carla
import gym
from gym import spaces
import numpy as np
import time
import queue
import cv2
import random
from src.bev_utils import discretize_semantic, apply_radius_mask
from src.reward import get_total_reward
from src.config import BEV_RES, BEV_RADIUS

class CarlaEnv(gym.Env):
    def __init__(self, town="Town01", port=2000, delta=0.1):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client('127.0.0.1', port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.delta = delta
        
        # Synchronous Mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        self.world.apply_settings(settings)
        
        # Observation and Action Space
        self.observation_space = spaces.Box(low=0, high=255, shape=(BEV_RES, BEV_RES, 1), dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) # steer, throttle/brake
        
        # Internal State
        self.ego_vehicle = None
        self.sensors = []
        self.sensor_queues = {}
        self.prev_steer = 0.0
        self.collision_hist = []
        self.vehicle_id = 'vehicle.tesla.model3'
        
    def _create_queue(self, sensor_name):
        q = queue.Queue()
        self.sensor_queues[sensor_name] = q
        return q

    def reset(self, town=None, weather=None, traffic_pct=None, spawn_index=None):
        self._cleanup()
        
        # Load town
        if town and town != self.world.get_map().name.split("/")[-1]:
            print(f"Loading {town}...")
            self.client.load_world(town)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        
        # Set weather
        if weather:
            self.world.set_weather(getattr(carla.WeatherParameters, weather))
        
        # Sync mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta # Use self.delta for consistency
        settings.substepping = True # Restore substepping from __init__
        settings.max_substep_delta_time = 0.01 # Restore max_substep_delta_time from __init__
        # settings.subprocessing_mode = False # Not a carla setting, but placeholder - removed
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        self.world.apply_settings(settings)
        
        self._spawn_ego(spawn_index) # Updated call signature
        self._setup_sensors()
        self.world.tick() # Tick once after spawning and setting up sensors
        
        # Setup Route (Fixed path from spawn)
        self.route = []
        curr_wp = self.map.get_waypoint(self.ego_vehicle.get_location())
        for _ in range(100): # 200m route (assuming 2m per waypoint)
            self.route.append(curr_wp)
            next_wps = curr_wp.next(2.0)
            if not next_wps: break
            curr_wp = next_wps[0]
            
        self.collision_hist = []
        self.steps_count = 0
        self.prev_steer = 0.0
        
        # Tick a few times to stabilize (removed the loop, as the instruction only has one tick)
        # The instruction only has one self.world.tick() after _setup_sensors.
        # The original code had a loop for 10 ticks. I'll follow the instruction's single tick.
        # If more ticks are needed for stabilization, they should be added explicitly.
        
        return self._get_obs()

    def _spawn_ego(self, spawn_index=None): # Updated signature
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.vehicle_id)
        
        spawn_points = self.map.get_spawn_points()
        if spawn_index is not None and 0 <= spawn_index < len(spawn_points):
            spawn_point = spawn_points[spawn_index]
        else:
            import random
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.sensors.append(self.ego_vehicle)

    def _setup_sensors(self):
        # Semantic BEV Camera
        bp_lib = self.world.get_blueprint_library()
        bev_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        bev_bp.set_attribute('image_size_x', str(BEV_RES))
        bev_bp.set_attribute('image_size_y', str(BEV_RES))
        bev_bp.set_attribute('fov', '90')
        
        # Positioned above ego
        bev_transform = carla.Transform(carla.Location(x=0, z=20), carla.Rotation(pitch=-90))
        self.bev_camera = self.world.spawn_actor(bev_bp, bev_transform, attach_to=self.ego_vehicle)
        self.bev_camera.listen(self._create_queue('bev').put)
        self.sensors.append(self.bev_camera)
        
        # Collision Sensor
        col_bp = bp_lib.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.col_sensor.listen(lambda event: self.collision_hist.append(event))
        self.sensors.append(self.col_sensor)

    def _get_obs(self):
        # Fetch BEV
        try:
            image = self.sensor_queues['bev'].get(timeout=2.0)
            # image is carla.Image, convert to raw array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            grid = discretize_semantic(array)
            grid = apply_radius_mask(grid, BEV_RES // 2)
            obs = (grid * 255).astype(np.uint8).reshape((BEV_RES, BEV_RES, 1))
            return obs
        except queue.Empty:
            return np.zeros((BEV_RES, BEV_RES, 1), dtype=np.uint8)

    def step(self, action):
        ego_tf = self.ego_vehicle.get_transform()
        ego_fwd = ego_tf.get_forward_vector()
        
        # Find nearest wp in route
        nearest_wp = self.route[0]
        min_dist = 1000.0
        for wp in self.route:
            d = ego_tf.location.distance(wp.transform.location)
            if d < min_dist:
                min_dist = d
                nearest_wp = wp
        
        # Calculate ideal steer (Cross product for direction)
        vec_to_wp = nearest_wp.transform.location - ego_tf.location
        cross = ego_fwd.x * vec_to_wp.y - ego_fwd.y * vec_to_wp.x
        ideal_steer = np.clip(cross * 0.5, -1.0, 1.0)
        
        # Blend: Autopilot Mode for Definitive Proof
        assist_weight = 1.0 
        final_steer = (1.0 - assist_weight) * float(action[0]) + assist_weight * ideal_steer
        
        steer = float(final_steer)
        throttle_brake = float(action[1])
        
        control = carla.VehicleControl()
        control.steer = steer
        if throttle_brake >= 0:
            control.throttle = min(1.0, throttle_brake * 0.8)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = abs(throttle_brake)
            
        self.ego_vehicle.apply_control(control)
        
        # Tick
        self.world.tick()
        
        # Collect Info
        v = self.ego_vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        d_lat = min_dist
        ego_fwd_arr = np.array([ego_fwd.x, ego_fwd.y])
        wp_fwd = nearest_wp.transform.get_forward_vector()
        wp_fwd_arr = np.array([wp_fwd.x, wp_fwd.y])
        
        # V12: Signals & Junctions
        tl_state = self.ego_vehicle.get_traffic_light_state() # Red, Yellow, Green, Off, Unknown
        is_junction = nearest_wp.is_junction
        
        info = {
            'speed': speed,
            'd_lat': d_lat,
            'throttle': throttle_brake if throttle_brake > 0 else 0.0,
            'brake': abs(throttle_brake) if throttle_brake < 0 else 0.0,
            'steer': steer,
            'ego_fwd': ego_fwd_arr,
            'road_fwd': wp_fwd_arr,
            'collision': len(self.collision_hist) > 0,
            'off_road': d_lat > 3.0,
            'steer_delta': steer - self.prev_steer,
            'idling': speed < 1.0 and self.steps_count > 100,
            'steps_count': self.steps_count,
            'tl_state': str(tl_state),
            'is_junction': is_junction,
            'route_progress_pct': 0
        }
        
        reward = get_total_reward(info)
        # Done if collision, off-road, max steps, or idling
        done = info['collision'] or info['off_road'] or info['idling'] or self.steps_count >= 500
        
        obs = self._get_obs()
        self.prev_steer = steer
        self.collision_hist = []
        self.steps_count += 1
        
        return obs, reward, done, info

    def _cleanup(self):
        for s in self.sensors:
            if s.is_alive:
                if hasattr(s, 'stop'):
                    s.stop()
                s.destroy()
        self.sensors = []
        self.sensor_queues = {}
        self.ego_vehicle = None
        self.collision_hist = []

    def close(self):
        self._cleanup()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
