import os
import numpy as np
import torch
import random
import time
import gym
from gym import spaces
try:
    import cv2
except ImportError:
    cv2 = None

# Try to find and import CARLA
try:
    import carla
except ImportError:
    import sys
    import glob
    egg_file = '/home/tinkerspace/carla project/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
    if os.path.exists(egg_file):
        sys.path.append(egg_file)
    import carla

class CarlaSyncManager:
    """Manages synchronous sensor data collection for CARLA 0.9.13"""
    def __init__(self, world, sensors, fps=10):
        self.world = world
        self.sensors = sensors
        self.delta_seconds = 1.0 / fps
        self._queues = []
        self.frame = None

        # Enable sync mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_seconds
        self.world.apply_settings(settings)
        
        self.setup_queues()

    def setup_queues(self):
        """Initializes queues for the world and all current sensors"""
        import queue
        self._queues = []
        
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout=2.0):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        return data

    def _retrieve_data(self, q, timeout):
        # Increased robustness: check frame numbers correctly
        while True:
            try:
                data = q.get(timeout=timeout)
                if data.frame == self.frame:
                    return data
                elif data.frame > self.frame:
                    return data # Already past
            except Exception:
                return None

    def cleanup(self):
        try:
            self.world.apply_settings(self.original_settings)
        except:
            pass

class CarlaEnv(gym.Env):
    """
    Gymnasium environment for CARLA 0.9.13 with Curriculum Learning support.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        super(CarlaEnv, self).__init__()
        self.config = config or {}
        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 2000)
        self.town = self.config.get("map", "Town01")
        self.fps = self.config.get("fps", 10)
        self.width = self.config.get("width", 128)
        self.height = self.config.get("height", 128)
        self.show_display = self.config.get("show_display", True)
        self.headless = not self.show_display # Inferred from display flag
        
        # RL spaces: Flat Vector for MlpPolicy (12 nav/light + 32 lidar)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count = 0
        
        # Display settings
        self.show_display = self.config.get("show_display", True)
        if cv2 is None:
            self.show_display = False

        # CARLA initialization
        print(f"Connecting to CARLA server at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(40.0)
        
        try:
            current_world = self.client.get_world()
            current_map = current_world.get_map().name
            if self.town in current_map:
                print(f"✅ World {self.town} is already loaded.")
                self.world = current_world
            else:
                print(f"🔄 Loading world {self.town}...")
                self.world = self.client.load_world(self.town)
                print(f"✅ Loaded {self.town}")
        except Exception as e:
            print(f"⚠️ World handle error: {e}")
            self.world = self.client.get_world()
            print(f"Using: {self.world.get_map().name}")

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.sensors = []
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.sync_manager = None
        self.obstacle_dist = 50.0
        self._stage_complete = False
        self.spectator = self.world.get_spectator() # Cache spectator

    def reset(self):
        self._stage_complete = False
        self._cleanup_actors()
        
        # 1. Validate and Pick Spawn Point
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            print("❌ ERROR: No spawn points found in this map!")
            # Fallback to a manual transform if empty
            spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2))
        else:
            spawn_point = random.choice(spawn_points)
            print(f"📍 Selected Spawn Point: {spawn_point.location}")

        # 2. Spawn Ego-Vehicle with Retry Logic
        vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        # Set Maroon Color (RGB approx for maroon is 128, 0, 0)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '128,0,0')

        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"🚗 Ego-Vehicle Spawned (Maroon) with ID: {self.vehicle.id}")
        except Exception as e:
            print(f"⚠️ Failed to spawn vehicle at primary point: {e}. Retrying with random selection...")
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if self.vehicle:
                print(f"✅ Retry Successful. Ego-Vehicle ID: {self.vehicle.id}")
            else:
                raise RuntimeError("❌ ERROR: Could not spawn ego-vehicle anywhere!")

        # 3. Asynchronous Stability Delay
        # Give physics engine time to place the car on the ground
        print("⏲️ Waiting for physics stabilization...", flush=True)
        # Tick the world manually if in sync mode
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        time.sleep(1.0) # More time for physics

        # 4. Setup sync manager
        self.sync_manager = CarlaSyncManager(self.world, [], fps=self.fps)
        
        # 5. Setup sensors
        self._setup_sensors()
        
        # 6. Update sync manager (No sensors needed for sync now)
        self.sync_manager.sensors = [] 
        self.sync_manager.setup_queues()
        self.sync_manager.tick()
        
        # 7. Move Spectator for Visualization
        self._set_spectator_follow()
        
        return self._get_obs()

    def _set_spectator_follow(self):
        """Moves the CARLA spectator to look at the ego-vehicle"""
        if self.headless: return # Save resources
        try:
            if self.vehicle is None or not self.vehicle.is_alive:
                return
            
            transform = self.vehicle.get_transform()
            # Position camera behind and above the car
            fwd = transform.get_forward_vector()
            location = transform.location - fwd * 8.0 + carla.Location(z=4.0)
            rotation = transform.rotation
            rotation.pitch = -25.0
            
            self.spectator.set_transform(carla.Transform(location, rotation))
        except Exception as e:
            pass

    def step(self, action):
        if self.vehicle is None:
            return self.reset(), 0, True, {}

        self.step_count += 1
        steer = float(action[0])
        throttle_brake = float(action[1])
        
        control = carla.VehicleControl()
        control.steer = steer
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = abs(throttle_brake)
        
        self.vehicle.apply_control(control)
        self.obstacle_dist = 50.0 # Reset for this frame
        snapshot, *sensor_data = self.sync_manager.tick()
        
        # Sync Spectator every step for smooth tracking
        self._set_spectator_follow()
        
        obs = self._get_obs(sensor_data)
        
        # Live Visualization
        if self.show_display:
            self._visualize(obs)
        reward, info = self._compute_reward()
        done = bool(self.collision_hist)
        
        if info.get("reward_route", 0) > 0.9:
             self._stage_complete = True

        reward = float(np.nan_to_num(reward))
        return obs, reward, done, info

    def _setup_sensors(self):
        # Obstacle Detector (Simpler & Safer than LIDAR)
        obs_bp = self.blueprint_library.find('sensor.other.obstacle')
        obs_bp.set_attribute('distance', '50')
        obs_bp.set_attribute('hit_radius', '0.5')
        obs_bp.set_attribute('only_dynamics', 'True')
        
        obs_transform = carla.Transform(carla.Location(x=1.6, z=1.0))
        self.seg_sensor = self.world.spawn_actor(
            obs_bp, 
            obs_transform, 
            attach_to=self.vehicle, 
            attachment_type=carla.AttachmentType.Rigid
        )
        self.seg_sensor.listen(lambda event: self._on_obstacle(event))
        self.sensors.append(self.seg_sensor)
        print(f"📡 Obstacle Sensor Attached with ID: {self.seg_sensor.id}")
        
    def _on_obstacle(self, event):
        if event.actor.id == self.vehicle.id: return # Ignore self
        self.obstacle_dist = min(self.obstacle_dist, event.distance)
        
        # Collision sensor
        col_bp = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(
            col_bp, 
            carla.Transform(), 
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.col_sensor.listen(lambda event: self.collision_hist.append(event))
        self.sensors.append(self.col_sensor)

    def _get_obs(self, sensor_data=None):
        # 44-DIMENSIONAL VECTOR OBSERVATION
        # [0-9]: Navigation (Speed, Control)
        # [10]: Traffic Light State
        # [11]: Traffic Light Distance
        # [12-43]: LIDAR 32-Sector Distances
        
        obs = np.zeros(44, dtype=np.float32)
        
        # 1. Navigation State
        if self.vehicle:
            v = self.vehicle.get_velocity()
            speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
            obs[0] = np.clip(np.nan_to_num(speed / 20.0), 0.0, 5.0)
            obs[1] = np.clip(np.nan_to_num(self.vehicle.get_control().steer), -1.0, 1.0)
            obs[2] = np.clip(np.nan_to_num(self.vehicle.get_control().throttle), 0.0, 1.0)
            
            # 2. Traffic Light (Phase 3B)
            tl = self.vehicle.get_traffic_light()
            if tl:
                state = tl.get_state()
                if state == carla.TrafficLightState.Red: obs[10] = 0.0
                elif state == carla.TrafficLightState.Yellow: obs[10] = 0.5
                else: obs[10] = 1.0 # Green
                
                # Distance
                # trigger_volume_extent = tl.get_trigger_volume().extent
                # Simple dist:
                # obs[11] = dist_normalized 
                obs[11] = 1.0 # Placeholder for now, assume close if affected
            else:
                obs[10] = 1.0 # Green by default
                
        # 3. Obstacle Processing (Replaced LIDAR)
        # We fill the forward sectors with the obstacle distance
        # Obs[12-43] (32 sectors). Let's say indices 14-18 are "front".
        # For simplicity, we fill ALL sectors with the nearest distance because the sensor is non-directional (or forward only)
        # Normalized distance
        norm_dist = np.clip(self.obstacle_dist / 50.0, 0.0, 1.0)
        obs[12:] = norm_dist 

        return obs

    def _compute_reward(self):
        r = 0.0
        info = {}
        if self.vehicle is None:
            return 0.0, info

        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2) 
        r_speed = 1.0 - (abs(speed - 30.0) / 30.0)
        r += max(-1.0, min(1.0, r_speed))
        info["reward_speed"] = float(np.nan_to_num(r_speed))
        info["reward_route"] = 0.0 
        r = float(np.nan_to_num(r))
        
        # Traffic Light Checks (Phase 3B)
        tl_state = "Green"
        traffic_light = self.vehicle.get_traffic_light()
        if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
            tl_state = "Red"
            if speed > 1.0: # Running global red light logic
                r -= 2.0
                info["penalty_red_light"] = -2.0
        
        # New: Log if reward is calculated
        if random.random() < 0.05: # Sample log (Increased freq for visibility)
             print(f"📊 Reward Sample: {r:.2f} (Speed: {speed:.1f} km/h) [Light: {tl_state}]", flush=True)
             
        return r, info

    def _cleanup_actors(self):
        """Robust cleanup to prevent segfaults on map change"""
        if self.sync_manager:
            try:
                self.sync_manager.cleanup()
                self.sync_manager = None
            except: pass
            
        # 1. Stop all sensors first
        for s in self.sensors:
            if s and s.is_alive:
                try: s.stop()
                except: pass
                
        # 2. Batch Destroy
        batch = []
        # Sensors
        for s in self.sensors:
            if s and s.is_alive:
                batch.append(carla.command.DestroyActor(s))
        # Vehicle
        if self.vehicle and self.vehicle.is_alive:
            batch.append(carla.command.DestroyActor(self.vehicle))
            
        if batch and self.client:
            try:
                self.client.apply_batch(batch)
                self.client.apply_batch(batch)
            except RuntimeError as e:
                print(f"⚠️ Cleanup Warning: {e}")
            time.sleep(0.2) # Allow server to process destruction
            
        # 3. Clear references
        self.sensors = []
        self.vehicle = None
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.rgb_sensor = None
        self.seg_sensor = None

    def _visualize(self, obs):
        """Shows the sensor data in a small OpenCV window"""
        if cv2 is None: return
        
        # 0. Initialize window if needed
        window_name = "AGENT_PERCEPTION_FEED"
        
        if "semantic_segmentation" not in obs:
            if self.step_count % 100 == 0:
               print(f"📡 Vector Observation: {obs[:5]}...", flush=True)
            return
            
        # 1. Perception View (Semantic)
        seg = obs["semantic_segmentation"]  # (H, W, 1)
        
        # Apply CityScapes-like Palette for better detail
        # 0=Unlabeled, 1=Building, 2=Fence, 3=Other, 4=Pedestrian, 5=Pole, 6=RoadLine, 7=Road, 8=Sidewalk, 9=Vegetation, 10=Vehicles, 12=TrafficSign, 18=TrafficLight
        # We'll map these to BGR colors
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        color_map[7] = [128, 64, 128]   # Road (Purple)
        color_map[6] = [244, 35, 232]   # RoadLine (Pink)
        color_map[10] = [142, 0, 0]     # Vehicles (Dark Blue)
        color_map[4] = [220, 20, 60]    # Pedestrian (Red)
        color_map[12] = [220, 220, 0]   # TrafficSign (Yellow)
        color_map[18] = [250, 170, 30]  # TrafficLight (Orange)
        color_map[9] = [107, 142, 35]   # Vegetation (Green)
        
        # Apply custom colormap
        seg_color = cv2.applyColorMap(seg, color_map)
        
        # Resize for display
        seg_large = cv2.resize(seg_color, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 2. Add Info Overlay
        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        cv2.putText(seg_large, f"SPEED: {speed:.1f} KM/H", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(seg_large, "SEMANTIC: 128x128 (Road=Purple, Car=Blue, Sign=Yel)", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 3. Show Window
        cv2.imshow(window_name, seg_large)
        cv2.waitKey(1)

    def close(self):
        self._cleanup_actors()
        if self.show_display and cv2:
            try: cv2.destroyAllWindows()
            except: pass

def make_carla_env(config=None):
    return CarlaEnv(config)
