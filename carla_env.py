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

class SafetyMonitor:
    """Intercepts dangerous actions and logs interventions for Phase 3 Proof."""
    def __init__(self):
        self.intervention_count = 0
        self.near_miss_count = 0
    
    def check_safety(self, info):
        # A rough heuristic for near-miss/intervention based on distance to center or obstacles
        # In this project, we primarily track if the agent *would* have failed.
        if info.get("penalty_collision", 0) < 0:
            self.intervention_count += 1
            return True
        return False

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
        
        # RL spaces: MultiInput (Image + Vector)
        self.observation_space = spaces.Dict({
            "semantic": spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.prev_steer = 0.0
        
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
        self.safety_monitor = SafetyMonitor()
        self.latest_semantic = np.zeros((1, 64, 64), dtype=np.uint8)
        self._stage_complete = False
        self.spectator = self.world.get_spectator() # Cache spectator

    def reset(self):
        print("\n🔄 reset() called", flush=True)
        self._cleanup_actors()
        
        # 1. Validate and Pick Spawn Point
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            print("❌ ERROR: No spawn points found in this map!")
            # Fallback to a manual transform if empty
            spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2))
        else:
            spawn_point = random.choice(spawn_points)
            
            # Apply randomized offset for recovery training (Stage 1)
            spawn_offset_range = self.config.get("spawn_offset_range", 0.0)
            if spawn_offset_range > 0:
                # Random lateral offset (assuming Y is horizontal relative to orientation)
                # This is a bit simplistic, but usually maps are aligned
                # For robustness, we could use the orientation vector
                offset = random.uniform(-spawn_offset_range, spawn_offset_range)
                fwd = spawn_point.get_forward_vector()
                right = carla.Vector3D(-fwd.y, fwd.x, 0) # Rotate 90 deg
                spawn_point.location += right * offset
                print(f"🔀 Applied spawn offset: {offset:.2f}m")
                
            print(f"📍 Selected Spawn Point: {spawn_point.location}")

        # 2. Spawn Ego-Vehicle with Retry Logic
        vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        # Set Maroon Color (RGB approx for maroon is 128, 0, 0)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '128,0,0')

        print(f"🥚 Attempting to spawn actor at {spawn_point.location}...", flush=True)
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
        
        print("✅ reset() returning obs", flush=True)
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
        if self.step_count % 100 == 0:
            print(f"👣 Step {self.step_count}", flush=True)
        self.step_count += 1
        if self.vehicle is None:
            return self.reset(), 0, True, {}

        steer = float(action[0])
        throttle_brake = float(action[1])
        
        control = carla.VehicleControl()
        control.steer = steer
        if throttle_brake >= 0:
            # Throttle Boost: Map [0, 1] -> [0.3, 1.0] to overcome static friction
            control.throttle = 0.3 + (0.7 * throttle_brake)
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
        
        # Track safety interventions
        if self.safety_monitor.check_safety(info):
            info["intervention_active"] = True
            
        info["total_interventions"] = self.safety_monitor.intervention_count
        
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
        # obs_bp.set_attribute('only_dynamics', 'True') # Disabled to see walls/poles
        
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
        
        # Semantic Segmentation Camera
        seg_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_cam_bp.set_attribute('image_size_x', '64')
        seg_cam_bp.set_attribute('image_size_y', '64')
        seg_cam_bp.set_attribute('fov', '90')
        seg_cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.semantic_sensor = self.world.spawn_actor(
            seg_cam_bp,
            seg_cam_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.semantic_sensor.listen(self._process_semantic_img)
        self.sensors.append(self.semantic_sensor)
        print(f"👁️ Semantic Camera Attached with ID: {self.semantic_sensor.id}")
        
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
        
        print("🎥 All sensors attached and ready", flush=True)
        # RGB Camera (Visualization Only - Not used by agent)
        if self.show_display:
            cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', '800')
            cam_bp.set_attribute('image_size_y', '600')
            cam_bp.set_attribute('fov', '90')
            cam_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
            
            self.rgb_sensor = self.world.spawn_actor(
                cam_bp,
                cam_transform,
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
            self.rgb_sensor.listen(self._process_rgb_img)
            self.sensors.append(self.rgb_sensor)
            print(f"📷 Visualization RGB Camera Attached ID: {self.rgb_sensor.id}")
            
    def _process_rgb_img(self, image):
        """Standard Async Callback for RGB Camera"""
        if not self.show_display: return
        # Debug Print (Throttle to avoid spam)
        if self.step_count % 100 == 0:
             print(f"📸 RGB Camera Callback! Frame: {image.frame} | Bytes: {len(image.raw_data)}")
        
        i = np.array(image.raw_data)
        # Reshape based on the dimensions set in _setup_sensors (800x600)
        i2 = i.reshape((600, 800, 4))
        self.latest_image = i2[:, :, :3] # BGRA -> BGR
        
    def _process_semantic_img(self, image):
        """Processes semantic segmentation for the agent's 'brain'"""
        i = np.array(image.raw_data)
        # CARLA semantic image is BGRA, but A is the tag
        i2 = i.reshape((64, 64, 4))
        # Mask: Road=7, RoadLine=6 in Tag (usually index 2 or 3 depending on version)
        # In CARLA 0.9.13, tags are in the Red channel? No, Tag is in Red channel.
        tags = i2[:, :, 2] # This might vary. Let's use standard mapper.
        
        # Standardized [0, 255] grid
        labeled = np.zeros_like(tags, dtype=np.uint8)
        labeled[tags == 7] = 255  # Road
        labeled[tags == 6] = 255  # Road Lines
        self.latest_semantic = labeled[np.newaxis, :, :] # (1, 64, 64)
        
    def _on_obstacle(self, event):
        if self.vehicle is None or not self.vehicle.is_alive: return 
        if event.actor.id == self.vehicle.id: return # Ignore self
        self.obstacle_dist = min(self.obstacle_dist, event.distance)

    def _get_obs(self, sensor_data=None):
        # 44-DIMENSIONAL VECTOR OBSERVATION
        # [0-9]: Navigation (Speed, Control)
        # [10]: Traffic Light State
        # [11]: Traffic Light Distance
        # [12-43]: LIDAR 32-Sector Distances
        
        obs = np.zeros(44, dtype=np.float32)
        # Image is updated asynchronously via _process_rgb_img
        
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

        # 4. Speed Limit (NEW)
        speed_limit = self.vehicle.get_speed_limit() if self.vehicle else 30.0
        obs[9] = np.clip(speed_limit / 120.0, 0.0, 1.0)

        # Return as Dict
        semantic = getattr(self, "latest_semantic", np.zeros((1, 64, 64), dtype=np.uint8))
        
        # Normalize to float32 [0, 1] to ensure NNPack compatibility and stable optimization
        semantic_float = semantic.astype(np.float32) / 255.0
        
        return {"semantic": semantic_float, "vector": obs}

    def _compute_reward(self):
        """
        RESTORED ORIGINAL from 50% branch + gentle lane centering.
        The original worked - agent drove at 30km/h.
        Adding only ONE refinement: lane centering bonus.
        """
        r = 0.0
        info = {}
        if self.vehicle is None:
            return 0.0, info

        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2) 
        
        # 1. Target Speed Reward (peak at 30 km/h) - ORIGINAL
        r_speed = 1.0 - (abs(speed - 30.0) / 30.0)
        
        # 2. Dense Movement Reward (bonus for ANY forward motion) - ORIGINAL
        r_movement = 0.02 * min(speed, 50.0)  # +1.0 at 50 km/h
        
        r += max(-1.0, min(1.0, r_speed)) + r_movement
        info["reward_speed"] = float(np.nan_to_num(r_speed))
        info["reward_movement"] = float(np.nan_to_num(r_movement))
        
        # 3. GENTLE Lane Centering (NEW - only when moving)
        # This is the ONLY addition to the original
        if speed > 10.0:  # Only apply when actually driving
            waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)
            vehicle_loc = self.vehicle.get_location()
            lane_center = waypoint.transform.location
            lane_dir = waypoint.transform.get_forward_vector()
            
            to_vehicle = carla.Vector3D(vehicle_loc.x - lane_center.x, vehicle_loc.y - lane_center.y, 0)
            lateral_dist = abs(to_vehicle.x * (-lane_dir.y) + to_vehicle.y * lane_dir.x)
            
            # Small bonus for staying centered (max +0.2 at center)
            r_lane = 0.2 * max(0, 1.0 - (lateral_dist / 2.0))
            r += r_lane
            info["reward_lane"] = float(r_lane)
        else:
            lateral_dist = 0.0
        
        r = float(np.nan_to_num(r))
        
        # Traffic Light Checks - ORIGINAL
        tl_state = "Green"
        traffic_light = self.vehicle.get_traffic_light()
        if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
            tl_state = "Red"
            if speed > 1.0:
                r -= 2.0
                info["penalty_red_light"] = -2.0
                
        # Stationary Penalty - ORIGINAL
        if speed < 1.0:
            r -= 0.5
            info["penalty_stationary"] = -0.5
            
        # Collision Penalty - ORIGINAL
        if len(self.collision_hist) > 0:
            r -= 10.0
            info["penalty_collision"] = -10.0
        
        # Reward logging
        if random.random() < 0.05:
             print(f"📊 R:{r:.2f} | Spd:{speed:.1f} | Lane:{lateral_dist:.2f}m", flush=True)
             
        return r, info

    def _cleanup_actors(self):
        """Robust cleanup to prevent segfaults on map change"""
        
        # 0. FORCE ASYNC MODE (Critical due to Signal 11 crashes)
        if self.world:
             try:
                 settings = self.world.get_settings()
                 settings.synchronous_mode = False
                 settings.fixed_delta_seconds = None
                 self.world.apply_settings(settings)
             except: pass

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
        
        # Allow callbacks to finish (Critical for Segfault prevention)
        if self.sensors:
            time.sleep(0.2)
                
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
        if cv2 is None or not self.show_display: return
        
        # 0. Initialize window if needed
        window_name = "AGENT_PERCEPTION_FEED"
        
        # Vector Observation Visualization
        # Use RGB Camera image if available, else black canvas
        if hasattr(self, 'latest_image') and self.latest_image is not None:
            canvas = np.ascontiguousarray(self.latest_image)
        else:
            canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # Extract vector from Dict observation
        vector = obs["vector"] if isinstance(obs, dict) else obs
        
        # Traffic Light
        tl_state = "Green"
        if vector[10] < 0.2: tl_state = "Red"
        elif vector[10] < 0.8: tl_state = "Yellow"
        
        cv2.putText(canvas, f"SPEED: {speed:.1f} KM/H", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"LIGHT: {tl_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f"STEER: {vector[1]:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"THROTTLE: {vector[2]:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Obstacle
        min_obs_dist = np.min(vector[12:]) * 50.0
        cv2.putText(canvas, f"NEAREST OBS: {min_obs_dist:.1f}m", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        self._cleanup_actors()
        if self.show_display and cv2:
            try: cv2.destroyAllWindows()
            except: pass

def make_carla_env(config=None):
    return CarlaEnv(config)
