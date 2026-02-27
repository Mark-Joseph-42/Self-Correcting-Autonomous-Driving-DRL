import os
import numpy as np
import torch
import random
import time
import gym
from gym import spaces
from collections import deque
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
        
        # RL spaces: Single Input (BEV Grid)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.prev_steer = 0.0
        self.offroad_steps = 0 # Track consecutive steps off-road
        
        # Display settings
        self.show_display = self.config.get("show_display", True)
        if cv2 is None:
            self.show_display = False

        # CARLA initialization
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)
        
        try:
            current_world = self.client.get_world()
            current_map = current_world.get_map().name
            if self.town in current_map:
                self.world = current_world
            else:
                self.world = self.client.load_world(self.town)
        except Exception as e:
            self.world = self.client.get_world()

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.sensors = []
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.sync_manager = None
        self.obstacle_dist = 50.0
        self.safety_monitor = SafetyMonitor()
        self.latest_semantic = np.zeros((1, 64, 64), dtype=np.float32)
        self._stage_complete = False
        self.spectator = None
        
        # Stage specific flags
        self.target_friction = self.config.get("tire_friction", None)

        # "Turbo" & Safety Buffers
        self.state_buffer = deque(maxlen=40)
        self.green_light_passes = 0
        self.pedestrian_collision = False
        self.last_red_light_id = None

    def reset(self):
        print("\n🔄 reset() called", flush=True)
        print("🧹 Cleaning up actors...", flush=True)
        self._cleanup_actors()
        
        # 1. Validate and Pick Spawn Point
        print("📍 Picking spawn point...", flush=True)
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            print("❌ ERROR: No spawn points found in this map!")
            # Fallback to a manual transform if empty
            spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2))
        else:
            # DEBUG: Use fixed spawn if configured (faster learning of specific segments)
            fixed_spawn_idx = self.config.get("fixed_spawn_idx", -1)
            if fixed_spawn_idx >= 0 and fixed_spawn_idx < len(spawn_points):
                spawn_point = spawn_points[fixed_spawn_idx]
                print(f"📌 DEBUG: Using fixed spawn point #{fixed_spawn_idx}")
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
            
            # Reset buffers on spawn
            self.state_buffer.clear()
            self.green_light_passes = 0
            self.pedestrian_collision = False
                
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

        # 3B. Apply Tire Friction (Stage 4)
        if self.target_friction is not None:
             try:
                 print(f"🛞 Setting tire friction to {self.target_friction}...")
                 physics_control = self.vehicle.get_physics_control()
                 for wheel in physics_control.wheels:
                     wheel.tire_friction = self.target_friction
                 self.vehicle.apply_physics_control(physics_control)
                 print("✅ Tire friction applied.")
             except Exception as e:
                 print(f"⚠️ Failed to apply tire friction: {e}")

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
        self.collision_hist = [] # Clear history on reset
        return self._get_obs()

    def _set_spectator_follow(self):
        """Moves the CARLA spectator to look at the ego-vehicle"""
        if self.headless: return # Save resources
        
        if self.spectator is None:
            try:
                self.spectator = self.world.get_spectator()
            except:
                return
                
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

        # 1A. Snapshot Recovery (Save state before action)
        current_transform = self.vehicle.get_transform()
        current_velocity = self.vehicle.get_velocity()
        self.state_buffer.append((current_transform, current_velocity))

        # Fetch current location and waypoint for logic/rewards
        vehicle_loc = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Any)

        steer = float(action[0])
        throttle_brake = float(action[1])
        
        control = carla.VehicleControl()
        control.steer = steer
        
        # New Action Mapping: 
        # [0, 1] -> Throttle
        # [-0.4, 0] -> Brake
        # [-1, -0.4] -> Reverse
        if throttle_brake >= 0:
            control.throttle = 0.3 + (0.7 * throttle_brake)
            control.brake = 0.0
            control.reverse = False
        elif throttle_brake > -0.4:
            control.throttle = 0.0
            control.brake = abs(throttle_brake) * 2.5 # Scale to [0, 1]
            control.reverse = False
        else:
            # Reverse: Apply partial throttle in reverse gear
            control.throttle = (abs(throttle_brake) - 0.4) / 0.6 # Scale remainder to [0, 1]
            control.brake = 0.0
            control.reverse = True
        
        self.vehicle.apply_control(control)
        
        # 1E. Damping Layer: Pedestrian Emergency Brake (Override if dangerous)
        # Search for pedestrian (value 0.5 in our mapped BEV grid) in the immediate front zone (5m x 2m)
        # 1px = ~0.3125m. 5m = ~16px. 2m = ~6.4px (round to 6: 3px each side of center).
        # BEV Center is (31.5, 31.5). Front zone is rows ~16 to 31, cols 28 to 34.
        if self.config.get("enable_pedestrian_safety", True):
            front_hazard = False
            if hasattr(self, 'latest_semantic'):
                sem = self.latest_semantic[0] # (64, 64) grid
                front_zone = sem[16:32, 28:35]
                if np.any(front_zone == 0.5): # Pedestrian value
                    front_hazard = True
            
            if front_hazard and speed > 1.0:
                 control.throttle = 0.0
                 control.brake = 1.0
                 self.vehicle.apply_control(control)
                 # print("🚨 DAMPING LAYER: Pedestrian detected in front zone!")

        # Save previous steer for jitter penalty
        self.prev_steer = steer
        
        self.obstacle_dist = 50.0 # Reset for this frame
        snapshot, *sensor_data = self.sync_manager.tick()
        
        # Sync Spectator every step for smooth tracking
        self._set_spectator_follow()
        
        obs = self._get_obs(sensor_data)
        
        # Live Visualization
        if self.show_display:
            self._visualize(obs)
        # Update off-road tracking BEFORE reward calculation
        if waypoint.lane_type != carla.LaneType.Driving:
            self.offroad_steps += 1
        else:
            self.offroad_steps = 0

        reward, info = self._compute_reward()
        
        # Track safety interventions
        if self.safety_monitor.check_safety(info):
            info["intervention_active"] = True
            
        info["total_interventions"] = self.safety_monitor.intervention_count
        
        # Termination conditions
        done = bool(self.collision_hist) or (self.offroad_steps > 40)
        
        # Mastery-Backtrack: Only for Stage-Ending failures (e.g., Pedestrian Collision after success)
        # Stage-Ending defined heuristically: Route almost done (>90%) but collided with a pedestrian
        if self.config.get("enable_rewind", False) and bool(self.collision_hist):
            is_stage_ending_failure = (info.get("reward_route", 0) > 0.9) and self.pedestrian_collision
            
            if is_stage_ending_failure:
                print(f"⚠️ [DEBUG] Stage-Ending Collision detected. Rewinding to safe state.")
                if len(self.state_buffer) >= 20:
                     safe_transform, safe_velocity = self.state_buffer[-20]
                     self.vehicle.set_transform(safe_transform)
                     self.vehicle.set_target_velocity(safe_velocity)
                     self.collision_hist = []
                     done = False 
                else:
                     print("⚠️ Cannot rewind: Buffer too small.")
                     
        if self.offroad_steps > 40:
             print(f"🛑 EARLY RESET: Stuck off-road for {self.offroad_steps} steps.")        
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
        
        # Local Semantic BEV Camera
        seg_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_cam_bp.set_attribute('image_size_x', '64')
        seg_cam_bp.set_attribute('image_size_y', '64')
        # FOV calculation: to capture 20m span at 15m height.
        # tan(FOV/2) = (10m) / 15m => FOV/2 = Math.atan(10/15) = 33.69 deg => FOV ~ 67 deg
        seg_cam_bp.set_attribute('fov', '67')
        seg_cam_transform = carla.Transform(
            carla.Location(x=0.0, z=15.0), 
            carla.Rotation(pitch=-90.0)
        )
        self.semantic_sensor = self.world.spawn_actor(
            seg_cam_bp,
            seg_cam_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.semantic_sensor.listen(self._process_semantic_img)
        self.sensors.append(self.semantic_sensor)
        print(f"🚁 Local BEV Camera Attached with ID: {self.semantic_sensor.id}")
        
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
        
        i = np.array(image.raw_data)
        # Reshape based on the dimensions set in _setup_sensors (800x600)
        i2 = i.reshape((600, 800, 4))
        self.latest_image = i2[:, :, :3] # BGRA -> BGR
        
    def _process_semantic_img(self, image):
        """Processes Local Semantic BEV for the agent's 'brain'"""
        i = np.array(image.raw_data)
        # CARLA semantic image is BGRA
        i2 = i.reshape((64, 64, 4))
        # Tag is in Red channel for semantics usually, sometimes Green or Blue based on version.
        # Common convention for CARLA: The Blue channel contains the semantic tag if accessed via raw_data or standard conversion.
        # Actually in CARLA 0.9.13 `image.raw_data` of semantic_segmentation has the tag in the R channel if reshaping (B,G,R,A) -> R is index 2.
        tags = i2[:, :, 2] 
        
        # Grid Transformation: Map Semantic IDs
        # Road (7) -> 1.0, Vehicle (10) -> 0.8, Pedestrian (4) -> 0.5, Other -> 0.0
        labeled = np.zeros_like(tags, dtype=np.float32)
        labeled[tags == 7] = 1.0   # Road
        labeled[tags == 6] = 1.0   # Road lines (treating as road for drivable area)
        labeled[tags == 10] = 0.8  # Vehicle
        labeled[tags == 4] = 0.5   # Pedestrian
        
        # Spatial Masking: 10m radius
        # Center of 64x64 grid is (31.5, 31.5). 1px = 20m / 64 = 0.3125m.
        # 10m radius = 10 / 0.3125 = 32 pixels.
        y, x = np.ogrid[:64, :64]
        dist_from_center = np.sqrt((x - 31.5)**2 + (y - 31.5)**2)
        mask = dist_from_center <= 32
        
        # Apply mask
        labeled = labeled * mask
        
        self.latest_semantic = labeled[np.newaxis, :, :] # (1, 64, 64)
        
    def _on_obstacle(self, event):
        if self.vehicle is None or not self.vehicle.is_alive: return 
        if event.actor.id == self.vehicle.id: return # Ignore self
        self.obstacle_dist = min(self.obstacle_dist, event.distance)

    def _get_obs(self, sensor_data=None):
        # Local Semantic BEV Observation (1, 64, 64)
        semantic = getattr(self, "latest_semantic", np.zeros((1, 64, 64), dtype=np.float32))
        return semantic

    def _compute_reward(self):
        """
        Stability-Based Reward:
        - Penalizes lateral velocity (sideways motion) relative to the road.
        - Scales lane centering with speed (higher speed = stricter discipline).
        - Maintains red light and collision protection.
        """
        r = 0.0
        info = {}
        if self.vehicle is None:
            return 0.0, info

        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2) 
        
        # 1. Target Speed Reward (peak at 30 km/h)
        r_speed = 1.0 - (abs(speed - 30.0) / 30.0)
        
        # 2. Dense Movement Reward
        r_movement = 0.02 * min(speed, 50.0)
        
        # Continuity Penalty (Jitter)
        current_steer = self.vehicle.get_control().steer
        r_jitter = 0.0
        if abs(current_steer - self.prev_steer) > 0.2:
            r_jitter = -0.5 * abs(current_steer - self.prev_steer)
            
        r += max(-1.0, min(1.0, r_speed)) + r_movement + r_jitter
        info["reward_speed"] = float(np.nan_to_num(r_speed))
        info["reward_movement"] = float(np.nan_to_num(r_movement))
        info["penalty_jitter"] = float(r_jitter)
        
        # Get Waypoint Info
        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(), 
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        road_fwd = waypoint.transform.get_forward_vector()
        road_right = carla.Vector3D(-road_fwd.y, road_fwd.x, 0) # Perpendicular to road
        
        # 3. HEADING ALIGNMENT (Fixes "Circular Driving" exploit)
        # Dot product between car forward and road forward
        vehicle_fwd = self.vehicle.get_transform().get_forward_vector()
        # Dot product (1.0 = aligned, 0 = perpendicular, -1 = reversed)
        heading_align = vehicle_fwd.x * road_fwd.x + vehicle_fwd.y * road_fwd.y
        
        # Linear alignment reward/penalty
        r_align = 0.5 * heading_align # +/- 0.5
        r += r_align
        info["reward_align"] = float(r_align)

        # 1B. Look-Ahead Waypoint Reward
        vehicle_loc = self.vehicle.get_location()
        if self.config.get("enable_lookahead_reward", True):
             # Extract waypoint 5 meters ahead
             next_wps = waypoint.next(5.0)
             if next_wps:
                 next_wp = next_wps[0]
                 wp_transform = next_wp.transform
                 wp_loc = wp_transform.location
                 
                 # Vector from car to waypoint
                 vec_to_wp = carla.Location(x=wp_loc.x - vehicle_loc.x, y=wp_loc.y - vehicle_loc.y)
                 norm = np.sqrt(vec_to_wp.x**2 + vec_to_wp.y**2 + 1e-6)
                 vec_to_wp.x /= norm
                 vec_to_wp.y /= norm
                 
                 # Dot product with forward vector
                 dot_product = vehicle_fwd.x * vec_to_wp.x + vehicle_fwd.y * vec_to_wp.y
                 r_lookahead = (speed / 10.0) * dot_product # Scale with speed
                 r += r_lookahead
                 info["reward_lookahead"] = float(r_lookahead)

        # 4. LATERAL VELOCITY PENALTY (Stability Focus)
        lat_vel = abs(v.x * road_right.x + v.y * road_right.y) * 3.6 
        
        # Stricter Stability: Allowed up to 20% sideways motion
        r_stability = 0.0
        if speed > 5.0:
            stability_threshold = speed * 0.2 
            if lat_vel > stability_threshold:
                r_stability = -0.15 * (lat_vel - stability_threshold) # Stiffened from -0.05
        
        r += r_stability
        info["penalty_stability"] = float(r_stability)
        
        # 4. SPEED-SCALED LANE DISCIPLINE (Linear for easier learning)
        lane_center = waypoint.transform.location
        to_vehicle = carla.Vector3D(vehicle_loc.x - lane_center.x, vehicle_loc.y - lane_center.y, 0)
        lateral_dist = abs(to_vehicle.x * road_right.x + to_vehicle.y * road_right.y)
        
        speed_scale = max(1.0, speed / 20.0)
        r_lane = 0.4 * speed_scale * max(0, 1.0 - (lateral_dist / 1.75)) # Linear falloff
        r += r_lane
        info["reward_lane"] = float(r_lane)
        
        # Wrong Lane Penalty (Left Side)
        if waypoint.lane_id > 0:
            r -= 1.0 * speed_scale # Harder penalty at speed
            info["penalty_wrong_lane"] = -1.0 * speed_scale
            
        # Sidewalk/Off-road Penalty (Major deterrent)
        if waypoint.lane_type != carla.LaneType.Driving:
            # If not in a driving lane, penalize heavily
            r -= 2.0 * speed_scale 
            info["penalty_off_road"] = -2.0 * speed_scale
        
        r = float(np.nan_to_num(r))
        
        # 5. Traffic Light (Strong Penalty + Smooth Braking Reward)
        tl_state = "Green"
        control = self.vehicle.get_control()
        traffic_light = self.vehicle.get_traffic_light()
        
        if traffic_light:
            state = traffic_light.get_state()
            if state == carla.TrafficLightState.Red:
                tl_state = "Red"
                # 1D. Refined Red Light Penalty
                if speed > 0.1:
                    r -= 20.0
                    info["penalty_red_light"] = -20.0
                elif speed <= 2.0 and control.brake > 0.1:
                    # Reward for successfully stopping at red
                    r += 0.5
                    info["reward_red_stop"] = 0.5
            elif state == carla.TrafficLightState.Green:
                # 1F. Mastery tracking (3 green lights)
                if self.last_red_light_id != traffic_light.id:
                    self.green_light_passes += 1
                    self.last_red_light_id = traffic_light.id
                    # print(f"🚦 Green light passed! Counter: {self.green_light_passes}")
                
        # 6. Spatial Reward (Distance to Road Edge via BEV)
        # BEV maps Road=1.0. We want to reward staying central on the road and penalize getting close to edges (0.0).
        r_spatial = 0.0
        if hasattr(self, 'latest_semantic'):
            sem = self.latest_semantic[0]
            # Define "Immediate Forward Path": A wedge or rectangle in front of the car
            # Rows 16 to 31 (front 5m approx), Cols 24 to 40 (center ~5m wide)
            forward_path = sem[16:32, 24:40]
            # Calculate ratio of road pixels in the forward path
            road_ratio = np.sum(forward_path == 1.0) / forward_path.size
            
            # Reward high road ratio, penalize low road ratio (getting close to edge/off-road)
            if road_ratio < 0.8:
                r_spatial = -1.0 * (0.8 - road_ratio)
            else:
                r_spatial = 0.2 * road_ratio # Small bonus for clear path
        
        r += r_spatial
        info["reward_spatial"] = float(r_spatial)

        # 1E. Pedestrian Safety Bonus/Penalty
        r_pedestrian = 0.0
        if hasattr(self, 'latest_semantic'):
            sem = self.latest_semantic[0]
            # Detect close pedestrian (Value 0.5) in the BEV map
            # Central front area
            if np.any(sem[16:36, 20:44] == 0.5):
                 r_pedestrian = -50.0 # Heavy legal penalty
                 # Mark for backtrack if collision occurs (handled in collision hist)
                 if bool(self.collision_hist):
                      # Check if collision was with a pedestrian
                      for event in self.collision_hist:
                           if 'walker' in event.other_actor.type_id:
                                self.pedestrian_collision = True
        
        r += r_pedestrian
        info["penalty_pedestrian"] = float(r_pedestrian)

        # 7. Reverse Penalty (Discourage unless needed for recovery)
        r_reverse = -0.2 if control.reverse else 0.0
        r += r_reverse
        info["penalty_reverse"] = float(r_reverse)

        # 8. Stationary/Collision/Stuck
        if speed < 1.0:
            r -= 0.5
            info["penalty_stationary"] = -0.5
        if len(self.collision_hist) > 0:
            r -= 10.0
            info["penalty_collision"] = -10.0
        if self.offroad_steps > 40:
            r -= 10.0
            info["penalty_stuck"] = -10.0
        
        if random.random() < 0.05:
             print(f"📊 R:{r:.2f} | V_Lat:{lat_vel:.1f} | Lane:{lateral_dist:.2f}m | S:{speed:.1f}", flush=True)
             
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
