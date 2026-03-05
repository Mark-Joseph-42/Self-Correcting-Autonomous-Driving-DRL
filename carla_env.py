import glob
import os
import sys
import time
import random
import math
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

# Add CARLA Python API to path
import glob
egg_file = '/workspace/carla_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg'
if egg_file not in sys.path:
    sys.path.append(egg_file)

import carla

class CarlaEnv(gym.Env):
    """
    Custom CARLA Gym Environment for SAC training with Curriculum Learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, host='127.0.0.1', port=2000, stage=1, no_render=True, tm_port=8000, map_name=None):
        super(CarlaEnv, self).__init__()
        
        # Parameters
        self.host = host
        self.port = port
        self.stage = stage
        self.im_width = 84
        self.im_height = 84
        self.dt = 0.1  # 10 FPS for more stable physics simulation
        self.no_render = no_render
        self.tm_port = tm_port
        self.map_name = map_name
        
        # Action space: [steering, accel]
        # Steering: [-1, 1], Accel: [-1, 1] (positive is throttle, negative is brake)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), 
                                       dtype=np.float32)
        
        # Observation space: 13-dimensional state vector
        # [speed, steer, lat_offset, heading_err, wp1_dx, wp1_dy, wp2_dx, wp2_dy, wp3_dx, wp3_dy, wp4_dx, wp4_dy, wp5_dx, wp5_dy, is_jct, tl_state, obs_dist, accel]
        # Wait, the plan said 13, let's count: 
        # 1 (speed) + 1 (steer) + 1 (lat_offset) + 1 (heading_err) + 5 (wp_distances) + 1 (is_jct) + 1 (tl_state) + 1 (obs_dist) + 1 (accel) = 13
        # But wait, dx,dy pairs is 10 values. Plan said: "10 values compressed to 5 distances + 5 angles -> keep 5 distances normalized". Or just use relative angles to the 5 waypoints. Let's use 5 relative angles.
        # So: speed, steer, lat_offset, heading_err, wp1_angle, wp2_angle, wp3_angle, wp4_angle, wp5_angle, is_jct, tl_state, obs_dist, accel -> 13 dimensions
        self.observation_space = spaces.Box(low=-1.0, high=1.0, 
                                            shape=(13,), 
                                            dtype=np.float32)
        
        # CARLA setup
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(60.0)
        self.world = None
        self.map = None
        self.vehicle = None
        self.camera_bev = None
        self.collision_sensor = None
        self._obs_buffer = None
        self._collision_event = False
        self.actor_list = []
        
        # Episode tracking
        self.total_dist = 0.0
        self.last_location = None
        self.episode_steps = 0
        self.stuck_steps = 0
        
        # Initialize world and map
        self._init_world()
        
    def _init_world(self):
        target_map = self.map_name if self.map_name else ('Town01' if self.stage == 1 else 'Town03')
        try:
            self.world = self.client.get_world()
            current_map = self.world.get_map().name.split('/')[-1]
            if current_map != target_map:
                print(f"Loading map {target_map}...")
                self.world = self.client.load_world(target_map)
        except Exception as e:
            print(f"Error connecting to CARLA: {e}")
            sys.exit(1)
            
        self.map = self.world.get_map()
        
        # Set synchronous mode for World and Traffic Manager
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.substepping = True
        settings.max_substep_delta_seconds = 0.05 # Optimization: 10 sub-steps -> 2 sub-steps per 0.1s tick
        settings.max_substeps = 10
        settings.no_rendering_mode = self.no_render
        self.world.apply_settings(settings)
        
        # Defensive Traffic Manager initialization
        tm_success = False
        for i in range(5):
            try:
                tm = self.client.get_trafficmanager(self.tm_port + i)
                tm.set_synchronous_mode(True)
                self.tm_port = self.tm_port + i # Update to the successful port
                tm_success = True
                break
            except RuntimeError as e:
                print(f"TM bind error on port {self.tm_port + i}, retrying next port...")
                time.sleep(1)
        
        if not tm_success:
            print("CRITICAL: Failed to initialize Traffic Manager after 5 attempts.")
            sys.exit(1)
        
        # Set all traffic lights to green
        for tl in self.world.get_actors().filter('traffic.traffic_light'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(10000.0)

    def set_stage(self, stage):
        if self.stage != stage:
            self.stage = stage
            self._init_world()

    def _cleanup(self):
        if hasattr(self, 'client') and self.client and hasattr(self, 'actor_list') and self.actor_list:
            batch = [carla.command.DestroyActor(x) for x in self.actor_list if x is not None and x.is_alive]
            if batch:
                self.client.apply_batch(batch)
                if self.world:
                    self.world.tick()
        self.actor_list = []
        self.vehicle = None
        self.camera_bev = None
        self.collision_sensor = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self._collision_event = False
        self._obs_buffer = None
        self.total_dist = 0.0
        self.episode_steps = 0
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Select spawn point based on stage
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        
        spawn_success = False
        print("DEBUG: Trying to spawn ego vehicle...")
        for spawn_point in spawn_points:
            self.vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if self.vehicle is not None:
                spawn_success = True
                break
        
        if not spawn_success:
            raise RuntimeError("Failed to spawn ego vehicle after multiple attempts")
            
        self.actor_list.append(self.vehicle)
        self.last_location = self.vehicle.get_location()
        self.stuck_steps = 0
        print("DEBUG: Ticking world 10 times to settle...")
        # Wait for vehicle to settle
        for _ in range(10):
            self.world.tick()
        
        print("DEBUG: Agent spawned, returning initial state.")
        return self._get_state(), {}

    def _spawn_obstacles(self, spawn_point):
        # Spawn 3-5 stalled vehicles ahead of the ego
        bp_lib = self.world.get_blueprint_library()
        wp = self.map.get_waypoint(spawn_point.location)
        
        for i in range(random.randint(3, 5)):
            # Spawn 20m, 40m, 60m etc ahead
            dist = (i + 1) * 25.0
            target_wp = wp.next(dist)[0]
            
            # Slightly offset from center or center
            offset = random.uniform(-1.5, 1.5)
            transform = target_wp.transform
            side_vec = transform.get_right_vector()
            transform.location += carla.Location(x=side_vec.x * offset, y=side_vec.y * offset)
            
            obs_bp = random.choice(bp_lib.filter('vehicle.*'))
            obs = self.world.try_spawn_actor(obs_bp, transform)
            if obs:
                self.actor_list.append(obs)
                obs.set_simulate_physics(True)

    def _get_state(self):
        if not self.vehicle:
            return np.zeros(13, dtype=np.float32)

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        control = self.vehicle.get_control()
        
        # 1. Speed (km/h) normalized by 40
        speed = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
        norm_speed = np.clip(speed / 40.0, 0.0, 1.0)
        
        # 2. Steering [-1, 1]
        steer = control.steer
        
        # Waypoint info
        wp = self.map.get_waypoint(transform.location)
        
        # 3. Lateral offset (normalized by 3.0)
        wp_loc = wp.transform.location
        vec_to_wp = np.array([wp_loc.x - transform.location.x, wp_loc.y - transform.location.y])
        wp_right = wp.transform.get_right_vector()
        vec_right = np.array([wp_right.x, wp_right.y])
        lat_offset = np.dot(vec_to_wp, vec_right)
        norm_lat_offset = np.clip(lat_offset / 3.0, -1.0, 1.0)
        
        # 4. Heading error (normalized by pi)
        vehicle_yaw = math.radians(transform.rotation.yaw)
        wp_yaw = math.radians(wp.transform.rotation.yaw)
        heading_err = math.atan2(math.sin(vehicle_yaw - float(wp_yaw)), math.cos(vehicle_yaw - float(wp_yaw)))
        norm_heading_err = np.clip(heading_err / math.pi, -1.0, 1.0)
        
        # 5-9. Upcoming waypoints
        wp_angles = []
        current_wp = wp
        for i in range(5):
            dist = (i + 1) * 5.0 # Check 5, 10, 15, 20, 25 meters ahead
            next_wps = current_wp.next(dist)
            if next_wps:
                next_wp = next_wps[0]
                # Angle relative to ego vehicle heading
                vec_to_next = np.array([next_wp.transform.location.x - transform.location.x, 
                                        next_wp.transform.location.y - transform.location.y])
                vec_dir = np.array([math.cos(vehicle_yaw), math.sin(vehicle_yaw)])
                # angle between vec_dir and vec_to_next
                angle = math.atan2(vec_to_next[1]*vec_dir[0] - vec_to_next[0]*vec_dir[1], 
                                   vec_dir[0]*vec_to_next[0] + vec_dir[1]*vec_to_next[1])
                wp_angles.append(np.clip(angle / math.pi, -1.0, 1.0))
            else:
                wp_angles.append(0.0)
                
        # 10. Is junction
        is_jct = 1.0 if wp.is_junction else 0.0
        
        # 11. Traffic light state (simplified: mostly green in Town01 unless manually set)
        tl_state = 0.0
        
        # 12. Obstacle distance
        obs_dist = 1.0 # default no obstacle
        if self.stage == 3:
            # Simple distance check to all actors for speed
            min_dist = 50.0
            for actor in self.actor_list:
                if actor != self.vehicle and 'vehicle' in actor.type_id:
                    d = transform.location.distance(actor.get_location())
                    # Check if it's in front of us
                    if d < min_dist:
                        vec_to_actor = np.array([actor.get_location().x - transform.location.x, 
                                                 actor.get_location().y - transform.location.y])
                        vec_dir = np.array([math.cos(vehicle_yaw), math.sin(vehicle_yaw)])
                        dot = np.dot(vec_to_actor, vec_dir)
                        if dot > 0: # It is in front
                            min_dist = d
            obs_dist = np.clip(min_dist / 50.0, 0.0, 1.0)
            
        # 13. Current Accel control
        accel = control.throttle if control.throttle > 0 else -control.brake
        
        state = [norm_speed, steer, norm_lat_offset, norm_heading_err] + wp_angles + [is_jct, tl_state, obs_dist, accel]
        return np.array(state, dtype=np.float32)

    def _on_collision(self, event):
        self._collision_event = True

    def step(self, action):
        # Action: [steer, accel]
        steer = float(action[0])
        accel = float(action[1])
        
        if accel > 0:
            control = carla.VehicleControl(steer=steer, throttle=accel, brake=0.0)
        else:
            control = carla.VehicleControl(steer=steer, throttle=0.0, brake=abs(accel))
            
        self.vehicle.apply_control(control)
        self.world.tick()
        self.episode_steps += 1
        
        obs = self._get_state()
        
        # Extract info for reward calculations
        v = self.vehicle.get_velocity()
        speed = 3.6 * np.linalg.norm([v.x, v.y, v.z])
        loc = self.vehicle.get_location()
        
        dist_delta = loc.distance(self.last_location)
        if self.last_location and speed > 1.0: # Only accumulate if actually moving
            self.total_dist += dist_delta
        
        # Stuck detection
        if speed < 0.5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            
        self.last_location = loc
        
        wp = self.map.get_waypoint(loc)
        wp_transform = wp.transform
        wp_fwd = wp_transform.get_forward_vector()
        
        # Velocity vector
        velocity_vec = np.array([v.x, v.y, v.z])
        wp_fwd_vec = np.array([wp_fwd.x, wp_fwd.y, wp_fwd.z])
        # Speed projected along road direction (m/s)
        speed_along_road = np.dot(velocity_vec, wp_fwd_vec)
        
        # Lateral offset
        lat_offset = obs[2] * 3.0 # unnormalize
        heading_err = obs[3] * math.pi # unnormalize
        
        # Steer delta for smoothness
        steer_diff = abs(obs[1] - self.last_steer) if hasattr(self, 'last_steer') else 0.0
        self.last_steer = obs[1]
        
        # Dense Reward
        reward = (
            1.0 * speed_along_road        # go fast in right direction
            - 2.0 * abs(lat_offset)       # stay in center
            - 1.0 * abs(heading_err)      # align with road
            - 0.1 * steer_diff            # steer smoothly
        )
        
        # DEBUG EXPLOSION
        if abs(reward) > 1e10:
            print(f"DEBUG REWARD EXPLOSION: reward={reward}, speed_along_road={speed_along_road}, lat_offset={lat_offset}, heading_err={heading_err}, v={v}")
            reward = np.clip(reward, -100.0, 100.0)
        
        terminated = False
        truncated = False
        info = {"speed": speed, "dist_traveled": self.total_dist, "lat_offset": abs(lat_offset)}
        
        # Terminals
        if self._collision_event:
            reward -= 50.0
            terminated = True
            info["reason"] = "collision"
            
        if abs(lat_offset) > 2.0: # ~Lane width is 3.5m, so 1.75m is edge. 2.0m is off road
            reward -= 10.0
            terminated = True
            info["reason"] = "off-road"
            
        if self.episode_steps > 600: # 60 seconds at 10 FPS
            truncated = True
            info["reason"] = "timeout"
            
        if self.stuck_steps > 30: # Stuck for 3 seconds
            terminated = True
            info["reason"] = "stuck"
            
        return obs, reward, terminated, truncated, info


    def close(self):
        self._cleanup()
        # Restore settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)

    def __del__(self):
        self._cleanup()
