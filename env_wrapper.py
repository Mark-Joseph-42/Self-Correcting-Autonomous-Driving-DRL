# Monkey patch simplepbr to fix MetaDrive 0.3.0.1 compatibility with simplepbr 0.11.2
# The user-requested version of simplepbr moved _load_shader_str to _shaderutils
import simplepbr
import simplepbr._shaderutils
if not hasattr(simplepbr, '_load_shader_str'):
    simplepbr._load_shader_str = simplepbr._shaderutils._load_shader_str
if not hasattr(simplepbr, '_add_shader_defines'):
    simplepbr._add_shader_defines = simplepbr._shaderutils._add_shader_defines

# Monkey patch gltf for MetaDrive 0.3.0.1 compatibility with panda3d-gltf 1.0+
import gltf
if not hasattr(gltf, 'patch_loader'):
    def patch_loader(loader):
        # Manually register the loader if needed, or assume auto-registration
        # For now, just a dummy to prevent AttributeError
        print("DEBUG: Applied gltf.patch_loader mock")
        pass
    gltf.patch_loader = patch_loader

# Monkey patch metadrive.engine.core.our_pbr.OurPipeline to handle None arguments
# simplepbr 0.11+ crashes if window=None is passed, but metadrive passes None by default.
try:
    import metadrive.engine.core.our_pbr as our_pbr_module
    import builtins
    from simplepbr import Pipeline
    
    def fixed_init(self, render_node=None, window=None, camera_node=None, taskmgr=None, msaa_samples=4, **kwargs):
        # Fill in defaults if None, matching simplepbr behavior
        if hasattr(builtins, 'base'):
            base = builtins.base
            if render_node is None: render_node = base.render
            if window is None: window = base.win
            if camera_node is None: camera_node = base.cam
            if taskmgr is None: taskmgr = base.task_mgr
        
        # Force MSAA to 4 for macOS compatibility (16 fails on many Arm64 GPUs)
        msaa_samples = 4
        
        # Call Pipeline.__init__ directly (skipping broken OurPipeline.__init__)
        Pipeline.__init__(self,
            render_node=render_node,
            window=window,
            camera_node=camera_node,
            taskmgr=taskmgr,
            msaa_samples=msaa_samples,
            **kwargs
        )

    # Patch the method in place
    our_pbr_module.OurPipeline.__init__ = fixed_init
    
    # Patch 'manager' property alias for simplepbr 0.11 compatibility
    # simplepbr renamed 'manager' (or equivalent) to '_filtermgr', but OurPipeline uses 'manager'
    setattr(our_pbr_module.OurPipeline, 'manager', property(lambda self: self._filtermgr))
    
    print("DEBUG: Applied OurPipeline.__init__ and manager alias patch for simplepbr compatibility")
except Exception as e:
    print(f"DEBUG: Failed to patch OurPipeline: {e}")

import gym
from gym import spaces
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

class SensorFusionEnv(MetaDriveEnv):
    def default_config(self):
        config = super(SensorFusionEnv, self).default_config()
        config.update({
            "map": "SCX",  # Tough map: S-Curve, Intersection, etc.
            "traffic_density": 0.2, 
            "use_render": False,
            "crash_vehicle_penalty": 40.0,
            "crash_object_penalty": 40.0, 
            "out_of_road_penalty": 20.0,
            "success_reward": 50.0,
            "use_lateral_reward": True,
            "yellow_line_penalty": 20.0,
            "vehicle_config": {
                "image_source": "semantic_camera",
                "show_lidar": False,
                # Fast-Track: 64x64 Semantic-only
                "rgb_camera": (64, 64),
                "depth_camera": (64, 64),
                "semantic_camera": (64, 64),
                "lidar": {"num_lasers": 0, "distance": 0, "num_others": 0}
            },
            "image_observation": False,
        })
        return config
        
    def __init__(self, config=None):
        super(SensorFusionEnv, self).__init__(config)
        
        self.vec_space = super().observation_space
        # Simplified 64x64 Space
        self.semantics_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
        
        # Turbo Mode: Semantic + Vector (ignore RGB/Depth for fast learning)
        self._custom_observation_space = spaces.Dict({
            "semantic": self.semantics_space,
            "vector": self.vec_space
        })

    @property
    def observation_space(self):
        if hasattr(self, '_custom_observation_space'):
            return self._custom_observation_space
        return super().observation_space

    def reset(self, *args, **kwargs):
        r = super(SensorFusionEnv, self).reset(*args, **kwargs)
        info = {}
        if isinstance(r, tuple):
            vec_obs, info = r if len(r) == 2 else (r[0], {})
        else:
            vec_obs = r

        obs = self._get_onboard_observations(vec_obs)
        
        if isinstance(r, tuple):
            return obs, info
        return obs
        
    def step(self, action):
        r = super(SensorFusionEnv, self).step(action)
        # obs, reward, done, info (gym 0.21)
        vec_obs, reward, done, info = r[0], r[1], r[2], r[3] if len(r) > 3 else {}
            
        # Add granular reward breakdown to info for TransparencyCallback
        # This allows us to see *why* the agent got this reward.
        if self.vehicle:
            # Note: These metrics are calculated by MetaDrive, we just expose them clearly
            info["reward_speed"] = info.get("velocity_reward", 0.0)
            info["reward_lateral"] = info.get("steering_reward", 0.0) # meta-drive name
            info["penalty_collision"] = -self.config["crash_vehicle_penalty"] if info.get("crash_vehicle", False) else 0.0
            info["penalty_object"] = -self.config["crash_object_penalty"] if info.get("crash_object", False) else 0.0
            info["penalty_offroad"] = -self.config["out_of_road_penalty"] if info.get("out_of_road", False) else 0.0
            info["reward_success"] = self.config["success_reward"] if info.get("arrive_dest", False) else 0.0
            
            # Distance reward (progress)
            info["reward_route"] = info.get("route_completion_reward", 0.0)

            # Yellow Line Failure Condition (Middle Strip)
            if getattr(self.vehicle, "on_yellow_continuous_line", False):
                done = True
                penalty = self.config.get("yellow_line_penalty", 20.0)
                reward -= penalty
                info["penalty_yellow_line"] = -penalty
                info["on_yellow_line"] = True
            else:
                info["penalty_yellow_line"] = 0.0

        obs = self._get_onboard_observations(vec_obs)
        return obs, reward, done, info

    def _get_onboard_observations(self, vec_obs):
        """
        Retrieves all onboard sensor data and packages it into a dict.
        """
        images = self._get_sensor_images()
        return {
            "semantic": images["semantic"],
            "vector": vec_obs
        }

    def _get_sensor_images(self):
        """
        Retrieves Semantic images from the vehicle sensors (Turbo Mode).
        """
        obs = {"semantic": np.zeros((64, 64, 1), dtype=np.uint8)}
        
        if not (self.vehicle and self.vehicle.image_sensors):
            return obs

        sensor_name = "semantic_camera"
        try:
            if sensor_name in self.vehicle.image_sensors:
                img = self.vehicle.image_sensors[sensor_name].perceive(self.vehicle, clip=True)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                obs["semantic"] = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        except Exception:
            pass
                
        return obs

# Removed setup_engine to avoid invalid imports and because it was empty.
            
# Helper function for SB3
def make_env(render=False, map_type="SCX"):
    config = dict(
        use_render=render,
        map=map_type,
        vehicle_config=dict(
            semantic_camera=(64, 64),
        )
    )
    env = SensorFusionEnv(config)
    
    try:
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        env = GymV21CompatibilityV0(env=env)
    except ImportError as e:
        print(f"Shimmy wrapper failed: {e}, proceeding without wrapper")
        
    return env

