import os
import argparse
import pandas as pd
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from src.carla_env import CarlaEnv
from src.config import SAC_PARAMS, CURRICULUM, BASELINE, LOG_DIR, CHECKPOINT_DIR

class TelemetryCallback(BaseCallback):
    """Logs episode telemetry (throttle, brake, steer) to CSV."""
    def __init__(self, log_dir, stage, verbose=0):
        super(TelemetryCallback, self).__init__(verbose)
        self.log_path = os.path.join(log_dir, f"stage_{stage}_telemetry.csv")
        self.episode_data = []

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            info = self.locals["infos"][0]
            if self.locals["dones"][0]:
                self.episode_data.append({
                    "step": self.num_timesteps,
                    "reward": self.locals["rewards"][0],
                    "speed": info.get("speed", 0),
                    "throttle": info.get("throttle", 0),
                    "brake": info.get("brake", 0),
                    "steer": info.get("steer", 0),
                    "d_lat": info.get("d_lat", 0),
                    "tl_state": info.get("tl_state", "Unknown"),
                    "is_junction": info.get("is_junction", False)
                })
                # Auto-save every 10 episodes
                if len(self.episode_data) % 10 == 0:
                    pd.DataFrame(self.episode_data).to_csv(self.log_path, index=False)
        return True

class MasteryCallback(BaseCallback):
    """Checks if mean reward exceeds threshold to graduate stage."""
    def __init__(self, threshold, verbose=0):
        super(MasteryCallback, self).__init__(verbose)
        self.threshold = threshold
        self.graduated = False

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = pd.Series([ep['r'] for ep in self.model.ep_info_buffer]).mean()
            if mean_reward > self.threshold:
                print(f"Mastery reached! Mean reward {mean_reward:.2f} > {self.threshold}")
                self.graduated = True
                return False # Stop training this stage
        return True

def train_curriculum():
    print("Starting Curriculum Training...")
    model = None
    env = CarlaEnv()
    
    total_logs = []
    
    for stage in CURRICULUM:
        print(f"--- Stage {stage['stage']}: {stage['name']} ---")
        env.reset(town=stage['town'], weather=stage['weather'], traffic_pct=stage['traffic_pct'])
        mon_env = Monitor(env, os.path.join(LOG_DIR, f"stage_{stage['stage']}"))
        
        if model is None:
            model = SAC("CnnPolicy", mon_env, **SAC_PARAMS, verbose=1)
        else:
            model.set_env(mon_env)
            
        from stable_baselines3.common.callbacks import CallbackList
        mastery_cb = MasteryCallback(stage['threshold'])
        telemetry_cb = TelemetryCallback(LOG_DIR, stage['stage'])
        model.learn(total_timesteps=stage['steps'], callback=CallbackList([mastery_cb, telemetry_cb]), reset_num_timesteps=False)
        
        model.save(os.path.join(CHECKPOINT_DIR, f"curriculum_stage_{stage['stage']}"))
        
    env.close()

def train_baseline():
    print("Starting Baseline Training...")
    env = CarlaEnv(town=BASELINE['town'])
    env.reset(weather=BASELINE['weather'], traffic_pct=BASELINE['traffic_pct'])
    mon_env = Monitor(env, os.path.join(LOG_DIR, "baseline"))
    
    model = SAC("CnnPolicy", mon_env, **SAC_PARAMS, verbose=1)
    
    from stable_baselines3.common.callbacks import CallbackList
    telemetry_cb = TelemetryCallback(LOG_DIR, "baseline")
    model.learn(total_timesteps=BASELINE['steps'], callback=CallbackList([telemetry_cb]))
    
    model.save(os.path.join(CHECKPOINT_DIR, "baseline_final"))
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["curriculum", "baseline"], required=True)
    args = parser.parse_args()
    
    if args.mode == "curriculum":
        train_curriculum()
    else:
        train_baseline()
