from stable_baselines3.common.callbacks import BaseCallback

class RewardThresholdCallback(BaseCallback):
    """
    Stop training if mean reward reaches threshold. 
    This is used to trigger the transition between curriculum stages.
    """
    def __init__(self, threshold, stage_num, output_dir, transition_log_path="./outputs/transition_log.txt", verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.threshold = threshold
        self.stage_num = stage_num
        self.output_dir = output_dir
        self.transition_log_path = transition_log_path

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            # Calculate mean reward from the recent episodes
            mean_reward = sum([info['r'] for info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
            if mean_reward >= self.threshold:
                if self.verbose > 0:
                    print(f"\n[Curriculum] Threshold reached: {mean_reward:.2f} >= {self.threshold}")
                    print(f"Curriculum Graduation: Stage {self.stage_num} -> Stage {self.stage_num + 1}")
                
                # 1. Save Model Weights (Per-Stage Bundle)
                import os
                model_save_path = os.path.join(self.output_dir, f"ppo_agent_stage_{self.stage_num}.zip")
                self.model.save(model_save_path)
                print(f"💾 Model saved to {model_save_path}")

                # 2. Log Transition (Phase 3 Deliverable)
                from datetime import datetime
                os.makedirs(os.path.dirname(self.transition_log_path), exist_ok=True)
                with open(self.transition_log_path, "a") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage {self.stage_num} -> Stage {self.stage_num+1} | Reward: {mean_reward:.2f}\n")

                # 3. Telemetry Snapshot (Per-Stage Bundle)
                # Note: We assume the TransparencyCallback (or similar) or metrics logger handles real-time stats.
                # Since we stop training here, we can trigger a final telemetry export if we have access to stats.
                # In train.py, we can pass the metrics or just signal completion.
                # For Phase 1, we'll assume train.py handles the final call after learn() returns.

                # 4. RESET BUFFER to prevent skipping future stages (CRITICAL FIX)
                from collections import deque
                self.model.ep_info_buffer = deque(maxlen=self.model.ep_info_buffer.maxlen)
                print("♻️ Reward buffer cleared for next stage.")

                return False
        return True

def get_curriculum_config():
    """
    Legacy 2-stage curriculum for MetaDrive.
    """
    return [
        {"name": "Stage 1: Straight Roads", "map": "S", "threshold": 50.0},
        {"name": "Stage 2: Complex Scenarios", "map": "SCX", "threshold": 50.0}
    ]

def get_carla_curriculum_config():
    """
    New 5-stage curriculum for CARLA 0.9.13.
    """
    return [
        {
            "name": "Stage 1: Recovery Training",
            "map": "Town01",
            "traffic_density": 0.0,
            "spawn_offset_range": 1.5,
            "weather": "ClearNoon",
            "threshold": 200.0,
            "timesteps": 50000
        },
        {
            "name": "Stage 2: Safety Engine", 
            "map": "Town01",
            "traffic_density": 0.0,
            "static_obstacles": True,
            "spawn_offset_range": 0.5,
            "weather": "ClearNoon",
            "threshold": 300.0,
            "timesteps": 75000
        },
        {
            "name": "Stage 3: Dynamic Traffic",
            "map": "Town03",
            "traffic_density": 0.2,
            "weather": "ClearNoon",
            "threshold": 400.0,
            "timesteps": 100000
        },
        {
            "name": "Stage 4: Weather Variations",
            "map": "Town03",
            "traffic_density": 0.2,
            "weather": "dynamic",
            "threshold": 500.0,
            "timesteps": 100000
        },
        {
            "name": "Stage 5: Gauntlet",
            "map": "Town05",
            "traffic_density": 0.4,
            "weather": "dynamic",
            "threshold": 600.0,
            "timesteps": 150000
        }
    ]
