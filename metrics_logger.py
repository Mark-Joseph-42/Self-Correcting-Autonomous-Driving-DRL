from stable_baselines3.common.callbacks import BaseCallback
import sys
import numpy as np

class TerminalDashboard(BaseCallback):
    """
    Live terminal dashboard for immediate feedback.
    Prints a single line every step to show the agent is alive and acting.
    """
    def __init__(self, verbose=0):
        super(TerminalDashboard, self).__init__(verbose)
        self.step_count = 0
        print("✅ Terminal Dashboard Active", flush=True)
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Get latest info
        infos = self.locals.get("infos", [{}])[0]
        speed = infos.get("speed_kmh", 0.0) # Need to ensure env provides this
        reward = self.locals.get("rewards", [0.0])[0]
        
        # Spinner for liveness
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self.step_count % 10]
        
        # Overwrite line
        sys.stdout.write(f"\r{spinner} Step {self.step_count} | Speed: {speed:5.1f} km/h | Rew: {reward:6.2f} | Action: {self.locals.get('actions', [[0.,0.]])[0]}")
        sys.stdout.flush()
        return True

class TransparencyCallback(BaseCallback):
    """
    Consolidated, single-line logger for cleaner training output.
    """
    def __init__(self, verbose=1):
        super(TransparencyCallback, self).__init__(verbose)
        self.stats = self._get_empty_stats()
        self.episode_count = 0

    @staticmethod
    def _get_empty_stats():
        return {
            "speed": [], "lateral": [], "route": [],
            "collision": [], "offroad": [], "yellow_line": [], "success": [],
            "interventions": [], "lane_stability_time": None, "steps": 0
        }

    def _on_step(self) -> bool:
        self.stats["steps"] += 1
        infos = self.locals.get("infos", [{}])[0]
        self.stats["speed"].append(infos.get("reward_speed", 0.0))
        self.stats["lateral"].append(infos.get("reward_lateral", 0.0))
        self.stats["route"].append(infos.get("reward_route", 0.0))
        self.stats["collision"].append(infos.get("penalty_collision", 0.0))
        self.stats["offroad"].append(infos.get("penalty_offroad", 0.0))
        self.stats["yellow_line"].append(infos.get("penalty_yellow_line", 0.0))
        self.stats["success"].append(infos.get("reward_success", 0.0))
        
        # Phase 3 Metrics
        interventions = infos.get("total_interventions", 0)
        self.stats["interventions"].append(interventions)
        
        # Track Lane Stability (e.g. 500 steps without significant lateral penalty)
        if self.stats["lane_stability_time"] is None:
            if abs(infos.get("reward_lateral", 0.0)) < 0.1 and self.stats["steps"] > 500:
                self.stats["lane_stability_time"] = self.stats["steps"]
                print(f"✨ Lane Stability Achieved at step {self.stats['steps']}!")
            
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            print_episode_summary(self.stats, count=self.episode_count)
            self.stats = self._get_empty_stats()
        return True

def print_episode_summary(stats, title="EPISODE", count=None):
    summary = {k: (sum(v) if isinstance(v, list) else v) for k, v in stats.items()}
    # Filter stats to only include types we can sum for total reward
    r_keys = ["speed", "lateral", "route", "collision", "offroad", "yellow_line", "success"]
    total = sum([summary[k] for k in r_keys if k in summary])
    
    # Concise Single-Line Format
    count_str = f"#{count}" if count else ""
    line = (f"🚗 {title} {count_str:4} | Reward: {total:7.2f} | "
            f"Dist: {summary['route']:5.1f} | Spd: {summary['speed']:5.1f} | "
            f"Align: {summary['lateral']:5.1f} | Safety: {summary['collision']+summary['offroad']+summary['yellow_line']:5.1f}")
    
    print(line)
    return summary

def save_telemetry_snapshot(stage_num, stats, output_dir):
    """
    Saves a summary of the stage performance to a text file.
    """
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"telemetry_stage_{stage_num}.txt")
    
    summary = {k: (sum(v) if isinstance(v, list) else v) for k, v in stats.items()}
    r_keys = ["speed", "lateral", "route", "collision", "offroad", "yellow_line", "success"]
    total = sum([summary[k] for k in r_keys if k in summary])
    
    # Calculate a mock success rate if episodes were tracked
    # In a real scenario, we'd track specific success flags
    success_rate = (summary.get('success', 0) / max(1, len(stats.get('success', [])))) * 100
    
    content = f"""
=========================================
STAGE {stage_num} COMPLETION SNAPSHOT
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=========================================
Total Reward Scored: {total:.2f}
Distance Progress:   {summary['route']:.1f}
Speed Reward:        {summary['speed']:.1f}
Alignment Reward:    {summary['lateral']:.1f}
Safety Penalties:    {summary['collision'] + summary['offroad'] + summary['yellow_line']:.1f}
-----------------------------------------
FINAL SUCCESS RATE:  {success_rate:.1f}%
=========================================
"""
    with open(filepath, "w") as f:
        f.write(content)
    print(f"📊 Telemetry snapshot saved to {filepath}")
