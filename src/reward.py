import numpy as np

def calculate_speed_reward(v_kmh, target=30.0):
    """Linear growth to target, then penalty."""
    return max(0, 1.0 - (abs(v_kmh - target) / target))

def calculate_lane_reward(d_lat):
    """Gaussian centering reward (AI-Boosted)."""
    # sigma=1.5 means at 1.5m distance (lane edge), reward is ~0.6
    sigma = 1.5
    return np.exp(-(d_lat**2) / (2 * sigma**2))

def calculate_alignment_reward(ego_fwd, road_fwd):
    """Dot product of forward vectors."""
    dot = np.dot(ego_fwd, road_fwd)
    return max(0, dot)

def get_total_reward(info):
    """
    AI-Boosted Reward Function V12:
    - Target 60s Survival Mastery
    - Red/Yellow Signal Penalty
    - Junction Centering Boost
    """
    if info['speed'] < 1.0:
        return -10.0
    
    total = 0.0
    
    # 1. Survival Bonus (Time-based scaling)
    # Increases slightly over time to incentivize long-run stability
    total += 2.0 + (info.get('steps_count', 0) / 500.0)
    
    # 2. Gaussian Centering
    # Boost weight in junctions for precise turns
    center_weight = 6.0 if info.get('is_junction', False) else 3.0
    total += center_weight * calculate_lane_reward(info['d_lat'])
    
    # 3. Target Speed Reward
    speed_eff = 1.0 - abs(info['speed'] - 40.0) / 40.0
    total += 5.0 * max(0, speed_eff) * calculate_alignment_reward(info['ego_fwd'], info['road_fwd'])
    
    # 4. V12: Signal & Junction Penalties
    if 'Red' in info['tl_state'] or 'Yellow' in info['tl_state']:
        if info['speed'] > 2.0:
            total -= 50.0 # Heavy penalty for running light
            
    if info['collision']:
        total -= 100.0
    if info['off_road']:
        total -= 30.0
    if info['speed'] > 45.0:
        total -= 5.0
        
    return float(total)
