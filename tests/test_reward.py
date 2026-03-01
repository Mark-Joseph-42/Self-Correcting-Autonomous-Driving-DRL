import numpy as np
from src.reward import calculate_speed_reward, calculate_lane_reward, calculate_alignment_reward, get_total_reward

def test_speed_reward():
    print("Testing speed reward...")
    assert calculate_speed_reward(30.0, target=30.0) == 1.0
    assert calculate_speed_reward(0.0, target=30.0) == 0.0
    assert calculate_speed_reward(60.0, target=30.0) == 0.0

def test_lane_reward():
    print("Testing lane reward...")
    # Centered at speed
    r1 = calculate_lane_reward(0.0, 30.0)
    # Off center
    r2 = calculate_lane_reward(1.0, 30.0)
    assert r1 > r2
    # Off road
    r3 = calculate_lane_reward(2.0, 30.0)
    assert r3 == 0

def test_total_reward_collision():
    print("Testing total reward with collision...")
    info = {
        'speed': 30.0,
        'd_lat': 0.0,
        'ego_fwd': np.array([1.0, 0.0]),
        'road_fwd': np.array([1.0, 0.0]),
        'collision': True,
        'off_road': False,
        'steer_delta': 0.0
    }
    # Base reward should be ~1.6 (1.0 speed, 0.4 lane, 0.2 align)
    # Collision penalty is -10.0
    r = get_total_reward(info)
    assert r < -8.0

if __name__ == "__main__":
    test_speed_reward()
    test_lane_reward()
    test_total_reward_collision()
    print("All reward unit tests passed!")
