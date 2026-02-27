# Improving Road Boundary Awareness

To address the issue of the agent running off-road, we will enhance its perception and reward structure to clearly distinguish "road" from "non-road" and penalize deviations more effectively.

## [NEW] Boundary Awareness Enhancement

1.  **Refine Radar Logic**: Update `_get_obs` to also consider the absence of road pixels (value 255) as a hazard in the virtual radar sectors. If a sector has no road, it should reflect a "blocked" proximity.
2.  **Explicit Off-Road Penalty**: Modify `_compute_reward` to use non-projected waypoints for accurate lane type detection, applying a heavy penalty for sidewalks/grass.
3.  **Terminal Penalty**: Add a significant negative reward (-100) to the `step` function when an `EARLY RESET` occurs due to being stuck off-road or colliding.
4.  **Lane Departure Slope**: Increase the penalty for `lateral_dist > 1.75` (crossing the lane line) to provide a continuous gradient back to the center.

## Proposed Changes

### CarlaEnv Component

#### [MODIFY] [carla_env.py](file:///home/tinkerspace/carla project/Self-Correcting-Autonomous-Driving-DRL/carla_env.py)

- **`_get_obs`**: Update sector processing to check for road presence (semantic tag 255).
- **`_compute_reward`**: 
    - Use `get_waypoint(..., project_to_road=False)` for accurate lane type checks.
    - Add a gradient penalty for being over the 1.75m lane boundary.
- **`step`**: Add a large terminal penalty for resets to ensure the agent avoids them at all costs.

## Verification Plan

### Automated Tests
- Run `debug_test.py` to verify the new radar logic and reward penalties.
- Monitor `run_curriculum.sh` logs for a reduction in "EARLY RESET" frequency over time.

### Manual Verification
- Verify in the visualization (if possible) that the agent makes corrective steering adjustments when approaching the road edge.
