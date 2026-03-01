# Self-Correcting Autonomous Driving (SAC-Version)

This repository implements an autonomous driving policy using **Soft Actor-Critic (SAC)** and **AI-Boosted Curriculum Learning** in the CARLA 0.9.13 simulator.

## Key Features
- **AI-Boosted Curriculum**: A staged approach to training, starting with "Steering Mastery" (Stage 1) to ensure robust centering before adding complex obstacles.
- **V12 60s Survival Goal**: Specifically optimized to handle 1 minute of crash-free, rule-following driving, including traffic signals and junctions.
- **Scientific Reward Engineering**:
  - **Gaussian Centering**: Dense centering signal using `exp(-d_lat²/σ²)`.
  - **Signal Sensitivity**: Automatic sensing and penalty for Red/Yellow traffic lights.
  - **Junction Awareness**: Dynamically boosted rewards for precise lane alignment in intersections.
- **Telemetry-Driven Insights**: Detailed logging of throttle, steer, brake, and signal states for every episode.

## Project Structure
- `src/`: Core Python source code.
  - `carla_env.py`: Gym-wrapped CARLA environment with AI-assist features.
  - `reward.py`: Definitive AI-Boosted reward logic (V12).
  - `train.py`: Training orchestration for Curriculum and Baseline models.
  - `drive_live.py`: High-resolution demonstration capture script.
- `results/`: Training logs, telemetry CSVs, and video demonstrations.
- `checkpoints/`: Saved SAC models.
- `run_experiment.sh`: One-click training and evaluation pipeline.

## Quick Start
1. Ensure CARLA 0.9.13 is running.
2. Setup environment:
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH=$PYTHONPATH:.
   ```
3. Run the V12 experiment:
   ```bash
   bash run_experiment.sh
   ```

## Results
Detailed results and video demonstrations can be found in the [Walkthrough](./results/walkthrough.md).

**Curriculum vs Baseline Performance**:
- **Curriculum Median Survival**: ~21 seconds (Stage 1).
- **Baseline Median Survival**: ~2 seconds (worst-case scenario).
- **Goal (V12)**: 60 seconds.

---
*Developed with the goal of proving that structured learning stages (Curriculum) beat end-to-end standard DRL for high-stakes robotics tasks.*
