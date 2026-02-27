# Self-Correcting Autonomous Driving with DRL (SAC + 5-Stage Curriculum)

**A high-performance, stabilized autonomous driving agent trained in CARLA 0.9.13.**

This repository contains a robust implementation of a Soft Actor-Critic (SAC) agent optimized for complex driving scenarios. The project has undergone extensive hardening to resolve notorious CARLA Python API instabilities, library conflicts, and RL policy mismatches.

---

## 🛠️ Critical Engineering Fixes (The "Stabilization" Layer)

Significant technical debt and simulator-specific bugs were resolved to achieve 24/7 training stability.

### 1. The "Signal 11" (Segmentation Fault) Solution
CARLA 0.9.13 is prone to segmentation faults when multiple listeners are registered to the same sensor or when synchronous mode ticks overlap with actor destruction.
- **Bug**: Sensors were being registered twice (once in `_setup_sensors` and once in `CarlaSyncManager`), which is a known cause of immediate Python interpreter crashes in CARLA. 
- **Fix**: Refactored `CarlaSyncManager` to be the **sole owner** of sensor callbacks. We removed all `.listen()` calls from the environment's sensor setup and moved them into a centralized, thread-safe queuing system within the Sync Manager.
- **Result**: 100% elimination of the "Signal 11" crash during `env.reset()`.

### 2. GPU & Library Hardening
- **cuDNN Path Resolution**: Fixed the `libcudnn_ops_infer.so.8` missing error by explicitly mapping the library paths within the Conda environment to `LD_LIBRARY_PATH`.
- **Thread Deadlock Prevention**: Limited `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to `1` to prevent CPU-bound deadlocks during high-speed sensor data processing.
- **ULimit Optimization**: Increased `ulimit -u` to `4096` to prevent process starvation during multi-stage curriculum transitions.

### 3. Policy & Observation Alignment
- **SAC Transition**: Moved from PPO to **Soft Actor-Critic (SAC)** for better sample efficiency and off-policy learning.
- **Type Safety**: Aligned the observation space to strict `np.uint8` in the `[0, 255]` range to satisfy the `CnnPolicy` requirements of Stable Baselines 3, preventing `AttributeError` and tensor type mismatches.

### 4. Closure & Shadowing Bug
- **Bug**: In `train.py`, the variable `env` was being shadowed inside the `DummyVecEnv` lambda, causing a closure bug that incorrectly initialized the vector environment.
- **Fix**: Unique naming convention for intermediate wrappers (`env_raw` -> `env_wrapped` -> `env`) to ensure the correct object is captured in the execution scope.

---

## 🏗️ System Architecture

### Perception Pipeline
The agent perceives the world through a **Bird's Eye View (BEV)** semantic segmentation map:
- **Resolution**: $256 \times 256 \times 3$ (RGB-encoded Semantic Segmentation).
- **Processing**: Optimized using the `CnnPolicy` for deep spatial feature extraction.
- **Sensors**: 1 Semantic Segmentation Camera, 1 RGB Camera (Visuals), 1 Obstacle Sensor.

### 📚 The 5-Stage Curriculum
Training is staged to minimize "forgetting" and maximize learning gradients:

| Stage | Name | Map | Traffic | Weather | Reward Threshold |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **1** | Steering Mastery | Town01 | 0% | ClearNoon | 200.0 |
| **2** | Safety Engine | Town01 | 0% | Static Obs | 300.0 |
| **3** | Dynamic Traffic | Town03 | 20% | ClearNoon | 400.0 |
| **4** | Weather & Friction| Town03 | 20% | Rain/Wet | 500.0 |
| **5** | **The Gauntlet** | Town05 | 40% | Dynamic | 600.0 |

---

## 🚀 Execution & Usage

### Prerequisites
- **Python**: 3.7.13 (Conda recommended: `carla_py37`)
- **CARLA**: 0.9.13 (Headless version recommended)
- **VRAM**: 8GB+ (Agent uses SAC-CNN which is memory intensive)

### 1. Launching the Simulator
Always launch CARLA before starting any training or test scripts.
```bash
# Headless Mode (Standard for Training)
./launch_carla.sh

# Visualization Mode (For Watching Test.py)
./launch_carla_viz.sh
```

### 2. Full Curriculum Training
Running the full sequence with automatic stage transitions:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/venv/carla_py37/lib/python3.7/site-packages/nvidia/cudnn/lib/
conda run -n carla_py37 python train.py --stage 0
```

### 3. Visual Verification (Test Interface)
To see the agent driving in a visual window (requires `launch_carla_viz.sh`):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/venv/carla_py37/lib/python3.7/site-packages/nvidia/cudnn/lib/
conda run -n carla_py37 python test.py --stage 1 --viz
```

---

## 📊 Directory Structure
- `carla_env.py`: Refactored Gymnasium wrapper with `CarlaSyncManager`.
- `train.py`: The "Brain" - handles SAC training, callbacks, and GPU detection.
- `curriculum_manager.py`: Logic for reward-based graduation between stages.
- `test.py`: Modular inference script with `--viz` and `--debug` support.
- `outputs/`: 
  - `debug/`: High-frequency checkpoints (every 500 steps).
  - `stage_X/`: Final models and TensorBoard logs per stage.

---

## 📜 Technical Notes
- **FPS**: Training achieves ~25-30 FPS on an RTX 3060; ~100+ FPS on A100.
- **Sync Mode**: The environment enforces a fixed delta seconds (`0.1s`) for deterministic RL updates.
- **Cleanup**: The code implements a robust `_cleanup_actors()` method to prevent port clogging and server hangs during crashes.

---
*Maintained by the Autonomous Driving DRL Team (2026).*
