# Self-Correcting Autonomous Driving with Deep Reinforcement Learning

**A Robust Curriculum Learning Approach for Safe and Sample-Efficient Driving**

This project demonstrates a high-performance Autonomous Driving agent trained using **Proximal Policy Optimization (PPO)** and **Curriculum Learning (CL)** in the **CARLA 0.9.13** simulator. By progressing through a 5-stage curriculum—from empty roads to a chaotic "Gauntlet" with traffic and storms—the agent learns to drive safely and efficiently.

---

## 🚀 Key Features

*   **5-Stage Curriculum**: Automatically graduates the agent from simple lane keeping to complex collision avoidance.
*   **Vector-Based Perception**: Replaces slow image processing with fast, raycast-based obstacle detection (LIDAR logic) and direct telemetry.
*   **Optimized Performance**: Runs at **~240 FPS** in headless mode (vs. 20 FPS standard) using `-RenderOffScreen` and disabled spectator rendering.
*   **Safety-First RL**: Implements **Action Shaping** (Throttle Boost) and **Dense Rewards** (Speed + Safety) to overcome static friction and fear of movement, while strictly penalizing collisions (-50.0).
*   **Robust Architecture**: Custom `CarlaSyncManager` with asynchronous cleanup handles the notoriously unstable CARLA server lifecycle, preventing segmentation faults.

---

## 🏗️ System Architecture

The system uses a flat vector observation space for maximum sample efficiency:

```
┌───────────────────────────────────────────────────────────────┐
│                     CARLA 0.9.13 Server                       │
│   (Town01-05 | Traffic Manager | Headless Mode | Async Kill)  │
└───────────────────────────────────────────────────────────────┘
                             ▲
                             │ API (Python 3.7)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                       CarlaEnv Wrapper                        │
│  ┌─────────────────────┐     ┌───────────────────────────┐   │
│  │  CarlaSyncManager   │────▶│   Sensor Suite            │   │
│  │  (Force Async Kill) │     │  - Obstacle Raycasts (32) │   │
│  └─────────────────────┘     │  - Collision Sensor       │   │
│                              │  - Traffic Light Telemetry│   │
│                              └───────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│               Gym Observation Space (44-Dim Vector)          │
│               - [0-2]:   Speed, Steer, Throttle              │
│               - [10-11]: Traffic Light State & Distance      │
│               - [12-43]: 32-Sector Obstacle Distances        │
└───────────────────────────────────────────────────────────────┘
                             ▲
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                      Training Pipeline                        │
│  ┌─────────────────────┐     ┌───────────────────────────┐   │
│  │  run_curriculum.sh  │────▶│   PPO Agent (SB3)         │   │
│  │  (Auto-Restart)     │     │  - MlpPolicy (Fast)       │   │
│  └─────────────────────┘     │  - Action Shaping         │   │
│                              └───────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## 📚 The 5-Stage Curriculum

| Stage | Name                | Map    | Traffic | Weather   | Goal (Reward) | Description |
|-------|---------------------|--------|---------|-----------|---------------|-------------|
| **1** | Empty Roads         | Town01 | 0%      | Clear     | 200.0         | Learn lane keeping & accelerator control. |
| **2** | Static Obstacles    | Town01 | 0%      | Clear     | 300.0         | Avoid walls/poles. Learn braking. |
| **3** | Dynamic Traffic     | Town03 | 20%     | Clear     | 400.0         | Interact with moving vehicles. |
| **4** | Weather Variations  | Town03 | 20%     | Dynamic   | 500.0         | Adapt to rain/fog/wet roads. |
| **5** | **The Gauntlet**    | Town05 | 40%     | Storm     | 600.0         | Assessing survival in high-density chaos. |

---

## ⚙️ Usage

### 1. Training (Headless & Robust)
The training pipeline is fully automated by `run_curriculum.sh`. It handles server restarts, model chaining, and error recovery.

```bash
# Start from Stage 1 (Fresh Start)
./run_curriculum.sh

# Resume from Stage 3
./run_curriculum.sh 3
```

### 2. Visualization (Watch it Drive)
To verify the agent's behavior with a live camera feed:

```bash
# 1. Start the visualizer server (in a separate terminal)
./launch_carla_viz.sh

# 2. Run the test script
# Replace with your specific model path
export USE_CARLA=1
conda run -n carla_py37 python test.py --model outputs/stage_1/checkpoints/stage1_model_50000_steps.zip --stage 1
```

---

## 🔧 Installation

**Requirements:**
- Linux (Ubuntu 20.04+)
- CARLA 0.9.13
- NVIDIA GPU (CUDA 11+)
- Python 3.7

**Setup:**
```bash
# Create Environment
conda create -n carla_py37 python=3.7 -y
conda activate carla_py37

# Install Dependencies
pip install numpy==1.21.6 stable-baselines3==1.8.0 gym==0.21.0 opencv-python pillow torch

# Fix missing libomp (if needed)
conda install -c conda-forge llvm-openmp -p ./carla_deps -y
```

---

## 🧠 Technical Innovations

1.  **Throttle Boosting**: We implemented an **Action Shaping** wrapper that maps the agent's `[0, 1]` throttle output to `[0.3, 1.0]`. This ensures the vehicle physically overcomes CARLA's static friction model, preventing the "stuck agent" problem.
2.  **Dense Rewards**: The agent receives a continuous reward `speed / 100.0`. This provides an immediate gradient for learning, acting as a "trail of breadcrumbs" out of the stationary zero-reward state.
3.  **Async Cleanup**: To fix "Signal 11" Segfaults, the environment forces the server into Asynchronous Mode before destroying actors. This prevents race conditions where the server tries to "tick" a destroyed vehicle.
4.  **Raycast Perception**: Instead of processing heavy 128x128 images, the agent uses a 32-ray obstacle sensor (simulated LIDAR). This reduces input dimensionality by **99.9%**, allowing for extremely fast inference and training.

---

## 📊 Deliverables structure

All training artifacts are automatically saved to `./outputs/`:
- `outputs/stage_X/checkpoints/`: Model weights (saved every 5k steps).
- `outputs/stage_X/tensorboard/`: Training metrics (Reward, Loss, FPS).
- `outputs/transition_log.txt`: Records of curriculum graduation.

---

#### License
Academic/Research Use Only.
**A Curriculum Learning Approach for Safe and Sample-Efficient Driving Policy Formation**

This project demonstrates the superiority of **Curriculum Learning (CL)** over traditional Deep Reinforcement Learning (DRL) for training autonomous driving agents. Using the high-fidelity **CARLA 0.9.13** simulator and a 5-stage curriculum, the agent learns to navigate from empty roads to complex "Gauntlet" scenarios with dynamic traffic and adverse weather.

---

## 🎯 Research Goal

To prove that a curriculum-trained agent achieves:
1.  **Higher Sample Efficiency**: Reaches peak performance with fewer training steps.
2.  **Improved Safety**: Exhibits significantly fewer collisions during the final "Gauntlet" test.

---

## 🏗️ System Architecture

The system consists of three core layers:

```
┌───────────────────────────────────────────────────────────────┐
│                     CARLA 0.9.13 Server                       │
│   (Town01-05 | Traffic Manager | Weather | Headless/RTX4000)  │
└───────────────────────────────────────────────────────────────┘
                             ▲
                             │ API (Python 3.7)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                       CarlaEnv Wrapper                        │
│  ┌─────────────────────┐     ┌───────────────────────────┐   │
│  │  CarlaSyncManager   │────▶│   Sensor Suite            │   │
│  │  (Synchronous Mode) │     │  - Semantic Segmentation  │   │
│  └─────────────────────┘     │  - Collision Sensor       │   │
│                              │  - Lane Invasion Sensor   │   │
│                              └───────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│               Gym Observation Space (Dict)                   │
│               - 'semantic_segmentation': (64, 64, 1)         │
│               - 'vector': (10,)                              │
└───────────────────────────────────────────────────────────────┘
                             ▲
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                      Training Pipeline                        │
│  ┌─────────────────────┐     ┌───────────────────────────┐   │
│  │  CurriculumManager  │────▶│   PPO Agent (SB3)         │   │
│  │  (5-Stage Logic)    │     │  - MultiInputPolicy       │   │
│  └─────────────────────┘     │  - RTX 4000 (CUDA)        │   │
│                              └───────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## 📚 The 5-Stage Curriculum

The agent progresses through increasingly difficult environments:

| Stage | Name                | Map    | Traffic | Weather   | Goal (Reward) |
|-------|---------------------|--------|---------|-----------|---------------|
| 1     | Empty Roads         | Town01 | 0%      | Clear     | 30.0          |
| 2     | Static Obstacles    | Town01 | 0%      | Clear     | 40.0          |
| 3     | Dynamic Traffic     | Town03 | 20%     | Clear     | 50.0          |
| 4     | Weather Variations  | Town03 | 20%     | Dynamic   | 60.0          |
| 5     | **The Gauntlet**    | Town05 | 40%     | Dynamic   | 70.0          |

---

## 📦 Project Structure

```
Self-Correcting-Autonomous-Driving-DRL/
├── carla_env.py          # CARLA Gymnasium environment wrapper
├── curriculum_manager.py # 5-stage config and auto-graduation callback
├── agent_logic.py        # PPO agent factory (Stable-Baselines3)
├── train.py              # Main training orchestrator
├── train_baseline.py     # Non-curriculum baseline training
├── compare_agents.py     # Research comparison (Curriculum vs Baseline)
├── test.py               # Inference / model evaluation
├── metrics_logger.py     # Episode logging and telemetry snapshots
├── run_smoke_test.py     # 50% Submission integration test
├── record_success_video.py # Video recording for deliverables
├── launch_carla.sh       # Headless CARLA server launcher
├── launch_carla_viz.sh   # Visualized CARLA server launcher
└── outputs/              # All auto-generated deliverables
    ├── stage_1/          # Model weights, TB logs, telemetry
    ├── stage_2/
    ├── ...
    ├── baseline/         # Baseline model outputs
    └── integration/      # Smoke test log
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **CARLA 0.9.13**: Installed at `/home/tinkerspace/carla project/`
- **NVIDIA Driver**: Compatible with CUDA 13.0 (RTX 4000 SFF)
- **Conda**: For Python environment management

### 2. Create the Environment
A dedicated Python 3.7 environment is required for the CARLA 0.9.13 API.

```bash
conda create -n carla_py37 python=3.7 -y
conda activate carla_py37

# Install dependencies
pip install numpy==1.21.6 stable-baselines3==1.8.0 gym==0.21.0 opencv-python pillow torch
```

### 3. Resolve `libomp.so.5` (Without Sudo)
If the CARLA server fails with a `libomp.so.5` error:
```bash
conda install -c conda-forge llvm-openmp -p ./carla_deps -y
# The launch script is already configured to add this to LD_LIBRARY_PATH
```

---

## 🚀 Usage

### Phase 1: Start the CARLA Server
```bash
# For headless training (recommended)
./launch_carla.sh

# For visualization/debugging
./launch_carla_viz.sh
```

### Phase 2: Run Curriculum Training (Optimized)
**NEW (2026):** Use the robust curriculum runner for maximum stability and speed (210+ FPS).

```bash
# Run full curriculum (Stages 1-5) with auto-restart and crash recovery:
./run_curriculum.sh

# Resume from a specific stage (e.g., Stage 3) if interrupted:
./run_curriculum.sh 3
```

This script automatically:
- Restarts the CARLA server between stages to prevent map-switch crashes.
- Cleans up stale checkpoints (only if starting fresh).
- Handles harmless exit crashes (Code 139).

*(Legacy Method)*
```bash
export USE_CARLA=1
conda run -n carla_py37 --no-capture-output python train_headless.py
```

### Phase 3: Run Baseline Training (for Research Comparison)
```bash
export USE_CARLA=1
conda run -n carla_py37 --no-capture-output python train_baseline.py
```

### Phase 4: Generate Research Comparison
```bash
conda run -n carla_py37 python compare_agents.py
# Output: ./outputs/research/comparison_results.csv
```

### Inference / Testing a Trained Model
```bash
conda run -n carla_py37 python test.py --model "./outputs/stage_5/ppo_agent_stage_5.zip"
```

---

## 📊 Deliverables & Outputs

All outputs are structured for direct inclusion in your research submissions.

| Milestone | Deliverable                     | Location                                   |
|-----------|---------------------------------|--------------------------------------------|
| **50%**   | Smoke Test Log                  | `./outputs/integration/smoke_test.log`     |
| **50%**   | Architecture Diagram            | (See `architecture_diagram.md` artifact)   |
| **80%**   | Transition Log                  | `./outputs/transition_log.txt`             |
| **80%**   | TensorBoard Logs (per stage)    | `./outputs/stage_X/tensorboard/`           |
| **100%**  | Final Model Weights             | `./outputs/stage_5/ppo_agent_stage_5.zip`  |
| **100%**  | Baseline Model Weights          | `./outputs/baseline/ppo_baseline_final.zip`|
| **100%**  | Comparison CSV                  | `./outputs/research/comparison_results.csv`|

---

## 🧠 Key Technical Decisions

1.  **Semantic Segmentation over Instance Segmentation**: The Instance Segmentation sensor caused segmentation faults in CARLA 0.9.13's Synchronous Mode. Semantic Segmentation provides equivalent class-level perception with full stability.

2.  **Legacy `gym` over `gymnasium`**: Stable-Baselines3 1.8.0 (required for Python 3.7) does not support `gymnasium` Dict spaces. The environment uses the legacy `gym==0.21.0` API.

3.  **Synchronous Mode First**: The `CarlaSyncManager` applies world settings (Sync Mode) *before* spawning sensors. This prevents race conditions that lead to crashes.

4.  **Headless Rendering (`-RenderOffScreen`)**: Maximizes training throughput on the RTX 4000 by avoiding GPU draw calls for a display window.

---

## 📜 License

This project is for academic and research purposes.
