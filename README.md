# Self-Correcting Autonomous Driving with Deep Reinforcement Learning

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
