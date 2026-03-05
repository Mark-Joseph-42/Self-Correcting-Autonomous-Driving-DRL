# Self-Correcting Autonomous Driving (DRL vs. Curriculum Learning)

This repository contains a high-performance, stable pipeline for training autonomous driving agents in CARLA using Soft Actor-Critic (SAC). 

## 🏆 The "New Discovery" Optimization
This branch (`newdiscovery`) represents a massive breakthrough in training efficiency and stability. By moving away from over-engineered "custom" configurations and returning to **Stable-Baselines3 (SB3) battle-tested defaults**, we achieved:
- **Throughput**: **33-38 FPS** (Environment steps/sec) on an RTX 4080 SUPER.
- **Learning Intensity**: **1 Update per step** with `batch_size=256`. 
- **Stability**: Zero `NaN` gradient crashes.
- **Performance**: **+6,991 Peak Reward** in Town03.

### Technical Insights
- **Network Right-Sizing**: Reduced the policy/critic network from `[1024, 1024, 1024]` to **`[256, 256]`**. The 13-dimensional observation vector did not require a massive network; right-sizing increased FPS by 5x while improving convergence speed.
- **SB3 Default Hyperparameters**: Reverted to standard SAC settings (`lr=3e-4`, `tau=0.005`, `batch=256`). This eliminated numerical instability and enabled faster "Learning RPM".
- **TensorFlow32/CuDNN Optimization**: Enabled TF32 matmul and CuDNN benchmarks for maximum GPU utilization on Ada Lovelace architecture.

---

## 📊 Results: CL vs. Standard DRL
We compared two training methodologies over a 20,000-step budget:

| Metric | Curriculum Learning (CL) | Standard DRL (Baseline) |
|--------|--------------------------|-------------------------|
| **Peak Episode Reward** | **+6,991** 🚀 | **+4,327** |
| **Converged State** | Smooth lane keeping & turning | Oscillatory behavior at turns |
| **Steps to Positive Reward** | ~1,500 Steps | ~4,500 Steps |

### Curriculum Stages
1. **Stage 1 (Town01)**: Straight road mastery, focus on speed maintenance and center-line tracking.
2. **Stage 2 (Town03)**: Handling curves and junctions.
3. **Stage 3 (Full Map)**: Integration of obstacle avoidance and complex navigation.

---

## 🛠️ Usage & Demos

You can now run the **Best Models** obtained from this research immediately:

### Run Curriculum Learning Demo
```bash
./run-cl
```
*This launches CARLA on Port 3000 and runs the best Stage 2 CL model (`sac_cl_final.zip`).*

### Run Standard DRL Demo
```bash
./run-drl
```
*Runs the baseline SAC model trained without curriculum support.*

---

## 📈 Data & Analysis
All training data is provided in CSV format in the `logs/` directory:
- `logs/cl_rewards.csv`: Step-by-step reward telemetry for the CL run.
- `logs/drl_rewards.csv`: Raw reward data for the baseline run.

These logs include `episode`, `step`, and `reward` columns, suitable for Matplotlib or Seaborn analysis. The final comparison plot can be found in `results/reward_growth_comparison.png`.

## 📦 Requirements
- **CARLA 0.9.13** (Linux)
- **Stable-Baselines3** (`v2.1.0+`)
- **Gymnasium**
- **PyTorch** (`v2.4+` with CUDA 12.1 supports)

---
*Developed for research in Autonomous Driving optimization.*
