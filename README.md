# CARLA Autonomous Driving: Curriculum Learning vs Standard DRL

A research project comparing **Curriculum Learning (CL)** against **standard Deep Reinforcement Learning (DRL)** for training a self-driving agent in the [CARLA simulator](https://carla.org/). The goal is to show that structured task progression (CL) produces better driving agents than flat training (DRL) within the same step budget.

## Project Structure

```
├── carla_env.py      # Gymnasium environment wrapping CARLA 0.9.13
├── train.py          # SAC training script for CL and DRL agents
├── evaluate.py       # Post-training evaluation, plotting, and video recording
├── run-cl.py         # Visual demo script for the CL agent
├── run-drl.py        # Visual demo script for the DRL agent
├── run-cl            # Shell launcher for CL demo (starts CARLA + agent)
├── run-drl           # Shell launcher for DRL demo (starts CARLA + agent)
├── carla_0.9.13/     # CARLA simulator installation
├── venv/             # Python 3.8 virtual environment with dependencies
├── models/           # Saved model checkpoints (created during training)
├── logs/             # Training reward and verification CSVs
└── results/          # Generated plots and evaluation videos
```

## How It Works

### Environment (`carla_env.py`)

The CARLA environment provides:

- **State**: 13-dimensional waypoint-based vector (not images) containing:
  - Normalized speed, steering angle
  - Lateral offset from lane center, heading error
  - Angles to 5 upcoming waypoints (5m, 10m, 15m, 20m, 25m ahead)
  - Junction flag, traffic light state, obstacle distance, current acceleration

- **Actions**: 2D continuous `[steer, accel]` where positive accel = throttle, negative = brake

- **Reward**: Dense reward function:
  ```
  reward = speed_along_road - 2.0 * |lateral_offset| - 1.0 * |heading_error| - 0.1 * |steer_delta|
           - 50.0 * collision - 10.0 * off_road
  ```

- **3 Curriculum Stages**:
  | Stage | Map    | Description                    |
  |-------|--------|--------------------------------|
  | 1     | Town01 | Straight roads only            |
  | 2     | Town03 | Full map (curves + junctions)  |
  | 3     | Town03 | Full map + obstacle vehicles   |

### Training (`train.py`)

Uses **Soft Actor-Critic (SAC)** from Stable-Baselines3 with:
- `MlpPolicy` with `net_arch=[256, 256]`
- Linear learning rate schedule from 3e-4 → 0
- TF32 Tensor Core acceleration on NVIDIA GPUs
- 30,000 total training steps per agent

**Curriculum Learning agent**: Trains 10k steps per stage (1→2→3), building skills progressively.  
**Standard DRL agent**: Trains 30k steps directly on Stage 3 (full difficulty).

Two callbacks log data during training:
- `RewardLoggerCallback`: Logs episode reward + global step to CSV after every episode
- `VerificationCallback`: Runs 3 evaluation episodes every 2k steps, reports pass/fail gates

### Evaluation (`evaluate.py`)

Generates:
- **Reward Growth Comparison Graph** (`results/reward_growth_comparison.png`): Rolling average reward vs training steps, CL (green) vs DRL (red), with curriculum stage transition markers
- **Verification Metrics** (`results/verification_metrics.png`): Distance and collision rate over training steps
- **Demo Videos**: Records 3 episodes per agent with a 3rd-person chase camera

## Prerequisites

- CARLA 0.9.13 server (included at `/workspace/carla_0.9.13/`)
- Python 3.8 virtual environment with:
  - `gymnasium`, `stable-baselines3`, `torch` (CUDA), `numpy`, `pandas`, `matplotlib`, `opencv-python`

## Usage

### 1. Train Both Agents

```bash
# Start CARLA server (headless)
/workspace/carla_0.9.13/CarlaUE4.sh -RenderOffScreen &
sleep 15

# Activate environment and run training
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace:/workspace/carla_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
python train.py --mode both    # Trains CL then DRL sequentially
```

Training modes: `--mode cl` (CL only), `--mode drl` (DRL only), `--mode both` (sequential)

### 2. Generate Comparison Graphs

```bash
python evaluate.py
# → results/reward_growth_comparison.png
# → results/verification_metrics.png
```

### 3. Run Visual Demos

```bash
./run-cl    # Launches CARLA in visual mode + CL agent
./run-drl   # Launches CARLA in visual mode + DRL agent
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Waypoint vectors over images | Converges in ~10k steps vs 100k+ for CNN-based policies |
| 2D `[steer, accel]` actions | Avoids degenerate throttle+brake region of 3D action space |
| Dense reward with `speed_along_road` | Provides continuous gradient signal instead of sparse collision penalties |
| TF32 instead of FP16 AMP | FP16 causes NaN gradients in SAC's Normal distribution; TF32 is numerically stable |
| `batch_size=256` | Prevents gradient explosion that occurs with large batches (2048) |

## Output Files

| File | Description |
|------|-------------|
| `logs/cl_rewards.csv` | Per-episode: episode number, global step, total reward |
| `logs/drl_rewards.csv` | Same format for DRL baseline |
| `logs/cl_verification.csv` | Every 2k steps: avg distance, speed, lateral offset, collision rate |
| `logs/drl_verification.csv` | Same for DRL |
| `models/sac_cl_stage{1,2}_final.zip` | CL model checkpoints |
| `models/sac_drl_final.zip` | DRL final model |
| `results/reward_growth_comparison.png` | The main deliverable graph |
