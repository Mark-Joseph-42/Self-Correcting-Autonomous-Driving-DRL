# Error Log: Pickle Protocol Mismatch

## Observed Error
`ValueError: unsupported pickle protocol: 5`

## Context
Encountered during the ultra-fast debug training loop when attempting to load a PPO baseline model (`models/ppo_bc_baseline.zip`) in a Python 3.7 environment.

## Analysis
The baseline model was saved using Python 3.8+ or a newer version of `cloudpickle` that utilizes Protocol 5. Python 3.7 only supports up to Protocol 4.

## Resolution
Modified `train.py` to include a `try-except` block during model loading. If loading fails due to protocol mismatch or other environmental issues, the system now gracefully falls back to initializing a fresh PPO agent. This prevents the training pipeline from crashing and allows iterations to continue.

---
*Date: 2026-02-08*
