import os

# Paths
BASE_DIR = "/workspace"
LOG_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [LOG_DIR, CHECKPOINT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# SAC Hyperparameters (Optimized for RTX 4080 SUPER)
SAC_PARAMS = {
    "learning_rate": 3e-4, # Stable standard
    "buffer_size": 100000,
    "batch_size": 512, 
    "learning_starts": 1000, 
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 2, 
    "ent_coef": "auto",
}

# BEV Parameters
BEV_RES = 64
BEV_RADIUS = 10.0

# Curriculum Stages (Focused for 20s demo)
CURRICULUM = [
    {
        "stage": 1,
        "name": "Steering Mastery",
        "town": "Town01",
        "weather": "ClearNoon",
        "traffic_pct": 0,
        "steps": 100000,
        "threshold": 1500.0 # Mastery reward threshold increased for longer runs
    }
]

# Baseline config (Hardest map, same budget)
BASELINE = {
    "town": "Town03",
    "weather": "HardRainNoon",
    "traffic_pct": 20,
    "steps": 100000
}
