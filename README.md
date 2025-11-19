# UR10e PPO with Stable-Baselines3 - Pure RL Control

This implementation replaces the RL-PID hybrid approach with pure reinforcement learning control using Stable-Baselines3.

## Key Features

- **Pure RL Control**: Direct torque control without PID tuning
- **6D Action Space**: Incremental torque control for each joint (Δτ)
- **Isaac Gym Integration**: High-fidelity physics simulation
- **Stable-Baselines3**: Leverages robust RL framework
- **RRT* Global Planning**: Future integration for global path planning

## Architecture

### Control Scheme: 方案A（直接力矩控制 + 增量控制）
- **Action Space**: 6D continuous (Δτ₁, Δτ₂, ..., Δτ₆)
- **Control Method**: Incremental torque updates
- **Safety**: Built-in torque limits and collision detection

### Environment Design
- **Base**: `gym.Env` interface for Stable-Baselines3 compatibility
- **Physics**: Isaac Gym simulation for realistic dynamics
- **State**: 16-dimensional observation space
- **Reward**: Shaped for learning torque control

## Directory Structure

```
ppo_ur10e_stab3/
├── README.md                    # This file
├── __init__.py                  # Package initialization
├── ur10e_incremental_env.py     # Main environment class (gym.Env)
├── train.py                     # Stable-Baselines3 training script
├── config.yaml                  # Environment configuration
├── inference.py                 # Model inference and testing
├── requirements.txt             # Python dependencies
├── global_planner/              # RRT* integration (future)
│   ├── __init__.py
│   ├── rrt_star.py
│   └── planner_interface.py
└── utils/                       # Utilities
    ├── __init__.py
    ├── kinematics.py
    └── safety.py
```

## Installation

```bash
cd ppo_ur10e_stab3
pip install -r requirements.txt
```

## Quick Start

```bash
# Train the model
python train.py --config config.yaml

# Run inference
python inference.py --model path/to/model.zip
```

## Control Architecture

The pure RL approach replaces PID parameter tuning with direct torque control:

```
RL Agent → Δτ (6D) → τ_current + Δτ → Isaac Gym Physics → Next State
```

### Advantages over RL-PID:
1. **Direct Control**: No intermediate PID layer
2. **Faster Response**: Immediate torque application
3. **Simpler Architecture**: Fewer hyperparameters to tune
4. **Better Learning**: More direct action-reward relationship

## Future Integration with RRT*

The architecture supports integration with global planners:
- **RRT*** generates global trajectory waypoints
- **RL Agent** handles local torque control between waypoints
- **Hybrid Approach**: Combines global planning with local control

## Key Differences from ppo_ur10e_gym

| Feature | ppo_ur10e_gym (RL-PID) | ppo_ur10e_stab3 (Pure RL) |
|---------|------------------------|---------------------------|
| Control | PID parameter tuning | Direct torque control |
| Action Space | 3D (kp, kd, ki) | 6D (Δτ₁...τ₆) |
| Framework | Custom PPO | Stable-Baselines3 |
| Interface | Custom environment | gym.Env |
| Global Planning | None | RRT* integration planned |
| Learning Target | PID optimization | Torque control policy |