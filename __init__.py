"""
UR10e PPO with Stable-Baselines3 - Pure RL Control

This package implements a pure reinforcement learning approach to UR10e robotic arm control
using Stable-Baselines3 and Isaac Gym physics simulation.

Key Features:
- Direct torque control (6D action space)
- Incremental control approach
- Isaac Gym integration
- Stable-Baselines3 compatibility
- Future RRT* global planning integration
"""

from .ur10e_incremental_env import UR10eIncrementalEnv

__version__ = "1.0.0"
__author__ = "RL Robotics Team"

__all__ = [
    "UR10eIncrementalEnv",
]