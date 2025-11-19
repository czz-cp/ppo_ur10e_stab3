"""
UR10e Trajectory Tracking Environment with Task-Space RRT* Integration

Advanced RL environment that combines Task-Space RRT* global planning with
local RL control for precise trajectory tracking. Replaces traditional PID
controllers with learned torque control.

Features:
- Task-Space RRT* planning in 3D Cartesian space
- 19D observation space with relative position and progress
- OI-style reward function for trajectory tracking
- Joint-specific action scaling and momentum inhibition
- Stable-Baselines3 compatibility
"""

# Isaac Gym imports MUST be before PyTorch imports
try:
    # Check if already imported to avoid Foundation object conflicts
    import sys
    if 'isaacgym.gymapi' in sys.modules:
        # Use existing imports
        gymapi = sys.modules['isaacgym.gymapi']
        # Import missing modules if needed
        if 'isaacgym.gymtorch' not in sys.modules:
            from isaacgym import gymtorch
        else:
            gymtorch = sys.modules['isaacgym.gymtorch']
        if 'isaacgym.gymutil' not in sys.modules:
            from isaacgym import gymutil
        else:
            gymutil = sys.modules['isaacgym.gymutil']
    else:
        from isaacgym import gymapi
        from isaacgym import gymtorch
        from isaacgym import gymutil
    from isaacgym.torch_utils import *
    print("✅ Isaac Gym imported successfully in ur10e_trajectory_env")
except (ImportError, KeyError) as e:
    print(f"❌ Failed to import Isaac Gym in ur10e_trajectory_env: {e}")
    # Don't sys.exit here, let the main script handle the error

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
from typing import Dict, Any, Tuple, Optional, List
import yaml
from collections import deque
import warnings

# Local imports
from ur10e_incremental_env import UR10eIncrementalEnv
from task_space_planner import TaskSpacePlannerInterface, TSPlanningRequest, TSWaypoint

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class UR10eTrajectoryEnv(UR10eIncrementalEnv):
    """
    UR10e Environment with Task-Space RRT* + RL trajectory tracking

    Extends the base UR10eIncrementalEnv with:
    - Task-Space global planning
    - 19D observation space with trajectory information
    - OI-style trajectory tracking rewards
    - Waypoint progression management
    """

    def __init__(self, config_path: str = "config.yaml", num_envs: int = 1, mode: str = "trajectory_tracking"):
        """
        Initialize the trajectory tracking environment

        Args:
            config_path: Path to configuration file
            num_envs: Number of parallel environments
            mode: "point_to_point" or "trajectory_tracking"
        """
        # Initialize base environment first
        super().__init__(config_path, num_envs)

        # Trajectory tracking configuration
        self.mode = mode
        self.trajectory_config = self.config.get('trajectory_tracking', {})
        self.task_space_config = self.config.get('task_space', {})
        self.ts_rrt_config = self.config.get('ts_rrt_star', {})

        # Trajectory tracking state
        self.current_ts_waypoints: List[TSWaypoint] = []
        self.current_waypoint_index: int = 0
        self.trajectory_completed: bool = False

        # Reward function parameters (MUST be set before planner initialization)
        self.waypoint_threshold = self.trajectory_config.get('waypoint_threshold', 0.05)
        self.waypoint_bonus = self.trajectory_config.get('waypoint_bonus', 5.0)
        self.smooth_coef = self.trajectory_config.get('smooth_coef', 0.1)
        self.use_deviation_penalty = self.trajectory_config.get('use_deviation_penalty', False)
        self.deviation_coef = self.trajectory_config.get('deviation_coef', 2.0)

        # Initialize Task-Space planner for trajectory tracking mode
        if self.mode == "trajectory_tracking":
            self._init_task_space_planner()

        # Override observation space for trajectory tracking (19D)
        if self.mode == "trajectory_tracking":
            self._define_trajectory_observation_space()

        print(f"✅ UR10eTrajectoryEnv initialized:")
        print(f"   🎯 Control Mode: {self.mode}")
        print(f"   🎬 Device: {self.device}")
        print(f"   🔧 Parallel envs: {num_envs}")
        print(f"   📊 Observation space: {self.observation_space}")
        print(f"   📐 Action space: {self.action_space}")
        print(f"   🛤️  Task-Space planner: {'✅' if self.mode == 'trajectory_tracking' else '❌'}")

    def _init_task_space_planner(self):
        """Initialize Task-Space RRT* planner"""
        # Extract workspace bounds from config
        workspace_bounds_list = []
        for axis in ['x', 'y', 'z']:
            bounds = self.task_space_config.get('workspace_bounds', {}).get(axis, [-0.5, 0.5])
            workspace_bounds_list.append(bounds)

        workspace_bounds = np.array(workspace_bounds_list)

        # Initialize planner interface
        self.ts_planner = TaskSpacePlannerInterface(
            workspace_bounds=workspace_bounds,
            waypoint_spacing=0.1,
            replanning_threshold=self.ts_rrt_config.get('replanning_threshold', 0.1),
            max_waypoints=self.ts_rrt_config.get('max_waypoints', 50)
        )

        print(f"   🗺️  Workspace bounds: {workspace_bounds}")
        print(f"   📏 Waypoint threshold: {self.waypoint_threshold}m")

    def _define_trajectory_observation_space(self):
        """
        Define 19D observation space for trajectory tracking:

        19D = [joint_pos(6) + joint_vel(6) + delta_to_waypoint(3) + progress(1) + tcp_pos(3)]

        Key insight: Use relative position (delta) instead of absolute positions
        for better alignment with reward function and generalization.
        """
        obs_dim = 19  # Trajectory tracking: joints + velocities + delta + progress + tcp_pos
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        print(f"🎯 Trajectory observation space: {obs_dim}D")
        print(f"   Structure: [joint_pos(6) + joint_vel(6) + delta_to_waypoint(3) + progress(1) + tcp_pos(3)]")

    def plan_trajectory(self, start_tcp: np.ndarray, goal_tcp: np.ndarray) -> bool:
        """
        Plan trajectory from start TCP to goal TCP using Task-Space RRT*

        Args:
            start_tcp: Starting TCP position [x, y, z]
            goal_tcp: Goal TCP position [x, y, z]

        Returns:
            True if planning successful, False otherwise
        """
        if self.mode != "trajectory_tracking":
            print("⚠️ Trajectory planning not available in point_to_point mode")
            return False

        print(f"🛤️  Planning trajectory from {start_tcp} to {goal_tcp}")

        # Create planning request
        request = TSPlanningRequest(
            start_tcp=start_tcp,
            target_tcp=goal_tcp,
            max_planning_time=5.0,
            tolerance=self.waypoint_threshold
        )

        # Plan trajectory
        result = self.ts_planner.plan_to_target(request)

        if result.success:
            self.current_ts_waypoints = result.waypoints
            self.current_waypoint_index = 0
            self.trajectory_completed = False

            print(f"✅ Trajectory planned: {len(self.current_ts_waypoints)} waypoints")
            return True
        else:
            print(f"❌ Trajectory planning failed: {result.error_message}")
            self.current_ts_waypoints = []
            self.current_waypoint_index = 0
            return False

    def set_waypoints(self, waypoints: List[TSWaypoint]):
        """
        Set waypoints directly (for testing or external planning)

        Args:
            waypoints: List of TSWaypoint objects
        """
        self.current_ts_waypoints = waypoints
        self.current_waypoint_index = 0
        self.trajectory_completed = False
        print(f"📍 Set {len(waypoints)} waypoints for trajectory tracking")

    def get_current_waypoint(self) -> Optional[TSWaypoint]:
        """Get current waypoint for trajectory tracking"""
        if not self.current_ts_waypoints or self.current_waypoint_index >= len(self.current_ts_waypoints):
            return None
        return self.current_ts_waypoints[self.current_waypoint_index]

    def get_observation(self) -> np.ndarray:
        """
        Get observation for trajectory tracking mode (19D)

        Returns:
            19D observation array: [joint_pos(6) + joint_vel(6) + delta_to_waypoint(3) + progress(1) + tcp_pos(3)]
        """
        obs_list = []

        for i in range(self.num_envs):
            # Joint positions and velocities (same as base environment)
            joint_pos = self.joint_positions[i].cpu().numpy()
            joint_vel = self.joint_velocities[i].cpu().numpy()

            if self.mode == "trajectory_tracking":
                # Get current waypoint and TCP position
                current_waypoint = self.get_current_waypoint()
                tcp_pos = self._forward_kinematics(self.joint_positions[i]).cpu().numpy()

                if current_waypoint is not None:
                    # 🎯 Relative position to current waypoint (3D)
                    # This is the core information for RL learning
                    delta_to_waypoint = current_waypoint.cartesian_position - tcp_pos

                    # 📈 Progress along trajectory (1D scalar)
                    # Normalized progress from 0 (start) to 1 (end)
                    progress = self.current_waypoint_index / max(1, len(self.current_ts_waypoints) - 1)
                else:
                    # No waypoints available - use zeros
                    delta_to_waypoint = np.zeros(3)
                    progress = 0.0

                # Combine for trajectory tracking observation
                obs = np.concatenate([
                    joint_pos,              # 6D: joint angles
                    joint_vel,              # 6D: joint velocities
                    delta_to_waypoint,      # 3D: relative position to waypoint (KEY!)
                    [progress],             # 1D: trajectory progress
                    tcp_pos                 # 3D: current TCP position
                ])
                assert len(obs) == 19, f"Expected 19D observation, got {len(obs)}D"
            else:
                # Point-to-point mode (use base environment observation)
                target_pos = self.target_positions[i].cpu().numpy() if hasattr(self, 'target_positions') else np.zeros(3)
                tcp_pos = self._forward_kinematics(self.joint_positions[i]).cpu().numpy()

                obs = np.concatenate([
                    joint_pos,      # 6D: joint angles
                    joint_vel,      # 6D: joint velocities
                    target_pos,     # 3D: target position
                    tcp_pos         # 3D: current TCP position
                ])
                assert len(obs) == 18, f"Expected 18D observation, got {len(obs)}D"

            obs_list.append(obs)

        # Return first environment's observation (for single env training)
        if self.num_envs == 1:
            return obs_list[0].astype(np.float32)
        else:
            return np.array(obs_list, dtype=np.float32)

    def _trajectory_reward(self, tcp_pos: torch.Tensor, action_tensor: torch.Tensor) -> Tuple[float, bool]:
        """
        OI-style trajectory tracking reward function

        Design philosophy:
        - Distance reward: Direct penalty for distance to current waypoint
        - Waypoint bonus: Reward for reaching waypoints (helps credit assignment)
        - Smoothness penalty: Penalize large action changes (prevents oscillation)
        - Deviation penalty (optional): Penalize deviation from planned path

        Args:
            tcp_pos: Current TCP position tensor
            action_tensor: Current action tensor

        Returns:
            Tuple of (reward, waypoint_reached)
        """
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            # No waypoints available - return default reward
            return -0.1, False

        # Convert to tensors for computation
        waypoint_pos = torch.tensor(current_waypoint.cartesian_position, device=self.device, dtype=torch.float32)

        # 1. 📏 Distance-based reward (core learning signal)
        distance = torch.norm(tcp_pos - waypoint_pos)
        r_distance = -distance.item()  # Linear penalty for distance

        # 2. 🎯 Waypoint arrival reward (helps credit assignment)
        reached = distance < self.waypoint_threshold
        r_waypoint = reached.float().item() * self.waypoint_bonus

        # 3. 🌊 Smoothness reward (prevents oscillation)
        action_norm = torch.norm(action_tensor)
        r_smooth = -self.smooth_coef * action_norm.item()

        # 4. 📐 Path deviation penalty (optional - more complex)
        r_deviation = 0.0
        if self.use_deviation_penalty and len(self.current_ts_waypoints) > 1:
            r_deviation = -self.deviation_coef * self._calculate_path_deviation(tcp_pos)

        # Total reward
        total_reward = r_distance + r_waypoint + r_smooth + r_deviation

        return total_reward, reached.item() > 0

    def _calculate_path_deviation(self, tcp_pos: torch.Tensor) -> float:
        """
        Calculate deviation from planned path (simplified line segment distance)

        Args:
            tcp_pos: Current TCP position

        Returns:
            Deviation distance
        """
        if len(self.current_ts_waypoints) < 2:
            return 0.0

        # Get current and next waypoints
        current_wp = self.current_ts_waypoints[self.current_waypoint_index]

        # Check if we're at the last waypoint
        if self.current_waypoint_index >= len(self.current_ts_waypoints) - 1:
            return 0.0

        next_wp = self.current_ts_waypoints[self.current_waypoint_index + 1]

        # Simple deviation: distance from line segment between current and next waypoint
        current_wp_pos = torch.tensor(current_wp.cartesian_position, device=self.device)
        next_wp_pos = torch.tensor(next_wp.cartesian_position, device=self.device)

        # Project onto line segment and calculate perpendicular distance
        line_vec = next_wp_pos - current_wp_pos
        point_vec = tcp_pos - current_wp_pos

        line_len = torch.norm(line_vec)
        if line_len < 1e-6:
            return torch.norm(point_vec).item()

        line_unitvec = line_vec / line_len
        proj_length = torch.dot(point_vec, line_unitvec).clamp(0, line_len)
        proj_point = current_wp_pos + proj_length * line_unitvec

        deviation = torch.norm(tcp_pos - proj_point)
        return deviation.item()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment with trajectory tracking support

        Args:
            action: 6D normalized action array [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Use parent step function for physics simulation
        obs, _, terminated, truncated, info = super().step(action)

        # Ensure terminated and truncated are Python booleans (not tensors)
        terminated = bool(terminated) if terminated is not None else False
        truncated = bool(truncated) if truncated is not None else False

        if self.mode == "trajectory_tracking":
            # Update waypoint progression if reached
            current_tcp = self._forward_kinematics(self.joint_positions[0])
            if self.ts_planner.update_progress(current_tcp.cpu().numpy()):
                print(f"📍 Waypoint {self.current_waypoint_index + 1}/{len(self.current_ts_waypoints)} reached")

            # Calculate trajectory-specific reward
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward, waypoint_reached = self._trajectory_reward(current_tcp, action_tensor)

            # Check if trajectory is completed
            if (len(self.current_ts_waypoints) > 0 and
                self.current_waypoint_index >= len(self.current_ts_waypoints) - 1):
                final_waypoint = self.current_ts_waypoints[-1]
                final_distance = torch.norm(current_tcp - torch.tensor(final_waypoint.cartesian_position, device=self.device))
                if final_distance < final_waypoint.tolerance:
                    self.trajectory_completed = True
                    terminated = True
                    print(f"🎉 Trajectory completed successfully!")

            # Update observation for trajectory tracking
            obs = self.get_observation()

            # Add trajectory info to info dict
            info.update({
                'trajectory_mode': True,
                'current_waypoint': self.current_waypoint_index,
                'total_waypoints': len(self.current_ts_waypoints),
                'waypoint_reached': waypoint_reached,
                'trajectory_completed': self.trajectory_completed,
                'distance_to_waypoint': self.ts_planner.get_distance_to_current_waypoint(current_tcp.cpu().numpy())
            })
        else:
            # Use base environment reward for point-to-point mode
            reward = self._calculate_reward()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        # Reset base environment
        obs, info = super().reset(seed=seed, options=options)

        # Reset trajectory tracking state
        self.current_waypoint_index = 0
        self.trajectory_completed = False

        # For trajectory tracking mode, we could optionally plan a new trajectory here
        if self.mode == "trajectory_tracking" and options and 'plan_trajectory' in options:
            plan_options = options['plan_trajectory']
            if 'start_tcp' in plan_options and 'goal_tcp' in plan_options:
                self.plan_trajectory(plan_options['start_tcp'], plan_options['goal_tcp'])

        # Return appropriate observation
        obs = self.get_observation()

        info.update({
            'trajectory_mode': self.mode == "trajectory_tracking",
            'trajectory_completed': False,
            'current_waypoint': 0,
            'total_waypoints': len(self.current_ts_waypoints) if self.mode == "trajectory_tracking" else 0
        })

        return obs, info

    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Get trajectory tracking statistics"""
        if self.mode != "trajectory_tracking":
            return {'trajectory_mode': False}

        return {
            'trajectory_mode': True,
            'current_waypoint': self.current_waypoint_index + 1,
            'total_waypoints': len(self.current_ts_waypoints),
            'progress_percentage': self.ts_planner.get_planning_progress()['progress_percentage'],
            'planner_stats': self.ts_planner.get_statistics(),
            'trajectory_completed': self.trajectory_completed
        }


def test_trajectory_environment():
    """Test UR10e trajectory tracking environment"""
    print("🧪 Testing UR10e Trajectory Tracking Environment")

    # Create environment in trajectory tracking mode
    env = UR10eTrajectoryEnv(config_path="config.yaml", mode="trajectory_tracking")

    # Reset environment
    obs, info = env.reset()
    print(f"📊 Initial observation shape: {obs.shape}")
    print(f"🎯 Environment info: {info}")

    # Plan a trajectory
    start_tcp = np.array([0.5, 0.0, 0.3])
    goal_tcp = np.array([-0.3, 0.4, 0.7])

    if env.plan_trajectory(start_tcp, goal_tcp):
        print("✅ Trajectory planned successfully")

        # Test waypoint information
        current_waypoint = env.get_current_waypoint()
        print(f"   Current waypoint: {current_waypoint}")

        # Test observation structure
        obs = env.get_observation()
        print(f"   Observation shape: {obs.shape}")
        print(f"   First 6 values (joint pos): {obs[:6]}")
        print(f"   Next 6 values (joint vel): {obs[6:12]}")
        print(f"   Next 3 values (delta to waypoint): {obs[12:15]}")
        print(f"   Progress value: {obs[15]}")
        print(f"   Last 3 values (TCP pos): {obs[16:19]}")

        # Test a few steps
        for step in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step + 1}: reward={reward:.3f}, waypoint={info.get('current_waypoint', 0)}")

            if terminated:
                print("Episode completed!")
                break
    else:
        print("❌ Trajectory planning failed")

    # Close environment
    env.close()
    print("✅ Trajectory environment test completed")


if __name__ == "__main__":
    test_trajectory_environment()