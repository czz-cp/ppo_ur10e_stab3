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

        # Trajectory tracking configuration--
        self.mode: str = None        # 初始化为 None，后面用 set_mode 设定
        self.ts_planner = None       # 先占一个属性，避免 hasattr 问题
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

        # 初始化规划器（如果初始 mode 需要）+ 设置 mode
        self.set_mode(mode)

        # Initialize Task-Space planner for trajectory tracking mode
        if self.mode == "trajectory_tracking":
            self._init_task_space_planner()

        # Override observation space always (19D)
        self._define_observation_space_19d()

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

    def set_mode(self, mode: str):
        """
        切换环境模式：
        - "trajectory_tracking": 启用任务空间轨迹规划 + 轨迹奖励
        - "point_to_point": 使用基础环境的点对点奖励

        约束：
        - 只能在 episode 之间调用（即 reset 前后），不要在单个 episode 中途换。
        - 观测维度固定为 19D，本函数不会修改 observation_space。
        """
        assert mode in ["trajectory_tracking", "point_to_point"], \
            f"Unsupported mode: {mode}"

        if self.mode == mode:
            return  # 不需要重复切换

        self.mode = mode

        if mode == "trajectory_tracking":
            # 确保任务空间规划器已初始化
            if self.ts_planner is None:
                self._init_task_space_planner()
        else:
            # 切回 point_to_point 模式：
            # 清空当前轨迹
            self.current_ts_waypoints = []
            self.current_waypoint_index = 0
            self.trajectory_completed = False

        print(f"🔁 Switched UR10eTrajectoryEnv mode to: {self.mode}")

    def _define_observation_space_19d(self):
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
        统一 19D 观测格式：

        [joint_pos(6) + joint_vel(6) + vec3(3) + progress(1) + tcp_pos(3)]

        - trajectory_tracking:
            vec3 = current_waypoint_pos - tcp_pos
            progress = 当前轨迹进度 [0,1]
        - point_to_point:
            vec3 = target_pos - tcp_pos
            progress = 0.0
        """
        obs_list = []

        for i in range(self.num_envs):
            # 关节角 / 速度：直接用前 6 自由度
            joint_pos = self.joint_positions[i, :6].cpu().numpy()
            joint_vel = self.joint_velocities[i, :6].cpu().numpy()

            # 当前 TCP 位置
            tcp_pos = self._forward_kinematics(self.joint_positions[i]).cpu().numpy()

            if self.mode == "trajectory_tracking":
                current_waypoint = self.get_current_waypoint()
                if current_waypoint is not None:
                    waypoint_pos = np.asarray(current_waypoint.cartesian_position, dtype=np.float32)
                    delta_vec = waypoint_pos - tcp_pos
                    total_wps = len(self.current_ts_waypoints)
                    if total_wps > 1:
                        progress = float(self.current_waypoint_index) / float(total_wps - 1)
                    else:
                        progress = 0.0
                else:
                    delta_vec = np.zeros(3, dtype=np.float32)
                    progress = 0.0
            else:
                # point_to_point: 用 target_pos
                if hasattr(self, "target_positions") and self.target_positions is not None:
                    target_pos = self.target_positions[i].cpu().numpy()
                else:
                    target_pos = np.zeros(3, dtype=np.float32)
                delta_vec = target_pos - tcp_pos
                progress = 0.0

            obs = np.concatenate([
                joint_pos.astype(np.float32),     # 6
                joint_vel.astype(np.float32),     # 6
                delta_vec.astype(np.float32),     # 3
                np.array([progress], np.float32), # 1
                tcp_pos.astype(np.float32)        # 3
            ])  # → 19

            assert obs.shape[0] == 19, f"Expected 19D observation, got {obs.shape[0]}D"

            # 如果你想这里也做归一化，可以在这加 self._normalize_state(obs)
            obs_list.append(obs)

        if self.num_envs == 1:
            return obs_list[0]
        return np.stack(obs_list, axis=0)

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
        waypoint_pos = torch.tensor(current_waypoint.cartesian_position, device=tcp_pos.device, dtype=torch.float32)

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
        current_wp_pos = torch.tensor(current_wp.cartesian_position, device=tcp_pos.device)
        next_wp_pos = torch.tensor(next_wp.cartesian_position, device=tcp_pos.device)

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
        # 添加调试信息，查看动作是否发生变化
        if hasattr(self, '_last_action'):
            action_change = np.linalg.norm(action - self._last_action)
            #print(f"🔄 Action change magnitude: {action_change:.6f}")
        self._last_action = action.copy()
        
        # 应用动作前记录关节位置
        joint_pos_before = self.joint_positions[0].clone()
        
        # Use parent step function for physics simulation
        obs, _, terminated, truncated, info = super().step(action)

        # 应用动作后记录关节位置
        joint_pos_after = self.joint_positions[0]
        joint_change = torch.norm(joint_pos_after - joint_pos_before).item()
        #print(f"🔧 Joint position change: {joint_change:.6f}")

        # Ensure terminated and truncated are Python booleans (not tensors)
        terminated = bool(terminated) if terminated is not None else False
        truncated = bool(truncated) if truncated is not None else False

        if self.mode == "trajectory_tracking":
            # Update waypoint progression if reached
            current_tcp = self._forward_kinematics(self.joint_positions[0])
            
            # 添加调试信息，查看TCP位置是否发生变化
            if hasattr(self, '_last_tcp'):
                tcp_change = torch.norm(current_tcp - self._last_tcp).item()
                #print(f"📍 TCP position change: {tcp_change:.6f}")
            self._last_tcp = current_tcp.clone()
            
            # 1) 更新 planner 的进度
            advanced = self.ts_planner.update_progress(current_tcp.cpu().numpy())
            if advanced:
                # ⭐ 关键：用 planner 的 index 同步 env 的 index
                self.current_waypoint_index = self.ts_planner.current_waypoint_index
                print(f"📍 Waypoint {self.current_waypoint_index + 1}/{len(self.current_ts_waypoints)} reached")

            # Calculate trajectory-specific reward
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward, waypoint_reached = self._trajectory_reward(current_tcp, action_tensor)

             # 3) 终点判定（用 planner 的 current waypoint）
            current_wp = self.ts_planner.get_current_waypoint()
            if current_wp is None and len(self.current_ts_waypoints) > 0:
                # planner 认为已经走完所有 waypoint
                self.trajectory_completed = True
                terminated = True
                print("🎉 Trajectory completed successfully!")
            elif current_wp is not None and self.current_waypoint_index == len(self.current_ts_waypoints) - 1:
                # 最后一个 waypoint 再做一次安全检查
                final_dist = torch.norm(
                    current_tcp - torch.tensor(current_wp.cartesian_position, device=self.device)
                )
                if final_dist < current_wp.tolerance:
                    self.trajectory_completed = True
                    terminated = True
                    print("🎉 Trajectory completed successfully!")

            # Update observation for trajectory tracking
            obs = self.get_observation()

            # 5) 在这里统一算 distance_to_waypoint，保证和 obs / progress 一致
            if current_wp is not None:
                distance_to_waypoint = float(
                    np.linalg.norm(current_tcp.cpu().numpy() - current_wp.cartesian_position)
                )
            else:
                distance_to_waypoint = float("inf")

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

        # 1) 如果 options 里指定了 mode，就先切换（保留你原来的逻辑）
        if options and "mode" in options:
            self.set_mode(options["mode"])

        # 2) 先 reset 底层增量力矩环境
        obs, info = super().reset(seed=seed, options=options)

        # 3) 重置轨迹跟踪状态
        self.current_waypoint_index = 0
        self.trajectory_completed = False
        # 删除_prev_distance_to_waypoint变量，确保每次reset都重新开始
        if hasattr(self, '_prev_distance_to_waypoint'):
            delattr(self, '_prev_distance_to_waypoint')
            
        # 删除调试用的变量
        if hasattr(self, '_last_action'):
            delattr(self, '_last_action')
        if hasattr(self, '_last_tcp'):
            delattr(self, '_last_tcp')

        planned = False

        # 4) 如果在 trajectory_tracking 模式，优先看 options 里是否显式给了 start/goal
        if self.mode == "trajectory_tracking":
            if options is not None and "plan_trajectory" in options:
                plan_options = options["plan_trajectory"]
                if "start_tcp" in plan_options and "goal_tcp" in plan_options:
                    planned = self.plan_trajectory(
                        np.array(plan_options["start_tcp"], dtype=np.float32),
                        np.array(plan_options["goal_tcp"], dtype=np.float32),
                    )

            # 5) 如果没有通过 options 规划成功，就自动采样一个轨迹
            if not planned:
                # 当前 TCP 作为起点
                with torch.no_grad():
                    start_tcp = (
                        self._forward_kinematics(self.joint_positions[0])
                        .cpu()
                        .numpy()
                    )

                # 从 task_space_config 里读 workspace_bounds
                ws_cfg = self.task_space_config.get("workspace_bounds", {})
                def _axis(name, default):
                    return ws_cfg.get(name, default)

                goal_tcp = np.array(
                    [
                        np.random.uniform(*_axis("x", [-0.6, 0.6])),
                        np.random.uniform(*_axis("y", [-0.6, 0.6])),
                        np.random.uniform(*_axis("z", [0.2, 0.8])),
                    ],
                    dtype=np.float32,
                )

                planned = self.plan_trajectory(start_tcp, goal_tcp)

                if not planned:
                    print("⚠️ Auto trajectory planning failed in reset()")
                else:
                    print(
                        f"🔁 New episode trajectory planned: "
                        f"{len(self.current_ts_waypoints)} waypoints"
                    )

        # 6) 生成观测（19D：joint_pos + joint_vel + delta_to_waypoint + progress + tcp_pos）
        obs = self.get_observation()

        # 7) 补充 info
        info.update(
            {
                "trajectory_mode": self.mode == "trajectory_tracking",
                "trajectory_completed": False,
                "current_waypoint": 0,
                "total_waypoints": len(self.current_ts_waypoints)
                if self.mode == "trajectory_tracking"
                else 0,
            }
        )

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