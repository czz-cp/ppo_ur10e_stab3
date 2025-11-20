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
        # Trajectory tracking state (vectorized)
        self.current_ts_waypoints = []
        self.current_waypoint_index = np.zeros(self.num_envs, dtype=np.int32)
        self.trajectory_completed = np.zeros(self.num_envs, dtype=bool)


        # Reward function parameters (MUST be set before planner initialization)
        self.waypoint_threshold = self.trajectory_config.get('waypoint_threshold', 0.15)
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
        workspace_bounds_list = []
        for axis in ['x', 'y', 'z']:
            bounds = self.task_space_config.get('workspace_bounds', {}).get(axis, [-0.5, 0.5])
            workspace_bounds_list.append(bounds)
        workspace_bounds = np.array(workspace_bounds_list)

        # ✅ 为每个 env 创建一个 planner（各自维护 waypoint_index）
        self.ts_planners = [
            TaskSpacePlannerInterface(
                workspace_bounds=workspace_bounds,
                waypoint_spacing=0.1,
                replanning_threshold=self.ts_rrt_config.get('replanning_threshold', 0.1),
                max_waypoints=self.ts_rrt_config.get('max_waypoints', 50)
            )
            for _ in range(self.num_envs)
        ]
        self.ts_planner = self.ts_planners[0]  # 兼容旧引用

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
            self.current_waypoint_index[:] = 0
            self.trajectory_completed[:] = False

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
            self.current_waypoint_index[:] = 0
            self.trajectory_completed[:] = False

            for p in self.ts_planners:
                p.current_waypoints = result.waypoints
                p.current_waypoint_index = 0

            print(f"✅ Trajectory planned: {len(self.current_ts_waypoints)} waypoints")
            return True
        else:
            print(f"❌ Trajectory planning failed: {result.error_message}")
            self.current_ts_waypoints = []
            self.current_waypoint_index[:] = 0
            return False

    def set_waypoints(self, waypoints: List[TSWaypoint]):
        """
        Set waypoints directly (for testing or external planning)

        Args:
            waypoints: List of TSWaypoint objects
        """
        self.current_ts_waypoints = waypoints
        self.current_waypoint_index[:] = 0
        self.trajectory_completed[:] = False
        print(f"📍 Set {len(waypoints)} waypoints for trajectory tracking")

    def get_current_waypoint_(self) -> Optional[TSWaypoint]:
        """Get current waypoint for trajectory tracking"""
        if not self.current_ts_waypoints or self.current_waypoint_index >= len(self.current_ts_waypoints):
            return None
        return self.current_ts_waypoints[self.current_waypoint_index]
    
    def get_current_waypoint(self, env_id: int = 0) -> Optional[TSWaypoint]:
        """Get current waypoint for a specific env (default env0 for logging/callback)."""
        if self.mode != "trajectory_tracking":
            return None
        if not self.current_ts_waypoints:
            return None

        # 每个 env 有自己的 planner 和标量 index
        planner = self.ts_planners[env_id] if hasattr(self, "ts_planners") else self.ts_planner
        if planner.current_waypoints is None or len(planner.current_waypoints) == 0:
            return None

        idx = int(planner.current_waypoint_index)
        if idx >= len(planner.current_waypoints):
            return None
        return planner.current_waypoints[idx]

    
    def get_observation(self):
        obs_list = []
        for i in range(self.num_envs):
            q = self.joint_positions[i].detach().cpu().numpy()
            qd = self.joint_velocities[i].detach().cpu().numpy()
            tcp = self._forward_kinematics(self.joint_positions[i]).detach().cpu().numpy()

            if self.mode == "trajectory_tracking" and self.current_ts_waypoints:
                wp = self.ts_planners[i].get_current_waypoint()
                if wp is not None:
                    wp_pos = np.asarray(wp.cartesian_position, np.float32)
                    delta = wp_pos - tcp
                    prog = self.ts_planners[i].current_waypoint_index / max(1, len(self.current_ts_waypoints)-1)
                else:
                    delta = np.zeros(3, np.float32)
                    prog = 1.0
            else:
                target = self.target_positions[i].detach().cpu().numpy()
                delta = target - tcp
                prog = 0.0

            obs_i = np.concatenate([q, qd, delta, np.array([prog], np.float32), tcp]).astype(np.float32)
            obs_list.append(obs_i)

        return obs_list[0] if self.num_envs == 1 else np.stack(obs_list, axis=0)
    
    def _trajectory_reward(self, tcp_pos, action_tensor, env_id: int = 0):
        planner = self.ts_planners[env_id]
        waypoint = planner.get_current_waypoint()
        if waypoint is None:
            return -10.0, False

        waypoint_pos = torch.tensor(waypoint.cartesian_position, device=tcp_pos.device)
        distance = torch.norm(tcp_pos - waypoint_pos)

        reward = -distance.item()
        waypoint_reached = distance.item() < waypoint.tolerance
        if waypoint_reached:
            reward += self.waypoint_bonus

        # 平滑项/偏离项如果启用，也用 planner 的 index
        if self.use_deviation_penalty and len(planner.current_waypoints) > 1:
            reward -= self.deviation_coef * self._calculate_path_deviation(tcp_pos, env_id)

        reward -= self.smooth_coef * torch.norm(action_tensor).item()
        return reward, waypoint_reached


    def _calculate_path_deviation(self, tcp_pos: torch.Tensor,env_id: int = 0) -> float:
        """
        Calculate deviation from planned path (simplified line segment distance)

        Args:
            tcp_pos: Current TCP position

        Returns:
            Deviation distance
        """
        planner = self.ts_planners[env_id]
        idx = planner.current_waypoint_index
        if idx <= 0 or idx >= len(planner.current_waypoints):
            return 0.0
        current_wp = planner.current_waypoints[idx-1]
        next_wp = planner.current_waypoints[idx]

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
    
    def step(self, action: np.ndarray):
        """
        Step the environment with trajectory tracking support (vectorized).

        Args:
            action: 6D normalized action array [-1, 1]
                    shape can be (6,) for single env or (num_envs, 6) for multi env

        Returns:
            (obs, reward, terminated, truncated, info)
            - single env: reward float, terminated bool, truncated bool, info dict
            - multi env: reward (num_envs,), terminated (num_envs,), truncated (num_envs,), info list[dict]
        """
        # --------- debug: action change ----------
        if hasattr(self, "_last_action"):
            try:
                action_change = np.linalg.norm(action - self._last_action)
                # print(f"🔄 Action change magnitude: {action_change:.6f}")
            except Exception:
                pass
        self._last_action = action.copy()

        # --------- record joint positions before ----------
        joint_pos_before = self.joint_positions.clone()

        # --------- physics step from parent ----------
        obs_base, reward_base, terminated_base, truncated_base, info_base = super().step(action)

        # --------- record joint positions after ----------
        joint_pos_after = self.joint_positions
        try:
            joint_change = torch.norm(joint_pos_after - joint_pos_before, dim=1).mean().item()
            # print(f"🔧 Joint position change(mean): {joint_change:.6f}")
        except Exception:
            pass

        # --------- normalize terminated/truncated to bool vectors ----------
        def _to_bool_vec(x):
            if x is None:
                return np.zeros(self.num_envs, dtype=bool)
            if isinstance(x, (bool, np.bool_)):
                return np.full(self.num_envs, bool(x), dtype=bool)
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x).reshape(-1)
            if x.size == 1:
                return np.full(self.num_envs, bool(x[0]), dtype=bool)
            return x.astype(bool)

        term_base = _to_bool_vec(terminated_base)
        trunc_base = _to_bool_vec(truncated_base)

        # --------- trajectory tracking mode ----------
        if self.mode == "trajectory_tracking":
            # action tensor to device
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)  # (1, 6)

            rewards = np.zeros(self.num_envs, dtype=np.float32)
            term_traj = np.zeros(self.num_envs, dtype=bool)
            infos = []

            for i in range(self.num_envs):
                # 1) TCP per env
                current_tcp = self._forward_kinematics(self.joint_positions[i])

                # debug: tcp change per env
                if not hasattr(self, "_last_tcp_list"):
                    self._last_tcp_list = [None for _ in range(self.num_envs)]
                last_tcp = self._last_tcp_list[i]
                if last_tcp is not None:
                    try:
                        tcp_change = torch.norm(current_tcp - last_tcp).item()
                        # print(f"[env {i}] 📍 TCP change: {tcp_change:.6f}")
                    except Exception:
                        pass
                self._last_tcp_list[i] = current_tcp.detach().clone()

                # 2) update planner progress per env
                advanced = self.ts_planners[i].update_progress(current_tcp.detach().cpu().numpy())
                if advanced:
                    self.current_waypoint_index[i] = self.ts_planners[i].current_waypoint_index
                    print(f"📍 [env {i}] Waypoint {self.current_waypoint_index[i] + 1}/{len(self.current_ts_waypoints)} reached")

                # 3) reward per env
                r_i, wp_reached_i = self._trajectory_reward(current_tcp, action_tensor[i], env_id=i)
                rewards[i] = r_i

                # 4) completion check per env
                current_wp = self.ts_planners[i].get_current_waypoint()
                if current_wp is None and len(self.current_ts_waypoints) > 0:
                    self.trajectory_completed[i] = True
                    term_traj[i] = True
                    print(f"🎉 [env {i}] Trajectory completed successfully!")
                elif current_wp is not None and self.current_waypoint_index[i] == len(self.current_ts_waypoints) - 1:
                    final_dist = torch.norm(
                        current_tcp - torch.tensor(current_wp.cartesian_position, device=self.device)
                    )
                    if final_dist < current_wp.tolerance:
                        self.trajectory_completed[i] = True
                        term_traj[i] = True
                        print(f"🎉 [env {i}] Trajectory completed successfully!")

                # 5) info per env
                infos.append({
                    "trajectory_mode": True,
                    "current_waypoint": int(self.current_waypoint_index[i]),
                    "total_waypoints": len(self.current_ts_waypoints),
                    "waypoint_reached": bool(wp_reached_i),
                    "trajectory_completed": bool(self.trajectory_completed[i]),
                    "distance_to_waypoint": float(
                        self.ts_planners[i].get_distance_to_current_waypoint(current_tcp.detach().cpu().numpy())
                    )
                })

            obs = self.get_observation()
            terminated_out = np.logical_or(term_base, term_traj)
            truncated_out = trunc_base

            if self.num_envs == 1:
                return obs, float(rewards[0]), bool(terminated_out[0]), bool(truncated_out[0]), infos[0]
            return obs, rewards, terminated_out, truncated_out, infos

        # --------- point-to-point / base mode ----------
        else:
            # 这里直接用父类 reward（它本身就是向量化的）
            reward_out = reward_base
            if torch.is_tensor(reward_out):
                reward_out = reward_out.detach().cpu().numpy()
            reward_out = np.asarray(reward_out).reshape(-1)
            if reward_out.size == 1:
                reward_out = np.full(self.num_envs, float(reward_out[0]), dtype=np.float32)

            if self.num_envs == 1:
                return obs_base, float(reward_out[0]), bool(term_base[0]), bool(trunc_base[0]), info_base
            # info_base 如果是 dict，就广播成 list[dict]
            if isinstance(info_base, dict):
                info_base = [info_base.copy() for _ in range(self.num_envs)]
            return obs_base, reward_out, term_base, trunc_base, info_base


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""

        # 1) 如果 options 里指定了 mode，就先切换（保留你原来的逻辑）
        if options and "mode" in options:
            self.set_mode(options["mode"])

        # 2) 先 reset 底层增量力矩环境
        obs, info = super().reset(seed=seed, options=options)

        # 3) 重置轨迹跟踪状态
        self.current_waypoint_index[:] = 0
        self.trajectory_completed[:] = False
        
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
        """info.update(
            {
                "trajectory_mode": self.mode == "trajectory_tracking",
                "trajectory_completed": False,
                "current_waypoint": 0,
                "total_waypoints": len(self.current_ts_waypoints)
                if self.mode == "trajectory_tracking"
                else 0,
            }
        )"""
        infos = []
        for i in range(self.num_envs):
            infos.append({
                "trajectory_mode": self.mode == "trajectory_tracking",
                "trajectory_completed": False,
                "current_waypoint": int(self.current_waypoint_index[i]),
                "total_waypoints": len(self.current_ts_waypoints) if self.mode=="trajectory_tracking" else 0
            })

        #return obs, info
        return (obs, infos[0]) if self.num_envs == 1 else (obs, infos)
    
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