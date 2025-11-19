"""
UR10e Environment with Incremental Torque Control for Stable-Baselines3

Pure RL control environment that replaces RL-PID hybrid approach.
Uses 6D incremental torque control (Δτ) for direct robot control.
Compatible with Stable-Baselines3 and Isaac Gym physics simulation.
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
    print("✅ Isaac Gym imported successfully in ur10e_incremental_env")
except (ImportError, KeyError) as e:
    print(f"❌ Failed to import Isaac Gym in ur10e_incremental_env: {e}")
    sys.exit(1)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
from typing import Dict, Any, Optional
from collections import deque


class RewardNormalizer:
    """
    奖励归一化器

    用于稳定PPO训练的奖励归一化技术，支持在线更新和多种归一化策略
    """

    def __init__(self,
                 gamma: float = 0.99,
                 clip_range: float = 5.0,
                 epsilon: float = 1e-8,
                 normalize_method: str = 'running_stats',
                 warmup_steps: int = 100,
                 history_size: int = 10000):
        """
        初始化奖励归一化器

        Args:
            gamma: 折扣因子，用于计算折扣奖励统计
            clip_range: 归一化值裁剪范围
            epsilon: 数值稳定性参数
            normalize_method: 归一化方法 ['running_stats', 'batch_stats', 'rank']
            warmup_steps: 预热步数，初期不进行归一化
            history_size: 奖励历史记录大小
        """
        self.gamma = gamma
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.normalize_method = normalize_method
        self.warmup_steps = warmup_steps
        self.history_size = history_size

        # 运行时统计量
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 0
        self.beta = 0.99  # 指数移动平均系数

        # 奖励历史
        self.reward_history = []
        self.discounted_reward_history = []

        # 批次统计
        self.batch_rewards = []

    def update(self, reward: float, done: bool = False):
        """
        更新归一化器统计量

        Args:
            reward: 当前奖励值
            done: 是否回合结束
        """
        self.reward_history.append(reward)
        self.running_count += 1

        # 指数移动平均更新
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward
        delta = reward - self.running_mean
        self.running_var = self.beta * self.running_var + (1 - self.beta) * delta * delta

        # 维护历史记录在合理范围内
        if len(self.reward_history) > self.history_size:
            self.reward_history = self.reward_history[-self.history_size//2:]

        # 回合结束时计算折扣奖励统计
        if done and len(self.reward_history) > 1:
            self._update_discounted_stats()

    def _update_discounted_stats(self):
        """更新折扣奖励统计"""
        if not self.reward_history:
            return

        # 计算最近一个episode的折扣奖励
        discounted_rewards = []
        reward_sum = 0.0
        for reward in reversed(self.reward_history):
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        self.discounted_reward_history.extend(discounted_rewards)

        # 维护折扣奖励历史
        if len(self.discounted_reward_history) > self.history_size:
            self.discounted_reward_history = self.discounted_reward_history[-self.history_size//2:]

    def normalize(self, reward: float) -> float:
        """
        归一化单个奖励

        Args:
            reward: 原始奖励值

        Returns:
            normalized_reward: 归一化后的奖励值
        """
        if self.running_count < self.warmup_steps:
            return reward  # 预热期不归一化

        if self.normalize_method == 'running_stats':
            return self._normalize_running_stats(reward)
        elif self.normalize_method == 'batch_stats':
            return self._normalize_batch_stats(reward)
        elif self.normalize_method == 'rank':
            return self._normalize_rank(reward)
        else:
            return reward

    def _normalize_running_stats(self, reward: float) -> float:
        """使用运行统计量归一化"""
        std = np.sqrt(self.running_var + self.epsilon)
        normalized = (reward - self.running_mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _normalize_batch_stats(self, reward: float) -> float:
        """使用批次统计量归一化"""
        if len(self.reward_history) < 10:
            return reward

        # 使用最近的奖励作为批次
        recent_rewards = self.reward_history[-min(100, len(self.reward_history)):]
        batch_mean = np.mean(recent_rewards)
        batch_std = np.std(recent_rewards) + self.epsilon

        normalized = (reward - batch_mean) / batch_std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _normalize_rank(self, reward: float) -> float:
        """使用秩归一化（均匀分布）"""
        if len(self.reward_history) < 10:
            return reward

        # 计算当前奖励在历史中的百分位
        count_smaller = sum(1 for r in self.reward_history if r < reward)
        percentile = count_smaller / len(self.reward_history)

        # 映射到[-1, 1]范围
        normalized = 2 * percentile - 1
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def get_stats(self) -> dict:
        """获取归一化器统计信息"""
        return {
            'method': self.normalize_method,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'running_std': np.sqrt(self.running_var + self.epsilon),
            'count': self.running_count,
            'recent_mean': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'recent_std': np.std(self.reward_history[-100:]) if len(self.reward_history) > 1 else 0.0,
            'history_size': len(self.reward_history),
            'warmup_progress': min(1.0, self.running_count / self.warmup_steps)
        }

    def reset(self):
        """重置归一化器（保留学习到的统计量）"""
        self.reward_history = []
        self.batch_rewards = []


# Global Isaac Gym instance manager - following Isaac Gym official patterns
_GLOBAL_GYM = None
_GLOBAL_SIM = None
_INITIALIZED = False

def get_global_gym_and_sim():
    """Get or create global Isaac Gym gym and sim instances"""
    global _GLOBAL_GYM, _GLOBAL_SIM, _INITIALIZED

    if not _INITIALIZED:
        print("🏗️  Initializing global Isaac Gym gym and sim instances...")

        # Acquire gym (only once per process)
        _GLOBAL_GYM = gymapi.acquire_gym()

        # Create sim parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.01  # Default timestep
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False  # Critical: disable to avoid CUDA issues

        # Create sim (only once per process)
        _GLOBAL_SIM = _GLOBAL_GYM.create_sim(
            compute_device=0,  # Use GPU 0
            graphics_device=-1,  # Headless mode
            type=gymapi.SIM_PHYSX,
            params=sim_params
        )

        if _GLOBAL_SIM is None:
            raise Exception("Failed to create Isaac Gym simulator")

        _INITIALIZED = True
        print("✅ Global Isaac Gym gym and sim instances created successfully")

    return _GLOBAL_GYM, _GLOBAL_SIM
from typing import Dict, Any, Tuple, Optional, List
import yaml
from collections import deque
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class UR10eIncrementalEnv(gym.Env):
    """
    UR10e Environment with Incremental Torque Control

    Pure RL control using Stable-Baselines3 compatible interface.
    Uses 6D incremental torque control instead of PID parameter tuning.

    Key Features:
    - Action Space: 6D continuous (Δτ₁, Δτ₂, ..., Δτ₆)
    - State Space: 16D observation
    - Direct torque control with safety limits
    - Isaac Gym physics simulation
    - Stable-Baselines3 compatible (gym.Env interface)
    """

    def __init__(self, config_path: str = "config.yaml", num_envs: int = 1):
        """
        Initialize the environment

        Args:
            config_path: Path to configuration file
            num_envs: Number of parallel environments
        """
        super().__init__()

        # Load configuration
        self.config = self._load_config(config_path)
        self.num_envs = num_envs

        # Device configuration
        self.device_id = self.config.get('env', {}).get('device_id', 0)
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")

        # Environment parameters
        self.max_steps = self.config.get('env', {}).get('max_steps', 500)
        self.dt = self.config.get('env', {}).get('dt', 0.01)

        # UR10e joint limits and torque limits
        self.ur10e_joint_limits = torch.tensor([
            [-2.0*np.pi, 2.0*np.pi],  # Shoulder pan
            [-np.pi, np.pi],          # Shoulder lift
            [-np.pi, np.pi],          # Elbow
            [-2.0*np.pi, 2.0*np.pi],  # Wrist 1
            [-2.0*np.pi, 2.0*np.pi],  # Wrist 2
            [-2.0*np.pi, 2.0*np.pi]   # Wrist 3
        ], dtype=torch.float32)

        # UR10e official torque limits (N⋅m) from UR10e User Manual v5.8
        self.ur10e_torque_limits = torch.tensor([
            330.0,  # shoulder_pan_joint
            330.0,  # shoulder_lift_joint
            150.0,  # elbow_joint (corrected from UR10e spec)
            54.0,   # wrist_1_joint
            54.0,   # wrist_2_joint
            54.0    # wrist_3_joint
        ], dtype=torch.float32)

        # UR10e official velocity limits (rad/s) from joint_limits.yaml
        self.ur10e_velocity_limits = torch.tensor([
            2.0944,  # shoulder_pan_joint: 120°/s
            2.0944,  # shoulder_lift_joint: 120°/s
            3.1416,  # elbow_joint: 180°/s
            3.1416,  # wrist_1_joint: 180°/s
            3.1416,  # wrist_2_joint: 180°/s
            3.1416   # wrist_3_joint: 180°/s
        ], dtype=torch.float32)

        # 🎯 Joint-specific action scaling for better control
        # 基于关节力矩限制的归一化，使[-1,1]动作对应合适的增量力矩
        self.joint_specific_scales = torch.tensor([
            0.06,  # 关节1: 330N·m × 0.06 = ~20N·m (shoulder_pan)
            0.06,  # 关节2: 330N·m × 0.06 = ~20N·m (shoulder_lift)
            0.13,  # 关节3: 150N·m × 0.13 = ~20N·m (elbow) - 修正!
            0.37,  # 关节4: 54N·m × 0.37 = ~20N·m (wrist_1)
            0.37,  # 关节5: 54N·m × 0.37 = ~20N·m (wrist_2)
            0.37   # 关节6: 54N·m × 0.37 = ~20N·m (wrist_3)
        ], device=self.device)

        # Safety parameters (must be set before defining spaces)
        self.max_increment_torque = self.config.get('control', {}).get('max_increment_torque', 20.0)
        self.emergency_stop_threshold = self.config.get('safety', {}).get('emergency_stop_threshold', 0.5)

        # 🆕 Momentum and control parameters
        self.torque_momentum_decay = self.config.get('control', {}).get('torque_momentum_decay', 0.95)
        self.velocity_penalty_strength = self.config.get('control', {}).get('velocity_penalty_strength', 10.0)

        # Initialize PyTorch tensors BEFORE Isaac Gym (following working implementation pattern)
        self._init_pytorch_tensors()

        # Initialize Isaac Gym
        self._init_isaac_gym()

        # Define action and observation spaces
        self._define_spaces()

        # Initialize state tracking
        self.current_step = 0
        self.episode_count = 0
        self.success_count = 0
        self.reward_history = deque(maxlen=1000)

        # 🎯 State normalization setup
        self._init_state_normalization()

        # 🎯 Reward normalization setup
        self._init_reward_normalization()

        print(f"✅ UR10eIncrementalEnv initialized:")
        print(f"   🎯 Control Mode: Pure RL (6D incremental torque)")
        print(f"   🎬 Device: {self.device}")
        print(f"   🔧 Parallel envs: {num_envs}")
        print(f"   📐 Action space: {self.action_space}")
        print(f"   📊 Observation space: {self.observation_space}")
        print(f"   ⚡ Max incremental torque: ±{self.max_increment_torque} N⋅m")

    def _define_spaces(self):
        """Define action and observation spaces for Stable-Baselines3"""

        # We'll set action space after knowing the actual DOF count in _init_state_tensors
        # Action space will be [num_dofs] incremental torque control

        # Observation Space: 18D - Complete state information for better learning
        # [joint_pos(6), joint_vel(6), target_pos(3), tcp_pos(3)] = 18D
        obs_dim = 18  # Full state: joints + velocities + target + TCP position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _init_state_normalization(self):
        """Initialize state normalization parameters"""
        # Check if state normalization is enabled in config
        norm_config = self.config.get('state_normalization', {})
        self.state_norm_enabled = norm_config.get('enabled', True)

        if self.state_norm_enabled:
            print("🎯 启用状态归一化")

            # State dimension breakdown (18D total)
            # [joint_pos(6), joint_vel(6), target_pos(3), tcp_pos(3)]

            # Joint position normalization ranges (based on joint limits)
            self.joint_pos_low = self.ur10e_joint_limits[:, 0].cpu().numpy()
            self.joint_pos_high = self.ur10e_joint_limits[:, 1].cpu().numpy()

            # Joint velocity normalization ranges (based on velocity limits)
            self.joint_vel_low = -self.ur10e_velocity_limits.cpu().numpy()
            self.joint_vel_high = self.ur10e_velocity_limits.cpu().numpy()

            # Target position normalization ranges (from config or reasonable defaults)
            target_range = norm_config.get('target_position_range', {
                'x': [-1.0, 1.0],
                'y': [-1.0, 1.0],
                'z': [0.0, 1.5]
            })
            self.target_pos_low = np.array([target_range['x'][0], target_range['y'][0], target_range['z'][0]])
            self.target_pos_high = np.array([target_range['x'][1], target_range['y'][1], target_range['z'][1]])

            # TCP position normalization ranges (estimated workspace)
            tcp_range = norm_config.get('tcp_position_range', {
                'x': [-1.2, 1.2],
                'y': [-1.2, 1.2],
                'z': [0.0, 1.8]
            })
            self.tcp_pos_low = np.array([tcp_range['x'][0], tcp_range['y'][0], tcp_range['z'][0]])
            self.tcp_pos_high = np.array([tcp_range['x'][1], tcp_range['y'][1], tcp_range['z'][1]])

            # Concatenated normalization ranges for all 18 dimensions
            self.state_low = np.concatenate([
                self.joint_pos_low,    # 6D joint positions
                self.joint_vel_low,    # 6D joint velocities
                self.target_pos_low,   # 3D target positions
                self.tcp_pos_low       # 3D TCP positions
            ])

            self.state_high = np.concatenate([
                self.joint_pos_high,   # 6D joint positions
                self.joint_vel_high,   # 6D joint velocities
                self.target_pos_high,  # 3D target positions
                self.tcp_pos_high      # 3D TCP positions
            ])

            # Avoid division by zero
            self.state_range = self.state_high - self.state_low
            self.state_range[self.state_range == 0] = 1.0

            print(f"   关节位置范围: [{self.joint_pos_low[:3].round(2)}...], [{self.joint_pos_high[:3].round(2)}...]")
            print(f"   目标位置范围: {target_range}")
            print(f"   TCP位置范围: {tcp_range}")
        else:
            print("📝 状态归一化已禁用")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] range"""
        if not self.state_norm_enabled:
            return state

        # Clip to valid range first
        state_clipped = np.clip(state, self.state_low, self.state_high)

        # Normalize to [-1, 1]
        normalized_state = 2.0 * (state_clipped - self.state_low) / self.state_range - 1.0

        return normalized_state

    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state from [-1, 1] back to original range"""
        if not self.state_norm_enabled:
            return normalized_state

        # Denormalize from [-1, 1]
        state = (normalized_state + 1.0) * self.state_range / 2.0 + self.state_low

        return state

    def _init_reward_normalization(self):
        """Initialize reward normalization parameters"""
        # Check if reward normalization is enabled in config
        reward_norm_config = self.config.get('reward_normalization', {})
        self.reward_norm_enabled = reward_norm_config.get('enabled', True)

        if self.reward_norm_enabled:
            print("🎯 启用奖励归一化")

            # Create reward normalizer for each environment
            gamma = reward_norm_config.get('gamma', 0.99)
            clip_range = reward_norm_config.get('clip_range', 5.0)
            normalize_method = reward_norm_config.get('normalize_method', 'running_stats')
            warmup_steps = reward_norm_config.get('warmup_steps', 100)

            self.reward_normalizers = []
            for i in range(self.num_envs):
                normalizer = RewardNormalizer(
                    gamma=gamma,
                    clip_range=clip_range,
                    normalize_method=normalize_method,
                    warmup_steps=warmup_steps
                )
                self.reward_normalizers.append(normalizer)

            print(f"   归一化方法: {normalize_method}")
            print(f"   折扣因子: {gamma}")
            print(f"   裁剪范围: ±{clip_range}")
            print(f"   预热步数: {warmup_steps}")
        else:
            print("📝 奖励归一化已禁用")
            self.reward_normalizers = [None] * self.num_envs

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found, using defaults")
            config = self._get_default_config()
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'env': {
                'max_steps': 500,
                'dt': 0.01,
                'device_id': 0,
                'num_envs': 1
            },
            'control': {
                'max_increment_torque': 20.0,
                'torque_safety_factor': 0.8
            },
            'safety': {
                'emergency_stop_threshold': 0.5,
                'collision_threshold': 0.1
            },
            'reward': {
                'distance_weight': 2.0,
                'success_reward': 10.0,
                'success_threshold': 0.05,
                'progress_weight': 3.0,
                'stability_weight': 0.3,
                'torque_penalty_weight': 0.01
            },
            'target': {
                'range': {
                    'x': [-0.6, 0.6],
                    'y': [-0.6, 0.6],
                    'z': [0.1, 0.8]
                }
            }
        }

    def _init_pytorch_tensors(self):
        """Initialize PyTorch tensors BEFORE Isaac Gym (following working implementation pattern)"""
        print("🔧 Initializing PyTorch tensors before Isaac Gym...")

        # Action buffer (will be resized after we know DOF count)
        self.actions_buf = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)

        # Observation buffer (18D: joints + velocities + target + TCP)
        self.obs_buf = torch.zeros((self.num_envs, 18), device=self.device, dtype=torch.float32)

        # Reward buffer
        self.rewards_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Done buffer
        self.dones_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # Step count buffer
        self.steps_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Torques buffer (will be created after Isaac Gym initialization to avoid CUDA conflicts)
        self.torques = None  # Will be created in reset() method

        print(f"✅ PyTorch tensors initialized on {self.device}")

    def _init_isaac_gym(self):
        """Initialize Isaac Gym simulator using global gym and sim"""
        # Use global gym and sim instances (only created once per process)
        self.gym, self.sim = get_global_gym_and_sim()

        # 🎬 Graphics and visualization configuration
        viz_config = self.config.get('visualization', {})
        graphics_config = self.config.get('graphics', {})

        self.enable_rendering = viz_config.get('enable', False)

        if self.enable_rendering:
            graphics_device_id = graphics_config.get('graphics_device_id', self.device_id)
            print(f"🎬 启用渲染模式，图形设备: {graphics_device_id}")
        else:
            print("🖥️ 无头模式，禁用渲染")

        # Create environments (this will create multiple envs in the same sim)
        self._create_environments()

        # Setup renderer if visualization is enabled
        if self.enable_rendering:
            self._setup_renderer()

        print("✅ Isaac Gym environment initialized successfully")

    def _create_environments(self):
        """Create parallel environments"""
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # Load UR10e asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True

        # Path to UR10e URDF file (use the working Isaac Gym version)
        asset_root = "../ppo_ur10e_gym"
        asset_file = "ur10e_isaac.urdf"

        ur10e_asset = self.gym.load_asset(
            self.sim,
            asset_root,
            asset_file,
            asset_options
        )

        # Configure DOF properties
        dof_props = self.gym.get_asset_dof_properties(ur10e_asset)
        for i in range(6):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT  # Torque control mode
            dof_props['stiffness'][i] = 0.0
            dof_props['damping'][i] = 0.0
            dof_props['friction'][i] = 0.0  # Remove friction for free movement
            dof_props['armature'][i] = 0.0
            dof_props['hasLimits'][i] = True
            dof_props['lower'][i] = self.ur10e_joint_limits[i, 0]
            dof_props['upper'][i] = self.ur10e_joint_limits[i, 1]

        # Create environment handles
        self.envs = []
        self.ur10e_handles = []

        env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        env_upper = gymapi.Vec3(1.0, 1.0, 1.0)

        for i in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(
                self.sim,
                env_lower,
                env_upper,
                int(self.num_envs**0.5)
            )
            self.envs.append(env)

            # Create UR10e actor
            ur10e_handle = self.gym.create_actor(
                env,
                ur10e_asset,
                gymapi.Transform(),
                "ur10e",
                i,
                1
            )
            self.ur10e_handles.append(ur10e_handle)

            # Set DOF properties
            self.gym.set_actor_dof_properties(env, ur10e_handle, dof_props)

        # Initialize state tensors
        self._init_state_tensors()

        print(f"✅ Created {self.num_envs} parallel environments")

    def _init_state_tensors(self):
        """Initialize state tensors for GPU computation"""
        # Use already imported gymtorch to avoid Foundation object conflicts
        global gymtorch

        # Create tensor viewers
        self.viewer = None

        # Get state tensors (handle potential None case)
        dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        if dof_states is None:
            raise RuntimeError("Failed to acquire DOF state tensor - asset might have no DOFs")

        self.ur10e_states = gymtorch.wrap_tensor(dof_states).view(self.num_envs, -1, 2)  # [num_envs, num_dofs, 2] where 2 = [pos, vel]

        # Force Isaac Gym tensors to target device (following working implementation)
        if self.device.type == 'cuda':
            self.ur10e_states = self.ur10e_states.to(self.device)
            print(f"🔧 Isaac Gym DOF states tensor moved to GPU {self.device.index}")

        # Debug: Check how many DOFs we actually have
        self.num_dofs = self.ur10e_states.shape[1]
        print(f"🔍 UR10e has {self.num_dofs} DOFs")

        # Create torques tensor now that we know the DOF count
        print(f"🔧 DOF count determined: {self.num_dofs} DOFs")
        self.torques = torch.zeros(
            (self.num_envs, self.num_dofs),
            dtype=torch.float32,
            device=self.device
        )
        print(f"✅ Torques tensor created with shape {self.torques.shape}")

        # Current joint positions and velocities
        self.joint_positions = self.ur10e_states[:, :, 0]  # positions
        self.joint_velocities = self.ur10e_states[:, :, 1]  # velocities

        # Initialize target positions
        self._reset_targets()

        # Now set action space based on actual DOF count
        # 🎯 使用归一化动作空间 [-1, 1]，通过 joint_specific_scales 映射到实际力矩
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_dofs,),
            dtype=np.float32
        )

        print("✅ State tensors initialized")
        print(f"🎯 Action space set to {self.num_dofs}D normalized torque control [-1,1]")
        print(f"🎯 Joint-specific scales: {self.joint_specific_scales.cpu().numpy()}")
        print(f"🆕 Momentum decay: {self.torque_momentum_decay}, Velocity penalty: {self.velocity_penalty_strength}")

    def _reset_targets(self):
        """Reset target positions for each environment"""
        target_config = self.config.get('target', {}).get('range', {})

        # Random targets within specified range
        x_range = target_config.get('x', [-0.6, 0.6])
        y_range = target_config.get('y', [-0.6, 0.6])
        z_range = target_config.get('z', [0.1, 0.8])

        self.target_positions = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device
        )

        for i in range(self.num_envs):
            self.target_positions[i, 0] = torch.rand(1) * (x_range[1] - x_range[0]) + x_range[0]
            self.target_positions[i, 1] = torch.rand(1) * (y_range[1] - y_range[0]) + y_range[0]
            self.target_positions[i, 2] = torch.rand(1) * (z_range[1] - z_range[0]) + z_range[0]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment (gymnasium interface)"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Reset joint positions to home pose
        home_positions = torch.tensor([
            0.0,  # shoulder pan
            -np.pi/2,  # shoulder lift
            np.pi/2,   # elbow
            0.0,   # wrist 1
            np.pi/2,   # wrist 2
            0.0    # wrist 3
        ], dtype=torch.float32).repeat(self.num_envs, 1)

        # Set joint positions (unwrap requires CPU tensor)
        if home_positions.device.type != 'cpu':
            home_positions_cpu = home_positions.cpu()
        else:
            home_positions_cpu = home_positions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(home_positions_cpu))

        # Reset targets
        self._reset_targets()

        # Reset step counter
        self.current_step = 0

        # Step simulation to settle (following working implementation)
        for _ in range(10):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # Refresh Isaac Gym state tensors (critical for proper device management)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Update state tensors
        self._update_states()

        # Reset torques to zero (tensor already created in __init__)
        self.torques.zero_()

        # Reset internal state buffers
        self.steps_buf.fill_(0.0)
        self.dones_buf.fill_(False)
        self.rewards_buf.fill_(0.0)

        # 🎯 Reset reward normalizers
        if self.reward_norm_enabled and hasattr(self, 'reward_normalizers'):
            for normalizer in self.reward_normalizers:
                if normalizer is not None:
                    normalizer.reset()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment (gymnasium interface)"""
        if np.isnan(action).any() or np.isinf(action).any():
            print("⚠️ NaN/Inf detected in action, clipping...")
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # Convert action to tensor
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        # Handle different action shapes from DummyVecEnv
        if action_tensor.numel() == 1:
            # Single scalar action, expand to 6D
            action_tensor = action_tensor.repeat(6)
        elif action_tensor.shape[-1] != self.num_dofs:
            # Wrong dimension, try to fix
            action_tensor = action_tensor.view(-1)[:self.num_dofs]
            if action_tensor.numel() < self.num_dofs:
                # Pad with zeros if needed
                action_tensor = torch.cat([
                    action_tensor,
                    torch.zeros(self.num_dofs - action_tensor.numel(), device=self.device)
                ])

        # 🎯 Joint-specific action scaling (normalized action [-1,1] -> actual torque)
        # 基于关节力矩限制的归一化，使[-1,1]动作对应合适的增量力矩
        if action_tensor.shape[-1] == self.num_dofs:
            # 6D action - apply joint-specific scaling
            scaled_action = action_tensor * self.joint_specific_scales * self.max_increment_torque
        else:
            # Fallback for other action dimensions
            scaled_action = torch.clamp(
                action_tensor,
                -self.max_increment_torque,
                self.max_increment_torque
            )

        # 🆕 Momentum inhibition: 防止力矩无限累积，产生更平滑的控制
        # 使用动量衰减而非简单的累加
        momentum_decay = self.torque_momentum_decay
        self.torques = self.torques * momentum_decay + scaled_action

        # 🆕 Velocity-dependent torque inhibition: 高速时自动减少力矩输出
        # 这可以防止高速运动时的振荡和失控
        velocity_penalty = torch.sigmoid(-self.joint_velocities.abs() * self.velocity_penalty_strength)
        self.torques *= velocity_penalty

        # Handle different action/DOF sizes
        if scaled_action.shape[-1] == 6 and self.num_dofs > 6:
            # If action is 6D but we have more DOFs, pad with zeros
            action_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
            action_padded[:, :6] = scaled_action
            self.torques[:, :6] = action_padded[:, :6]
        elif scaled_action.shape[-1] != self.num_dofs:
            raise ValueError(f"Action shape {scaled_action.shape} incompatible with DOFs {self.num_dofs}")

        # Apply safety limits (official UR10e parameters)
        torque_safety_factor = self.config.get('control', {}).get('torque_safety_factor', 0.8)
        velocity_safety_factor = self.config.get('control', {}).get('velocity_safety_factor', 0.8)

        for i in range(min(6, self.num_dofs)):
            # Torque limits (80% of official limits)
            max_torque = self.ur10e_torque_limits[i] * torque_safety_factor
            max_torque_tensor = torch.tensor(max_torque, device=self.torques.device)
            self.torques[:, i] = torch.clamp(self.torques[:, i], -max_torque_tensor, max_torque_tensor)

            # Velocity limits - warn if approaching limits
            max_velocity = self.ur10e_velocity_limits[i] * velocity_safety_factor
            if torch.any(torch.abs(self.joint_velocities[:, i]) > max_velocity):
                # Warning for velocity exceeding safety threshold
                pass  # Could add logging here if needed

        # Apply torques to simulation (unwrap requires CPU tensor)
        if self.torques.device.type != 'cpu':
            torques_cpu = self.torques.cpu()
        else:
            torques_cpu = self.torques

        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques_cpu))

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 🎬 Render if visualization is enabled
        if self.enable_rendering:
            try:
                self.gym.step_graphics(self.sim)
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.gym.draw_viewer(self.viewer, self.sim, True)
            except Exception as e:
                print(f"⚠️ 渲染错误: {e}")

        # Update states
        self._update_states()

        # Get observation, reward, done
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = False  # No early truncation for now
        info = self._get_info()

        # 🛠️ 修复奖励归一化
        if hasattr(self, 'reward_norm_enabled') and self.reward_norm_enabled:
            if hasattr(self, 'reward_normalizers') and self.reward_normalizers is not None:
                # 确保 reward 是标量
                if isinstance(reward, (np.ndarray, torch.Tensor)):
                    if reward.size == 1:
                        reward_scalar = float(reward.item() if isinstance(reward, torch.Tensor) else reward.flat[0])
                    else:
                        # 多环境情况，只处理第一个环境
                        reward_scalar = float(reward[0].item() if isinstance(reward, torch.Tensor) else reward[0])
                else:
                    reward_scalar = float(reward)
                
                # 安全地更新归一化器
                try:
                    self.reward_normalizers[0].update(reward_scalar, bool(terminated))
                except (IndexError, AttributeError) as e:
                    print(f"⚠️ Reward normalizer update failed: {e}")

        self.current_step += 1

         # 检查输出
        if (np.isnan(observation).any() or np.isinf(observation).any() or
            np.isnan(reward).any() or np.isinf(reward).any()):
            print("❌ NaN/Inf in environment output!")
            # 返回安全值
            observation = np.zeros_like(observation)
            reward = -10.0
            terminated = True

        return observation, reward, terminated, truncated, info

    def _update_states(self):
        """Update state tensors from simulation"""
        self.joint_positions = self.ur10e_states[:, :, 0].clone()
        self.joint_velocities = self.ur10e_states[:, :, 1].clone()

    def _get_observation(self) -> np.ndarray:
        """Get observation for Stable-Baselines3"""
        obs_list = []

        for i in range(self.num_envs):
            # Joint positions (6D)
            joint_pos = self.joint_positions[i].cpu().numpy()

            # Joint velocities (6D)
            joint_vel = self.joint_velocities[i].cpu().numpy()

            # Target position (3D)
            target_pos = self.target_positions[i].cpu().numpy()

            # Current TCP position (3D) - approximate using forward kinematics
            tcp_pos = self._forward_kinematics(self.joint_positions[i]).cpu().numpy()

            # Combine: [joint_pos(6), joint_vel(6), target_pos(3), tcp_pos(3)] = 18D
            # Full state with complete position information for better learning
            obs = np.concatenate([
                joint_pos,      # 6D: joint angles
                joint_vel,      # 6D: joint velocities
                target_pos,     # 3D: target position in world frame
                tcp_pos         # 3D: current TCP position (total 18D)
            ])

            # 🎯 Apply state normalization
            obs_normalized = self._normalize_state(obs)
            obs_list.append(obs_normalized)

        # Return first environment's observation (for single env training)
        return obs_list[0] if self.num_envs == 1 else np.array(obs_list)

    def _forward_kinematics(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Simple forward kinematics to get TCP position"""
        # Simplified UR10e forward kinematics
        # This is a basic implementation - in practice you'd use the proper kinematics

        # UR10e DH parameters (simplified)
        a1, a2, a3 = 0.612, 0.572, 0.174  # Link lengths
        d1 = 0.1273
        d4 = 0.1199
        d6 = 0.11655

        q1, q2, q3, q4, q5, q6 = joint_positions

        # Simplified forward kinematics (approximate)
        x = (a1 * torch.cos(q1) + a2 * torch.cos(q1) * torch.cos(q2) +
             a3 * torch.cos(q1) * torch.cos(q2 + q3))
        y = (a1 * torch.sin(q1) + a2 * torch.sin(q1) * torch.cos(q2) +
             a3 * torch.sin(q1) * torch.cos(q2 + q3))
        z = (d1 + a2 * torch.sin(q2) + a3 * torch.sin(q2 + q3) + d4)

        return torch.stack([x, y, z])

    def _calculate_reward(self) -> float:
        """Calculate reward for pure RL control"""
        reward_config = self.config.get('reward', {})

        total_rewards = []

        for i in range(self.num_envs):
            # Get current TCP position
            tcp_pos = self._forward_kinematics(self.joint_positions[i])
            target_pos = self.target_positions[i]

            # Calculate distance to target
            distance = torch.norm(tcp_pos - target_pos)

            # Distance reward (linear penalty)
            distance_reward = -reward_config.get('distance_weight', 2.0) * distance.item()

            # Success reward
            success_threshold = reward_config.get('success_threshold', 0.05)
            success_reward = reward_config.get('success_reward', 10.0) if distance < success_threshold else 0.0

            # Progress reward (if getting closer to target)
            if hasattr(self, 'prev_distance'):
                progress = self.prev_distance[i] - distance.item()
                progress_reward = reward_config.get('progress_weight', 3.0) * max(0, progress)
            else:
                progress_reward = 0.0

            # Stability penalty (large torques)
            torque_magnitude = torch.norm(self.torques[i])
            stability_penalty = -reward_config.get('stability_weight', 0.3) * torque_magnitude.item()

            # Total reward
            total_reward = distance_reward + success_reward + progress_reward + stability_penalty
            total_rewards.append(total_reward)

        # Store previous distances for progress calculation
        if hasattr(self, 'prev_distance'):
            for i in range(self.num_envs):
                tcp_pos = self._forward_kinematics(self.joint_positions[i])
                self.prev_distance[i] = torch.norm(tcp_pos - self.target_positions[i]).item()
        else:
            self.prev_distance = torch.zeros(self.num_envs)
            for i in range(self.num_envs):
                tcp_pos = self._forward_kinematics(self.joint_positions[i])
                self.prev_distance[i] = torch.norm(tcp_pos - self.target_positions[i]).item()

        # Return first environment's reward (for single env training)
        return total_rewards[0] if self.num_envs == 1 else np.array(total_rewards)

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Check step limit
        if self.current_step >= self.max_steps:
            return True

        # Check success (for first environment)
        tcp_pos = self._forward_kinematics(self.joint_positions[0])
        distance = torch.norm(tcp_pos - self.target_positions[0])
        success_threshold = self.config.get('reward', {}).get('success_threshold', 0.05)

        return distance < success_threshold

    def _get_info(self) -> Dict:
        """Get additional information"""
        tcp_pos = self._forward_kinematics(self.joint_positions[0])
        distance = torch.norm(tcp_pos - self.target_positions[0])

        return {
            'episode': {
                'r': self.reward_history[-1] if self.reward_history else 0.0,
                'l': self.current_step
            },
            'distance': distance.item(),
            'tcp_position': tcp_pos.cpu().numpy(),
            'target_position': self.target_positions[0].cpu().numpy(),
            'torques': self.torques[0].cpu().numpy()
        }

    def render(self):
        """Render the environment (optional)"""
        pass  # Isaac Gym rendering handled separately

    def close(self):
        """Close the environment"""
        if hasattr(self, 'sim'):
            self.gym.destroy_sim(self.sim)
        print("✅ Environment closed")

    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        return [seed]

    def get_success_rate(self) -> float:
        """Get recent success rate"""
        return self.success_count / max(1, self.episode_count)

    def print_performance_stats(self):
        """Print performance statistics"""
        if self.reward_history:
            print(f"📊 Performance Stats (Last 100 steps):")
            print(f"   Average Reward: {np.mean(list(self.reward_history)[-100:]):.4f}")
            print(f"   Success Rate: {self.get_success_rate()*100:.1f}%")
            print(f"   Episodes: {self.episode_count}")

            if self.episode_count > 0:
                tcp_pos = self._forward_kinematics(self.joint_positions[0])
                current_distance = torch.norm(tcp_pos - self.target_positions[0]).item()
                print(f"   Current Distance: {current_distance:.4f}m")

    def _setup_renderer(self):
        """Setup Isaac Gym renderer for visualization"""
        try:
            # Create viewer using Isaac Gym
            self.viewer = self.gym.create_viewer(
                self.sim,
                gymapi.CameraProperties()
            )

            if self.viewer is None:
                print("⚠️ 无法创建viewer，使用无头模式")
                self.enable_rendering = False
                return

            # Set camera position
            cam_pos = gymapi.Vec3(2.0, 0.0, 2.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            print(f"✅ 渲染器设置完成，使用Isaac Gym viewer")
            print("   📹 按 ESC 键关闭可视化窗口")

            # Test render with a few simulation steps
            self._test_render_simple()

        except Exception as e:
            print(f"⚠️ 渲染器设置失败: {e}")
            print("   继续使用无头模式")
            self.enable_rendering = False
            self.viewer = None

    def _test_render_simple(self):
        """Test rendering functionality"""
        try:
            # Run a few simulation steps for testing
            for i in range(3):
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.gym.draw_viewer(self.viewer, self.sim, True)

            print("✅ 渲染测试完成")

        except Exception as e:
            print(f"⚠️ 渲染测试失败: {e}")
            self.enable_rendering = False


if __name__ == "__main__":
    # Test the environment
    env = UR10eIncrementalEnv("config.yaml")

    print("\n🧪 Testing environment...")

    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful. Obs shape: {obs.shape}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Step successful. Reward: {reward:.4f}")

    # Test multiple steps
    for step in range(10):
        action = np.random.uniform(-1, 1, 6) * 5  # Small random torques
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 5 == 0:
            print(f"   Step {step}: Reward={reward:.4f}, Distance={info['distance']:.4f}")

        if terminated:
            print(f"   🎯 Episode completed at step {step}!")
            break

    env.print_performance_stats()
    env.close()    