#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
UR10e Trajectory Tracking Training with Stable-Baselines3 (PPO)

- Isaac Gym 负责底层多环境并行 (num_envs in config.yaml)
- 我们用自定义 VecEnv (IsaacGymTrajectoryVecEnv) 把 UR10eTrajectoryEnv
  包装成 Stable-Baselines3 能直接吃的 VecEnv 接口。

训练逻辑：
1. 随机采样 TCP 起点/终点
2. Task-Space RRT* 规划路径 → 一系列 waypoint
3. RL (PPO) 输出 6 维关节力矩增量动作，跟踪 waypoint
"""

import os
import sys
import time
from typing import Dict, Optional
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# Isaac Gym 必须在 torch 之前导入
# ───────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym import gymutil
    from isaacgym.torch_utils import *
    print("✅ All Isaac Gym modules imported successfully in train_trajectory.py")
    print("   - gymapi: Isaac Gym API")
    print("   - gymtorch: PyTorch bindings")
    print("   - gymutil: Utilities")
    print("   - torch_utils: Torch utilities")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym in train_trajectory.py: {e}")
    print("Please ensure Isaac Gym is properly installed")
    sys.exit(1)

import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
print("✅ Using gymnasium (recommended)")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure

# 你的环境
from ur10e_trajectory_env import UR10eTrajectoryEnv


# ───────────────────────────────────────────────────────────────────────────────
# 通用工具
# ───────────────────────────────────────────────────────────────────────────────
def _setup_environment_variables(config: Dict):
    """配置 GPU / 内存 环境变量"""

    device_id = config.get('env', {}).get('device_id', 0)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        print(f"🖥️  Using existing CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["ISAACGYM_GPU_CACHE_SIZE"] = "1073741824"  # 1GB

    print("🖥️  Environment setup:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")


def _load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file {config_path} not found")
        sys.exit(1)


def _sample_random_tcp_position() -> np.ndarray:
    """工作空间内随机采样一个 TCP 位置"""
    workspace_bounds = {
        'x': [-0.6, 0.6],
        'y': [-0.6, 0.6],
        'z': [0.2, 0.8]
    }
    position = np.array([
        np.random.uniform(*workspace_bounds['x']),
        np.random.uniform(*workspace_bounds['y']),
        np.random.uniform(*workspace_bounds['z'])
    ])
    return position


def _setup_logging(config: Dict, run_name: str = None):
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"trajectory_training_{timestamp}"

    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_log = f"./tensorboard_logs/{run_name}"
    os.makedirs(tensorboard_log, exist_ok=True)

    print("📊 Logging setup:")
    print(f"   Log directory: {log_dir}")
    print(f"   Tensorboard: {tensorboard_log}")

    return log_dir, tensorboard_log


# ───────────────────────────────────────────────────────────────────────────────
# 自定义 VecEnv：把 UR10eTrajectoryEnv 包装成 SB3 VecEnv
# ───────────────────────────────────────────────────────────────────────────────
def create_isaac_gym_vec_env(config_path: str, num_envs: int):
    """
    Create a proper VecEnv using the restored IsaacGymTrajectoryVecEnv
    that correctly inherits from VecEnv base class.
    """
    print(f"🚀 Creating Isaac Gym VecEnv with {num_envs} parallel environments...")
    return IsaacGymTrajectoryVecEnv(config_path=config_path, num_envs=num_envs)

class IsaacGymTrajectoryVecEnv(VecEnv):
    """
    VecEnv implementation compatible with older SB3 versions
    """

    def __init__(self, config_path: str, num_envs: int):
        # Create the underlying Isaac Gym environment
        self.underlying_env = UR10eTrajectoryEnv(
            config_path=config_path,
            num_envs=num_envs,
            mode="trajectory_tracking"
        )

        print("✅ UR10eTrajectoryEnv created successfully")

        # Get spaces from the environment
        obs, info = self.underlying_env.reset()
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        obs_dim = obs_np.shape[1]
        print(f"📐 Observation dimension: {obs_dim}")

        # Create spaces
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        if hasattr(self.underlying_env, "action_space"):
            action_space = self.underlying_env.action_space
        else:
            action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32
            )

        # Initialize buffers
        self._observations = np.zeros((num_envs, obs_dim), dtype=np.float32)
        self._actions = np.zeros((num_envs, action_space.shape[0]), dtype=np.float32)

        # 🎯 OLD SB3 VERSION COMPATIBILITY - No metadata/render_mode
        try:
            # Try old version signature first
            VecEnv.__init__(
                self,
                num_envs=num_envs,
                observation_space=observation_space,
                action_space=action_space
            )
            print("✅ VecEnv.__init__ (old version) succeeded!")
        except TypeError as e:
            # If that fails, try with minimal new parameters
            print("🔄 Trying alternative VecEnv initialization...")
            try:
                VecEnv.__init__(
                    self,
                    num_envs=num_envs,
                    observation_space=observation_space,
                    action_space=action_space,
                    metadata=getattr(self, 'metadata', {'render_modes': []})
                )
            except Exception as e2:
                print(f"❌ All VecEnv initialization attempts failed: {e2}")
                raise

        print(f"✅ IsaacGymTrajectoryVecEnv initialized with {self.num_envs} environments")

    def reset(self):
        obs, info = self.underlying_env.reset()
        self._observations = self._to_numpy(obs)
        
        # Ensure correct shape
        if self._observations.shape != (self.num_envs, self.observation_space.shape[0]):
            if self._observations.size == self.num_envs * self.observation_space.shape[0]:
                self._observations = self._observations.reshape(self.num_envs, self.observation_space.shape[0])
        
        return self._observations.copy()

    def step_async(self, actions):
        self._actions = self._to_numpy(actions)

    def step_wait(self):
        obs, reward, terminated, truncated, info = self.underlying_env.step(self._actions)

       
        
        # Convert to numpy
        obs_np = self._to_numpy(obs)
        rew_np = self._to_numpy(reward).flatten()
        term_np = np.asarray(terminated, dtype=bool).flatten()
        trunc_np = np.asarray(truncated, dtype=bool).flatten()
        dones = np.logical_or(term_np, trunc_np)

        # Handle info
        if isinstance(info, dict):
            infos = [info.copy() for _ in range(self.num_envs)]
        else:
            infos = info if info is not None else [{}] * self.num_envs
        
          # ⭐ Debug：打印 reward 和 distance_to_waypoint
        if not hasattr(self, "_debug_step"):
            self._debug_step = 0
        self._debug_step += 1

        if self._debug_step % 200 == 0:
            

            # 奖励统计
            r_mean = float(np.mean(rew_np))
            r_max = float(np.max(rew_np))
            r_min = float(np.min(rew_np))

            # 距离统计（有可能 info 里暂时没这个 key）
            dists = []
            for inf in infos:
                if isinstance(inf, dict) and "distance_to_waypoint" in inf:
                    dists.append(inf["distance_to_waypoint"])

            if dists:
                dists = np.array(dists, dtype=np.float32)
                d_mean = float(dists.mean())
                d_max = float(dists.max())
                d_min = float(dists.min())
                print(
                    f"[VecEnv step {self._debug_step}] "
                    f"reward mean = {r_mean:.3f}, max = {r_max:.3f}, min = {r_min:.3f} | "
                    f"dist_to_wp mean = {d_mean:.3f}, max = {d_max:.3f}, min = {d_min:.3f}"
                )
            else:
                print(
                    f"[VecEnv step {self._debug_step}] "
                    f"reward mean = {r_mean:.3f}, max = {r_max:.3f}, min = {r_min:.3f} | "
                    f"dist_to_wp: (no key in info)"
                )

        self._observations = obs_np
        return obs_np, rew_np, dones, infos

    def close(self):
        self.underlying_env.close()

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        value = getattr(self.underlying_env, attr_name, None)
        return [value] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        if hasattr(self.underlying_env, attr_name):
            setattr(self.underlying_env, attr_name, value)
            return [True] * len(indices)
        return [False] * len(indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = range(self.num_envs)
        if hasattr(self.underlying_env, method_name):
            result = getattr(self.underlying_env, method_name)(*method_args, **method_kwargs)
            return [result] * len(indices)
        return [None] * len(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [False] * len(indices)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)


# ───────────────────────────────────────────────────────────────────────────────
# PPO 创建
# ───────────────────────────────────────────────────────────────────────────────
def _create_ppo_model(config: Dict, env: VecEnv, device=None) -> PPO:
    ppo_config = config.get('ppo', {})

    num_envs = env.num_envs
    base_batch_size = ppo_config.get('batch_size', 64)
    base_n_steps = ppo_config.get('n_steps', 2048)

    # batch_size 至少是 num_envs 的若干倍，且尽量整除 n_steps
    adjusted_batch_size = max(base_batch_size, num_envs * 4)
    adjusted_n_steps = base_n_steps
    if adjusted_n_steps % adjusted_batch_size != 0:
        for bs in range(adjusted_batch_size, adjusted_n_steps + 1, num_envs):
            if adjusted_n_steps % bs == 0:
                adjusted_batch_size = bs
                break

    print(f"🔧 PPO参数调整 (num_envs={num_envs}):")
    print(f"   batch_size: {base_batch_size} -> {adjusted_batch_size}")
    print(f"   n_steps: {base_n_steps}")

    policy_kwargs = ppo_config.get('policy_kwargs', {
        'net_arch': [512, 256, 128],
        'activation_fn': 'relu',
        'ortho_init': True,  # 🆕 正交初始化
        'log_std_init': -0.5  # 🆕 初始化较小的动作方差
    }).copy()

    if isinstance(policy_kwargs.get('activation_fn'), str):
        act_str = policy_kwargs['activation_fn'].lower()
        if act_str == 'relu':
            policy_kwargs['activation_fn'] = nn.ReLU
        elif act_str == 'tanh':
            policy_kwargs['activation_fn'] = nn.Tanh
        elif act_str == 'sigmoid':
            policy_kwargs['activation_fn'] = nn.Sigmoid
        else:
            print(f"⚠️ Unknown activation function: {act_str}, using ReLU")
            policy_kwargs['activation_fn'] = nn.ReLU

    model = PPO(
        policy=ppo_config.get('policy', "MlpPolicy"),
        env=env,
        learning_rate=ppo_config.get('learning_rate', 1.0e-4),
        n_steps=adjusted_n_steps,
        batch_size=adjusted_batch_size,
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        clip_range_vf=ppo_config.get('clip_range_vf', None),
        normalize_advantage=ppo_config.get('normalize_advantage', True),
        ent_coef=ppo_config.get('ent_coef', 0.02),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        target_kl=ppo_config.get('target_kl', 0.02),
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        device=str(device) if device is not None else "auto"
    )

    print("🧠 PPO model created:")
    print(f"   Policy: {ppo_config.get('policy', 'MlpPolicy')}")
    print(f"   Network architecture: {policy_kwargs.get('net_arch', [512, 256, 128])}")
    print(f"   Activation function: {policy_kwargs['activation_fn'].__name__}")
    print(f"   Learning rate: {ppo_config.get('learning_rate', 3.0e-4)}")

    return model


# ───────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ───────────────────────────────────────────────────────────────────────────────
def train_trajectory_tracker(config_path: str = "config.yaml"):
    print("🎯 UR10e Trajectory Tracking Training")
    print("=" * 60)

    config = _load_config(config_path)
    _setup_environment_variables(config)
    log_dir, tensorboard_log = _setup_logging(config)

    num_envs = config.get('env', {}).get('num_envs', 2)
    print(f"🚀 Creating training VecEnv (num_envs = {num_envs}) ...")

    # 使用新的包装器避免 VecEnv 兼容性问题
    train_env = create_isaac_gym_vec_env(config_path=config_path, num_envs=num_envs)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = _create_ppo_model(config, train_env, device)

    model_save_path = f"{log_dir}/trajectory_model"
    total_timesteps = config.get('ppo', {}).get('total_timesteps', 1000)

    print("🏋️  Starting training:")
    print(f"   Total timesteps: {total_timesteps:,}")

    class ProgressCallback(BaseCallback):
        def __init__(self, verbose=1):
            super().__init__(verbose)
            self.last_log_time = time.time()
            self.last_log_steps = 0

        def _on_step(self):
            current_time = time.time()
            if (current_time - self.last_log_time > 5.0 or
                    self.num_timesteps - self.last_log_steps >= 1000):

                elapsed = current_time - self.last_log_time
                steps_done = self.num_timesteps - self.last_log_steps
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0

                progress = (self.num_timesteps / total_timesteps) * 100.0
                eta = (total_timesteps - self.num_timesteps) / steps_per_sec if steps_per_sec > 0 else float('inf')

                print(f"📈 Progress: {progress:.1f}% | Steps: {self.num_timesteps:,}/{total_timesteps:,} | "
                      f"Speed: {steps_per_sec:.1f} steps/s | ETA: {eta/60:.1f} min")

                # 添加更详细的训练统计信息
                if hasattr(self.model, 'env') and hasattr(self.model.env, 'underlying_env'):
                    # 获取环境中的信息
                    env = self.model.env.underlying_env
                    if hasattr(env, 'current_waypoint_index') and hasattr(env, 'current_ts_waypoints'):
                        if len(env.current_ts_waypoints) > 0:
                            current_wp = env.current_waypoint_index + 1
                            total_wp = len(env.current_ts_waypoints)
                            print(f"📍 Waypoint Progress: {current_wp}/{total_wp}")
                            
                            # 计算当前TCP到路径点的距离
                            if hasattr(env, 'joint_positions') and len(env.joint_positions) > 0:
                                tcp_pos = env._forward_kinematics(env.joint_positions[0])
                                current_waypoint = env.get_current_waypoint()
                                if current_waypoint is not None:
                                    waypoint_pos = torch.tensor(current_waypoint.cartesian_position, 
                                                              device=tcp_pos.device, dtype=tcp_pos.dtype)
                                    distance = torch.norm(tcp_pos - waypoint_pos).item()
                                    print(f"📏 Distance to Current Waypoint: {distance:.4f}m")
                                    
                                    # 显示TCP位置和路径点位置
                                    print(f"🤖 Current TCP Position: [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
                                    print(f"🚩 Current Waypoint Position: [{waypoint_pos[0]:.4f}, {waypoint_pos[1]:.4f}, {waypoint_pos[2]:.4f}]")

                self.last_log_time = current_time
                self.last_log_steps = self.num_timesteps
            return True

    progress_callback = ProgressCallback()

    try:
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            log_interval=10,
            tb_log_name="trajectory_tracking",
            progress_bar=True,
            callback=[progress_callback]
        )

        training_time = time.time() - start_time
        print("🎉 Training completed successfully!")
        print(f"   Training time: {training_time:.2f} seconds")

        model.save(f"{model_save_path}_final")
        print(f"💾 Final model saved to {model_save_path}_final")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        model.save(f"{model_save_path}_interrupted")
        print(f"💾 Checkpoint saved to {model_save_path}_interrupted")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        train_env.close()
        print("🏁 Training session ended")


# ───────────────────────────────────────────────────────────────────────────────
# 测试函数
# ───────────────────────────────────────────────────────────────────────────────
def test_trajectory_model(model_path: str, config_path: str = "config.yaml"):
    print(f"🧪 Testing trajectory model: {model_path}")

    config = _load_config(config_path)
    _setup_environment_variables(config)

    # 测试时只用 1 个 Isaac 环境即可
    test_vec_env = create_isaac_gym_vec_env(config_path=config_path, num_envs=1)
    model = PPO.load(model_path, env=test_vec_env)
    print("✅ Model loaded successfully")

    num_tests = 10
    successful_trajectories = 0

    # 直接访问底层 env 以调用 plan_trajectory / get_trajectory_statistics
    env = test_vec_env.underlying_env

    for t in range(num_tests):
        print(f"\n--- Test {t + 1}/{num_tests} ---")

        start_tcp = _sample_random_tcp_position()
        goal_tcp = _sample_random_tcp_position()
        print(f"Start TCP: {start_tcp}")
        print(f"Goal TCP: {goal_tcp}")

        if env.plan_trajectory(start_tcp, goal_tcp):
            print(f"✅ Trajectory planned: {len(env.current_ts_waypoints)} waypoints")

            obs, info = env.reset()
            done = False
            steps = 0
            total_reward = 0.0
            max_steps = 500

            while not done and steps < max_steps:
                # obs: (1, obs_dim)
                if isinstance(obs, torch.Tensor):
                    obs_np = obs.detach().cpu().numpy()
                else:
                    obs_np = np.asarray(obs, dtype=np.float32)

                action, _ = model.predict(obs_np, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                done = bool(terminated or truncated)
                steps += 1

                if steps % 50 == 0:
                    print(f"  Step {steps}: reward={total_reward:.2f}, "
                          f"waypoint={info.get('current_waypoint', 0)}")

            stats = env.get_trajectory_statistics()
            if stats['trajectory_completed']:
                successful_trajectories += 1
                print(f"🎉 Trajectory {t + 1} completed successfully!")
            else:
                print(f"❌ Trajectory {t + 1} failed")

            print(f"   Final stats: {stats}")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Steps taken: {steps}")
        else:
            print(f"❌ Failed to plan trajectory {t + 1}")

    print("\n📊 Test Summary:")
    print(f"   Successful trajectories: {successful_trajectories}/{num_tests}")
    print(f"   Success rate: {successful_trajectories / num_tests * 100:.1f}%")

    test_vec_env.close()


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e Trajectory Tracking Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Training or testing mode")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (for testing)")

    args = parser.parse_args()

    if args.mode == "train":
        train_trajectory_tracker(args.config)
    elif args.mode == "test":
        if args.model is None:
            print("❌ Please provide model path for testing using --model")
            sys.exit(1)
        test_trajectory_model(args.model, args.config)
    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)
