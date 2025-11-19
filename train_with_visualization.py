#!/usr/bin/env python3
"""
UR10e Trajectory Tracking Training with Real-time Visualization
带实时可视化的训练脚本
"""

import os
import sys

# Set CUDA device before importing Isaac Gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Isaac Gym imports MUST be before any PyTorch imports
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym import gymutil
    from isaacgym.torch_utils import *
    print("✅ All Isaac Gym modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
import yaml
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Local imports
from ur10e_trajectory_env import UR10eTrajectoryEnv
from visualization_tool import TrajectoryVisualizer


class VisualizationCallback(BaseCallback):
    """带可视化的训练回调函数"""

    def __init__(self, visualizer, eval_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.visualizer = visualizer
        self.eval_freq = eval_freq
        self.training_trajectories = []
        self.start_time = time.time()
        self.last_viz_time = self.start_time
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        current_time = time.time()

        # 每10秒显示一次进度
        if current_time - self.last_viz_time > 10.0:
            elapsed = current_time - self.start_time
            progress = (self.num_timesteps / self.training_total_timesteps) * 100

            print(f"🎨 训练可视化更新:")
            print(f"   📊 进度: {progress:.1f}% | 步数: {self.num_timesteps:,}")
            print(f"   ⏱️  已训练: {elapsed/60:.1f}分钟")
            print(f"   🔥 平均奖励: {getattr(self, 'current_mean_reward', 'N/A')}")

            self.last_viz_time = current_time

        # 定期评估和可视化
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate_and_visualize()

        return True

    def _evaluate_and_visualize(self):
        """评估当前策略并可视化"""
        print(f"\n🎯 评估周期 {self.n_calls} - 开始可视化...")

        # 生成测试轨迹
        num_test_trajectories = 3
        test_results = []

        for i in range(num_test_trajectories):
            # 生成随机起点和终点
            start_tcp = self._sample_random_position()
            goal_tcp = self._sample_random_position()

            # 使用RRT*规划路径
            try:
                waypoints = self.visualizer.visualize_rrt_star_planning(
                    start_tcp, goal_tcp,
                    show_tree=False,
                    save_path=f"eval_trajectory_{self.n_calls}_{i+1}.png"
                )
                test_results.append({
                    'start': start_tcp,
                    'goal': goal_tcp,
                    'waypoints': waypoints,
                    'success': waypoints is not None
                })
            except Exception as e:
                print(f"   ❌ 轨迹 {i+1} 规划失败: {e}")
                test_results.append({
                    'start': start_tcp,
                    'goal': goal_tcp,
                    'waypoints': None,
                    'success': False
                })

        # 统计成功率
        success_count = sum(1 for r in test_results if r['success'])
        success_rate = success_count / len(test_results) * 100

        print(f"📈 评估结果:")
        print(f"   ✅ 成功轨迹: {success_count}/{len(test_results)} ({success_rate:.1f}%)")
        print(f"   📸 生成了 {len(test_results)} 个可视化图片")

        # 记录到tensorboard
        if hasattr(self, 'logger'):
            self.logger.record("eval/trajectory_planning_success_rate", success_rate)
            self.logger.record("eval/test_trajectories", len(test_results))

    def _sample_random_position(self) -> np.ndarray:
        """采样随机位置"""
        # UR10e工作空间边界
        return np.array([
            np.random.uniform(-0.5, 0.5),  # x
            np.random.uniform(-0.5, 0.5),  # y
            np.random.uniform(0.2, 0.7)    # z
        ])


def create_visualization_env(config_path: str, num_envs: int = 1):
    """创建带可视化的环境"""
    def _init():
        # 启用可视化
        config = _load_config(config_path)

        # 临时启用可视化
        original_vis = config.get('visualization', {}).get('enable', False)
        config['visualization']['enable'] = True

        env = UR10eTrajectoryEnv(
            config_path=config_path,
            num_envs=num_envs,
            mode="trajectory_tracking"
        )

        # 恢复原始设置
        if not original_vis:
            config['visualization']['enable'] = False

        env = Monitor(env, filename="./trajectory_monitor_logs/")
        return env
    return _init


def _load_config(config_path: str) -> Dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file {config_path} not found")
        return {}


def train_with_visualization(config_path: str = "config.yaml"):
    """带可视化的主训练函数"""
    print("🎨 UR10e 轨迹跟踪训练 - 带实时可视化")
    print("=" * 60)

    # 加载配置
    config = _load_config(config_path)

    # 创建可视化工具
    visualizer = TrajectoryVisualizer(config_path)

    print(f"\n🎬 初始化可视化系统...")

    # 首先演示RRT*规划可视化
    print("\n1. 🎯 演示RRT*路径规划可视化...")
    start_pos = np.array([0.4, 0.3, 0.5])
    goal_pos = np.array([-0.3, -0.2, 0.6])

    waypoints = visualizer.visualize_rrt_star_planning(
        start_pos, goal_pos,
        show_tree=True,
        save_path="demo_rrt_star_planning.png"
    )

    # 获取环境数量配置
    num_envs = config.get('env', {}).get('num_envs', 1)
    print(f"\n🚀 创建 {num_envs} 个训练环境...")

    # 创建训练环境 (可视化模式下通常使用单环境)
    env_fn = create_visualization_env(config_path, num_envs)
    train_env = DummyVecEnv([env_fn])  # 为了可视化稳定性，使用单环境

    # 创建PPO模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ppo_config = config.get('ppo', {})

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=ppo_config.get('learning_rate', 3.0e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.995),
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        device=str(device)
    )

    print(f"🧠 PPO模型已创建，设备: {device}")

    # 创建可视化回调
    viz_callback = VisualizationCallback(
        visualizer=visualizer,
        eval_freq=10000,  # 每10K步评估一次
        verbose=1
    )
    viz_callback.training_total_timesteps = config.get('ppo', {}).get('total_timesteps', 1000000)

    # 获取训练参数
    total_timesteps = config.get('ppo', {}).get('total_timesteps', 1000000)

    print(f"\n🏋️  开始可视化训练:")
    print(f"   总步数: {total_timesteps:,}")
    print(f"   可视化频率: 每10,000步")
    print(f"   评估轨迹数: 3个/次")

    try:
        # 测试环境
        print("🧪 测试环境...")
        obs = train_env.reset()
        action = train_env.action_space.sample()
        obs, reward, done, info = train_env.step(action)
        print(f"   环境测试成功，奖励: {reward:.4f}")

        print("✅ 环境测试通过，开始训练...")

        # 开始训练
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            log_interval=100,
            tb_log_name="trajectory_tracking_with_viz",
            callback=[viz_callback],
            progress_bar=True
        )

        training_time = time.time() - start_time

        print(f"\n🎉 训练完成!")
        print(f"   总训练时间: {training_time/3600:.2f} 小时")
        print(f"   平均速度: {total_timesteps/training_time:.1f} 步/秒")

        # 最终可视化
        print(f"\n🎨 生成最终轨迹可视化...")
        visualizer.real_time_robot_visualization(num_trajectories=5)

        # 生成分析报告
        visualizer.generate_analysis_report("final_analysis_report.html")

        # 保存模型
        model_save_path = f"trajectory_model_with_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(model_save_path)
        print(f"💾 模型已保存: {model_save_path}")

    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        train_env.close()
        print("🏁 训练会话结束")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UR10e 带可视化的轨迹跟踪训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--demo-only", action="store_true", help="只运行可视化演示，不训练")

    args = parser.parse_args()

    if args.demo_only:
        # 只运行可视化演示
        print("🎨 只运行可视化演示模式...")
        visualizer = TrajectoryVisualizer(args.config)

        # 演示RRT*规划
        start_pos = np.array([0.4, 0.3, 0.5])
        goal_pos = np.array([-0.3, -0.2, 0.6])
        visualizer.visualize_rrt_star_planning(start_pos, goal_pos, show_tree=True)

        # 演示机器人轨迹
        visualizer.real_time_robot_visualization(num_trajectories=3)

        print("✅ 可视化演示完成!")
    else:
        # 完整训练
        train_with_visualization(args.config)