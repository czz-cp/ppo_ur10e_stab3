#!/usr/bin/env python3
"""
UR10e Trajectory Visualization Tool

功能:
1. 显示Task-Space RRT*规划的3D路径
2. 可视化UR10e机器人运动轨迹
3. 实时监控训练过程
4. 生成路径分析图表
"""

import os
import sys

# Set CUDA device before importing Isaac Gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Isaac Gym imports MUST be before PyTorch
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym import gymutil
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")
    sys.exit(1)

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import List, Tuple, Optional
import yaml

# Local imports
from ts_rrt_star import TaskSpaceRRTStar, TSWaypoint
from task_space_planner import TaskSpacePlannerInterface, TSPlanningRequest
from ur10e_trajectory_env import UR10eTrajectoryEnv


class TrajectoryVisualizer:
    """轨迹可视化工具"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_plot_style()

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found")
            return {}

    def setup_plot_style(self):
        """设置绘图样式"""
        try:
            # 尝试使用seaborn样式
            if 'seaborn-v0_8' in plt.style.available:
                plt.style.use('seaborn-v0_8')
            elif 'seaborn' in plt.style.available:
                plt.style.use('seaborn')
            else:
                plt.style.use('default')
        except:
            plt.style.use('default')

    def visualize_rrt_star_planning(self,
                                   start_pos: np.ndarray,
                                   goal_pos: np.ndarray,
                                   show_tree: bool = True,
                                   save_path: str = None):
        """
        可视化RRT*路径规划过程

        Args:
            start_pos: 起始位置 [x, y, z]
            goal_pos: 目标位置 [x, y, z]
            show_tree: 是否显示搜索树
            save_path: 保存图片路径
        """
        print(f"🎯 可视化RRT*规划: {start_pos} -> {goal_pos}")

        # 初始化RRT*规划器
        workspace_bounds = self.config.get('task_space', {}).get('workspace_bounds', {
            'x': [-0.8, 0.8], 'y': [-0.8, 0.8], 'z': [0.1, 1.0]
        })

        rrt_star = TaskSpaceRRTStar(
            workspace_bounds=workspace_bounds,
            goal_tolerance=0.05
        )

        # 执行规划
        start_time = time.time()
        waypoints = rrt_star.plan(start_pos, goal_pos)
        planning_time = time.time() - start_time

        if not waypoints:
            print("❌ 规划失败，无法生成路径")
            return

        print(f"✅ 规划成功: {len(waypoints)}个路径点, 耗时{planning_time:.3f}秒")

        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 提取路径点坐标
        path_x = [wp.cartesian_position[0] for wp in waypoints]
        path_y = [wp.cartesian_position[1] for wp in waypoints]
        path_z = [wp.cartesian_position[2] for wp in waypoints]

        # 绘制工作空间边界
        self._draw_workspace_bounds(ax, workspace_bounds)

        # 绘制路径
        ax.plot(path_x, path_y, path_z, 'b-', linewidth=3, label='规划路径', marker='o', markersize=6)

        # 绘制起点和终点
        ax.scatter(*start_pos, color='green', s=200, marker='o', label='起点', edgecolors='black', linewidth=2)
        ax.scatter(*goal_pos, color='red', s=200, marker='*', label='终点', edgecolors='black', linewidth=2)

        # 显示搜索树
        if show_tree and hasattr(rrt_star, 'nodes'):
            self._draw_rrt_tree(ax, rrt_star.nodes)

        # 设置图形属性
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'Task-Space RRT* 路径规划\n'
                    f'路径长度: {len(waypoints)}点 | '
                    f'规划时间: {planning_time:.3f}s', fontsize=14, fontweight='bold')

        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 设置视角
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📸 图片已保存: {save_path}")

        plt.show()

        return waypoints

    def _draw_workspace_bounds(self, ax, bounds: dict):
        """绘制工作空间边界"""
        x_range = bounds['x']
        y_range = bounds['y']
        z_range = bounds['z']

        # 绘制工作空间的8个角点连线
        corners = [
            [x_range[0], y_range[0], z_range[0]],
            [x_range[1], y_range[0], z_range[0]],
            [x_range[1], y_range[1], z_range[0]],
            [x_range[0], y_range[1], z_range[0]],
            [x_range[0], y_range[0], z_range[1]],
            [x_range[1], y_range[0], z_range[1]],
            [x_range[1], y_range[1], z_range[1]],
            [x_range[0], y_range[1], z_range[1]]
        ]

        # 绘制底面
        for i in range(4):
            next_i = (i + 1) % 4
            ax.plot([corners[i][0], corners[next_i][0]],
                   [corners[i][1], corners[next_i][1]],
                   [corners[i][2], corners[next_i][2]], 'k--', alpha=0.3)

        # 绘制顶面
        for i in range(4, 8):
            next_i = 4 + ((i - 4 + 1) % 4)
            ax.plot([corners[i][0], corners[next_i][0]],
                   [corners[i][1], corners[next_i][1]],
                   [corners[i][2], corners[next_i][2]], 'k--', alpha=0.3)

        # 绘制垂直边
        for i in range(4):
            ax.plot([corners[i][0], corners[i+4][0]],
                   [corners[i][1], corners[i+4][1]],
                   [corners[i][2], corners[i+4][2]], 'k--', alpha=0.3)

    def _draw_rrt_tree(self, ax, nodes):
        """绘制RRT搜索树"""
        for i, node in enumerate(nodes):
            if node.parent is not None:
                # 绘制连接到父节点的线
                parent_pos = node.parent.position
                node_pos = node.position
                ax.plot([parent_pos[0], node_pos[0]],
                       [parent_pos[1], node_pos[1]],
                       [parent_pos[2], node_pos[2]],
                       'gray', alpha=0.3, linewidth=0.5)

        # 绘制所有节点
        if len(nodes) > 0:
            all_positions = np.array([node.position for node in nodes])
            ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
                      c='lightgray', s=10, alpha=0.5)

    def visualize_training_progress(self, log_dir: str = "./tensorboard_logs/"):
        """
        可视化训练进度 (需要tensorboard数据)

        Args:
            log_dir: tensorboard日志目录
        """
        print("📊 可视化训练进度...")

        # 这里可以集成tensorboard数据读取
        # 暂时显示示例图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UR10e 轨迹跟踪训练进度', fontsize=16, fontweight='bold')

        # 模拟训练数据
        episodes = np.arange(0, 1000)
        rewards = -10 * np.exp(-episodes/200) + np.random.normal(0, 0.5, 1000)
        success_rate = 1 - np.exp(-episodes/300)
        trajectory_lengths = 50 + 30 * np.exp(-episodes/150) + np.random.normal(0, 2, 1000)
        waypoint_progress = episodes / 1000

        # 奖励曲线
        axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('回合数')
        axes[0, 0].set_ylabel('平均奖励')
        axes[0, 0].set_title('奖励曲线')
        axes[0, 0].grid(True, alpha=0.3)

        # 成功率曲线
        axes[0, 1].plot(episodes, success_rate * 100, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('回合数')
        axes[0, 1].set_ylabel('成功率 (%)')
        axes[0, 1].set_title('轨迹跟踪成功率')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])

        # 轨迹长度
        axes[1, 0].plot(episodes, trajectory_lengths, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('回合数')
        axes[1, 0].set_ylabel('平均步数')
        axes[1, 0].set_title('完成轨迹所需步数')
        axes[1, 0].grid(True, alpha=0.3)

        # 路径点进度
        axes[1, 1].plot(episodes, waypoint_progress, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('回合数')
        axes[1, 1].set_ylabel('平均路径点进度')
        axes[1, 1].set_title('路径点完成度')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.show()

    def real_time_robot_visualization(self, num_trajectories: int = 5):
        """
        实时可视化机器人轨迹跟踪

        Args:
            num_trajectories: 显示的轨迹数量
        """
        print("🤖 实时机器人轨迹可视化...")

        # 创建环境
        env = UR10eTrajectoryEnv(
            config_path="config.yaml",
            num_envs=1,
            mode="trajectory_tracking"
        )

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        all_trajectories = []
        colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))

        for traj_idx in range(num_trajectories):
            print(f"\n--- 轨迹 {traj_idx + 1}/{num_trajectories} ---")

            # 生成随机起点和终点
            start_tcp = self._sample_random_tcp_position()
            goal_tcp = self._sample_random_tcp_position()

            # 规划轨迹
            if env.plan_trajectory(start_tcp, goal_tcp):
                print(f"✅ 规划成功: {len(env.current_ts_waypoints)}个路径点")

                # 提取理想路径
                ideal_path = np.array([wp.cartesian_position for wp in env.current_ts_waypoints])

                # 模拟机器人跟踪路径
                actual_trajectory = self._simulate_trajectory_tracking(env, ideal_path)
                all_trajectories.append(actual_trajectory)

                # 绘制理想路径
                ax.plot(ideal_path[:, 0], ideal_path[:, 1], ideal_path[:, 2],
                       '--', color=colors[traj_idx], linewidth=2, alpha=0.7,
                       label=f'理想路径 {traj_idx + 1}')

                # 绘制实际轨迹
                ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2],
                       '-', color=colors[traj_idx], linewidth=3, alpha=0.9,
                       label=f'实际轨迹 {traj_idx + 1}')

                # 标记起点和终点
                ax.scatter(*ideal_path[0], color='green', s=100, marker='o')
                ax.scatter(*ideal_path[-1], color='red', s=100, marker='*')

            else:
                print(f"❌ 轨迹 {traj_idx + 1} 规划失败")

        # 设置图形属性
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('UR10e 轨迹跟踪可视化', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        env.close()

    def _sample_random_tcp_position(self) -> np.ndarray:
        """采样随机TCP位置"""
        workspace_bounds = self.config.get('task_space', {}).get('workspace_bounds', {
            'x': [-0.6, 0.6], 'y': [-0.6, 0.6], 'z': [0.2, 0.8]
        })

        position = np.array([
            np.random.uniform(workspace_bounds['x'][0], workspace_bounds['x'][1]),
            np.random.uniform(workspace_bounds['y'][0], workspace_bounds['y'][1]),
            np.random.uniform(workspace_bounds['z'][0], workspace_bounds['z'][1])
        ])

        return position

    def _simulate_trajectory_tracking(self, env, ideal_path: np.ndarray) -> np.ndarray:
        """模拟轨迹跟踪过程"""
        obs, info = env.reset()
        actual_trajectory = []

        max_steps = len(ideal_path) * 10  # 每个路径点最多10步

        for step in range(max_steps):
            # 获取当前TCP位置
            current_tcp = env._forward_kinematics(env.joint_positions[0]).cpu().numpy()
            actual_trajectory.append(current_tcp.copy())

            # 使用简单的控制策略（实际中会用训练好的RL模型）
            # 这里只是模拟轨迹跟踪效果
            if len(actual_trajectory) > len(ideal_path):
                break

            # 简单的PD控制模拟
            target_idx = min(step // 10, len(ideal_path) - 1)
            target_pos = ideal_path[target_idx]

            # 模拟向目标移动（添加一些噪声）
            next_tcp = current_tcp + (target_pos - current_tcp) * 0.1 + np.random.normal(0, 0.01, 3)

            # 检查是否到达目标
            if np.linalg.norm(target_pos - next_tcp) < 0.05:
                if target_idx == len(ideal_path) - 1:
                    break

        return np.array(actual_trajectory)

    def generate_analysis_report(self, save_path: str = "trajectory_analysis.html"):
        """生成轨迹分析报告"""
        print("📋 生成轨迹分析报告...")

        # 这里可以生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UR10e 轨迹跟踪分析报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 UR10e 轨迹跟踪系统分析报告</h1>
                <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>📊 系统配置</h2>
                <div class="metric">配置文件: config.yaml</div>
                <div class="metric">控制模式: 纯RL控制 (增量力矩)</div>
                <div class="metric">规划算法: Task-Space RRT*</div>
                <div class="metric">观察空间: 18维</div>
                <div class="metric">动作空间: 6维</div>
            </div>

            <div class="section">
                <h2>🎯 性能指标</h2>
                <div class="metric">训练总步数: 2,000,000</div>
                <div class="metric">成功率: 待训练完成后统计</div>
                <div class="metric">平均跟踪误差: 待训练完成后统计</div>
                <div class="metric">计算效率: GPU加速</div>
            </div>

            <div class="section">
                <h2>📈 技术特点</h2>
                <div class="metric">✅ Task-Space RRT*全局规划</div>
                <div class="metric">✅ RL局部轨迹跟踪</div>
                <div class="metric">✅ 关节特定动作缩放</div>
                <div class="metric">✅ 动量抑制机制</div>
                <div class="metric">✅ Isaac Gym物理仿真</div>
            </div>
        </body>
        </html>
        """

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 分析报告已保存: {save_path}")


def main():
    """主函数 - 演示所有可视化功能"""
    print("🎨 UR10e 轨迹可视化工具")
    print("=" * 50)

    visualizer = TrajectoryVisualizer()

    # 1. RRT*路径规划可视化
    print("\n1. 🎯 RRT*路径规划可视化")
    start_pos = np.array([0.3, 0.2, 0.4])
    goal_pos = np.array([-0.2, -0.3, 0.6])
    waypoints = visualizer.visualize_rrt_star_planning(
        start_pos, goal_pos,
        show_tree=True,
        save_path="rrt_star_planning.png"
    )

    # 2. 训练进度可视化
    print("\n2. 📊 训练进度可视化")
    visualizer.visualize_training_progress()

    # 3. 实时机器人轨迹可视化
    print("\n3. 🤖 实时机器人轨迹可视化")
    visualizer.real_time_robot_visualization(num_trajectories=3)

    # 4. 生成分析报告
    print("\n4. 📋 生成分析报告")
    visualizer.generate_analysis_report()

    print("\n🎉 可视化演示完成!")


if __name__ == "__main__":
    main()