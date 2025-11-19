#!/usr/bin/env python3
"""
UR10e 可视化演示脚本
快速演示可视化功能，无需完整训练
"""

import os
import sys

# Set CUDA device before importing Isaac Gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Isaac Gym imports MUST be before PyTorch
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")
    # Don't exit for demo only

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Local imports
try:
    from visualization_tool import TrajectoryVisualizer
    from ts_rrt_star import TaskSpaceRRTStar
    print("✅ Local imports successful")
except ImportError as e:
    print(f"⚠️ Local imports failed: {e}")
    print("Running with basic demo only...")


def demo_rrt_star():
    """演示RRT*路径规划"""
    print("\n🎯 演示 1: RRT* 路径规划")
    print("-" * 40)

    # 工作空间边界
    workspace_bounds = {
        'x': [-0.8, 0.8],
        'y': [-0.8, 0.8],
        'z': [0.1, 1.0]
    }

    # 创建RRT*规划器
    rrt_star = TaskSpaceRRTStar(
        workspace_bounds=workspace_bounds,
        goal_tolerance=0.05
    )

    # 生成随机起点和终点
    start_pos = np.array([0.5, 0.3, 0.4])
    goal_pos = np.array([-0.4, -0.2, 0.6])

    print(f"📍 起点: {start_pos}")
    print(f"🎯 终点: {goal_pos}")

    # 执行规划
    start_time = time.time()
    waypoints = rrt_star.plan(start_pos, goal_pos)
    planning_time = time.time() - start_time

    if waypoints:
        print(f"✅ 规划成功!")
        print(f"   路径点数: {len(waypoints)}")
        print(f"   规划时间: {planning_time:.3f}秒")

        # 简单可视化
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 提取路径点坐标
        path_x = [wp.cartesian_position[0] for wp in waypoints]
        path_y = [wp.cartesian_position[1] for wp in waypoints]
        path_z = [wp.cartesian_position[2] for wp in waypoints]

        # 绘制路径
        ax.plot(path_x, path_y, path_z, 'b-', linewidth=3, label='规划路径', marker='o', markersize=6)

        # 标记起点和终点
        ax.scatter(*start_pos, color='green', s=200, marker='o', label='起点', edgecolors='black', linewidth=2)
        ax.scatter(*goal_pos, color='red', s=200, marker='*', label='终点', edgecolors='black', linewidth=2)

        # 设置标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'RRT* 路径规划演示\n{len(waypoints)}个路径点 | {planning_time:.3f}秒')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("demo_rrt_star_path.png", dpi=300, bbox_inches='tight')
        print("📸 路径图片已保存: demo_rrt_star_path.png")
        plt.show()

    else:
        print("❌ 规划失败")


def demo_workspace():
    """演示UR10e工作空间"""
    print("\n🏗️  演示 2: UR10e 工作空间可视化")
    print("-" * 40)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # UR10e工作空间参数
    reach = 1.3  # UR10e最大臂展
    base_height = 0.0

    # 生成工作空间采样点
    print("🎲 生成工作空间采样点...")
    n_samples = 1000
    workspace_points = []

    for _ in range(n_samples):
        # 简化的球形工作空间模型
        r = np.random.uniform(0.2, reach)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + base_height

        # 过滤掉地面以下和过高点
        if 0.1 <= z <= 1.5:
            workspace_points.append([x, y, z])

    workspace_points = np.array(workspace_points)

    # 绘制工作空间点云
    ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2],
              c='lightblue', s=1, alpha=0.3, label='可达工作空间')

    # 绘制机器人基座
    ax.scatter([0], [0], [0], color='black', s=500, marker='s', label='机器人基座')

    # 绘制几个示例轨迹
    print("🛤️  生成示例轨迹...")
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(5):
        # 随机起点和终点
        start = workspace_points[np.random.randint(len(workspace_points))]
        end = workspace_points[np.random.randint(len(workspace_points))]

        # 生成简单的曲线路径
        t = np.linspace(0, 1, 20)
        mid = (start + end) / 2 + np.random.normal(0, 0.1, 3)

        # 二次贝塞尔曲线
        path = (1-t)**2[:, np.newaxis] * start + \
               2*(1-t)[:, np.newaxis] * t[:, np.newaxis] * mid + \
               t**2[:, np.newaxis] * end

        ax.plot(path[:, 0], path[:, 1], path[:, 2],
               color=colors[i], linewidth=2, alpha=0.7,
               label=f'示例轨迹 {i+1}')

    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UR10e 工作空间和示例轨迹')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 设置视角
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("demo_workspace.png", dpi=300, bbox_inches='tight')
    print("📸 工作空间图片已保存: demo_workspace.png")
    plt.show()


def demo_trajectory_tracking():
    """演示轨迹跟踪概念"""
    print("\n🤖 演示 3: 轨迹跟踪概念")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UR10e 轨迹跟踪概念演示', fontsize=16, fontweight='bold')

    # 1. 理想轨迹 vs 实际轨迹
    ax1 = axes[0, 0]
    t = np.linspace(0, 10, 100)
    ideal_x = np.sin(t)
    ideal_y = np.cos(t) * 0.5

    # 模拟有噪声的实际轨迹
    actual_x = ideal_x + np.random.normal(0, 0.05, len(t))
    actual_y = ideal_y + np.random.normal(0, 0.05, len(t))

    ax1.plot(ideal_x, ideal_y, 'b--', linewidth=2, label='理想轨迹')
    ax1.plot(actual_x, actual_y, 'r-', linewidth=1.5, alpha=0.7, label='实际轨迹')
    ax1.scatter(ideal_x[0], ideal_y[0], color='green', s=100, marker='o', label='起点', zorder=5)
    ax1.scatter(ideal_x[-1], ideal_y[-1], color='red', s=100, marker='*', label='终点', zorder=5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('轨迹跟踪效果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. 跟踪误差随时间变化
    ax2 = axes[0, 1]
    tracking_error = np.sqrt((actual_x - ideal_x)**2 + (actual_y - ideal_y)**2)
    ax2.plot(t, tracking_error, 'r-', linewidth=2)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('跟踪误差 (m)')
    ax2.set_title('跟踪误差随时间变化')
    ax2.grid(True, alpha=0.3)

    # 3. 控制输入 (力矩)
    ax3 = axes[1, 0]
    # 模拟6关节力矩
    torque_commands = np.random.normal(0, 5, (6, len(t)))
    joint_labels = ['关节1', '关节2', '关节3', '关节4', '关节5', '关节6']

    for i in range(6):
        ax3.plot(t, torque_commands[i], linewidth=1.5, label=joint_labels[i], alpha=0.7)

    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('力矩 (N·m)')
    ax3.set_title('关节力矩控制信号')
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. 奖励函数
    ax4 = axes[1, 1]
    # 模拟奖励变化
    rewards = -tracking_error * 10 + np.random.normal(0, 0.5, len(t))
    cumulative_reward = np.cumsum(rewards)

    ax4.plot(t, rewards, 'g-', linewidth=1, alpha=0.7, label='即时奖励')
    ax4.plot(t, cumulative_reward/np.max(np.abs(cumulative_reward)) * np.max(rewards),
             'b-', linewidth=2, label='累积奖励')

    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('奖励')
    ax4.set_title('奖励函数变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("demo_trajectory_tracking.png", dpi=300, bbox_inches='tight')
    print("📸 轨迹跟踪演示图片已保存: demo_trajectory_tracking.png")
    plt.show()


def main():
    """主演示函数"""
    print("🎨 UR10e 可视化演示程序")
    print("=" * 50)
    print("📋 演示内容:")
    print("   1. RRT* 路径规划")
    print("   2. UR10e 工作空间")
    print("   3. 轨迹跟踪概念")
    print("=" * 50)

    try:
        # 演示1: RRT*规划
        demo_rrt_star()

        # 演示2: 工作空间
        demo_workspace()

        # 演示3: 轨迹跟踪
        demo_trajectory_tracking()

        print("\n🎉 所有演示完成!")
        print("📸 生成的图片:")
        print("   - demo_rrt_star_path.png")
        print("   - demo_workspace.png")
        print("   - demo_trajectory_tracking.png")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()