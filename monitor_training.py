#!/usr/bin/env python3
"""
训练实时监控工具
实时监控训练进度和可视化轨迹
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import threading
import yaml
from datetime import datetime

# Set CUDA device before importing Isaac Gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Isaac Gym imports MUST be before PyTorch
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")

import numpy as np
from visualization_tool import TrajectoryVisualizer


class TrainingMonitor:
    """训练实时监控器"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.visualizer = TrajectoryVisualizer(config_path)
        self.is_running = False

        # 监控数据
        self.training_data = {
            'timestamps': [],
            'steps': [],
            'rewards': [],
            'success_rates': [],
            'trajectories_planned': 0,
            'trajectories_successful': 0
        }

        # 设置matplotlib
        plt.ion()
        self.setup_plots()

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def setup_plots(self):
        """设置监控图表"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('UR10e 轨迹跟踪训练监控', fontsize=16, fontweight='bold')

        # 奖励曲线
        self.reward_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        self.axes[0, 0].set_title('平均奖励')
        self.axes[0, 0].set_xlabel('训练步数')
        self.axes[0, 0].set_ylabel('奖励')
        self.axes[0, 0].grid(True, alpha=0.3)

        # 成功率曲线
        self.success_line, = self.axes[0, 1].plot([], [], 'g-', linewidth=2)
        self.axes[0, 1].set_title('轨迹成功率')
        self.axes[0, 1].set_xlabel('训练步数')
        self.axes[0, 1].set_ylabel('成功率 (%)')
        self.axes[0, 1].set_ylim([0, 100])
        self.axes[0, 1].grid(True, alpha=0.3)

        # 3D轨迹可视化
        self.ax_3d = self.fig.add_subplot(223, projection='3d')
        self.ax_3d.set_title('实时轨迹可视化')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        # 统计信息
        self.axes[1, 1].axis('off')
        self.stats_text = self.axes[1, 1].text(0.1, 0.5, '', fontsize=12, family='monospace')

        plt.tight_layout()

    def update_plots(self, frame=None):
        """更新监控图表"""
        if not self.is_running:
            return

        # 模拟训练数据更新 (实际中会从训练进程获取)
        current_time = time.time()
        if len(self.training_data['timestamps']) == 0:
            last_time = current_time
        else:
            last_time = self.training_data['timestamps'][-1]

        # 每2秒更新一次
        if current_time - last_time < 2.0:
            return

        # 生成新的训练点 (模拟数据)
        step = len(self.training_data['steps']) * 1000
        if step > 0:
            # 模拟奖励改善
            reward = -5 * np.exp(-step/100000) + np.random.normal(0, 0.5)
            success_rate = min(100, (1 - np.exp(-step/150000)) * 100 + np.random.normal(0, 5))

            # 添加数据
            self.training_data['timestamps'].append(current_time)
            self.training_data['steps'].append(step)
            self.training_data['rewards'].append(reward)
            self.training_data['success_rates'].append(success_rate)

            # 更新奖励曲线
            self.reward_line.set_data(self.training_data['steps'], self.training_data['rewards'])
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()

            # 更新成功率曲线
            self.success_line.set_data(self.training_data['steps'], self.training_data['success_rates'])
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()

            # 每10秒生成一个新的轨迹可视化
            if len(self.training_data['timestamps']) % 5 == 0:
                self._add_random_trajectory()

            # 更新统计信息
            self._update_stats_display()

        plt.draw()
        plt.pause(0.001)

    def _add_random_trajectory(self):
        """添加随机轨迹到3D可视化"""
        # 清除旧的轨迹
        self.ax_3d.clear()

        # 生成随机起点和终点
        start_pos = self._sample_random_position()
        goal_pos = self._sample_random_position()

        # 尝试规划轨迹
        try:
            waypoints = self.visualizer.visualize_rrt_star_planning(
                start_pos, goal_pos,
                show_tree=False
            )

            if waypoints:
                # 提取路径点
                path = np.array([wp.cartesian_position for wp in waypoints])

                # 绘制轨迹
                self.ax_3d.plot(path[:, 0], path[:, 1], path[:, 2],
                               'b-', linewidth=2, marker='o', markersize=4)
                self.ax_3d.scatter(*start_pos, color='green', s=100, marker='o', label='起点')
                self.ax_3d.scatter(*goal_pos, color='red', s=100, marker='*', label='终点')

                # 更新计数
                self.training_data['trajectories_planned'] += 1
                self.training_data['trajectories_successful'] += 1

            else:
                self.training_data['trajectories_planned'] += 1

        except Exception as e:
            print(f"轨迹规划失败: {e}")
            self.training_data['trajectories_planned'] += 1

        # 设置3D图形属性
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title(f'实时轨迹 #{self.training_data["trajectories_planned"]}')
        self.ax_3d.legend()

        # 设置固定视角
        self.ax_3d.view_init(elev=20, azim=45)

    def _sample_random_position(self) -> np.ndarray:
        """采样随机位置"""
        return np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.2, 0.7)
        ])

    def _update_stats_display(self):
        """更新统计信息显示"""
        current_time = datetime.now().strftime('%H:%M:%S')
        total_steps = self.training_data['steps'][-1] if self.training_data['steps'] else 0
        current_reward = self.training_data['rewards'][-1] if self.training_data['rewards'] else 0
        current_success = self.training_data['success_rates'][-1] if self.training_data['success_rates'] else 0

        stats_text = f"""
🕐 监控时间: {current_time}

📊 训练统计:
   总步数: {total_steps:,}
   当前奖励: {current_reward:.2f}
   成功率: {current_success:.1f}%

🎯 轨迹统计:
   已规划: {self.training_data['trajectories_planned']}
   成功: {self.training_data['trajectories_successful']}
   规划成功率: {self.training_data['trajectories_successful']/max(1,self.training_data['trajectories_planned'])*100:.1f}%

💻 系统状态:
   监控状态: {'运行中' if self.is_running else '已停止'}
   GPU设备: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}
   配置文件: config.yaml
        """

        self.stats_text.set_text(stats_text)

    def start_monitoring(self):
        """开始监控"""
        print("🖥️  开始训练监控...")
        self.is_running = True

        # 创建动画更新
        self.animation = FuncAnimation(
            self.fig, self.update_plots,
            interval=1000,  # 每秒更新
            blit=False
        )

        plt.show(block=True)

    def stop_monitoring(self):
        """停止监控"""
        print("⏹️  停止监控")
        self.is_running = False

    def save_monitoring_report(self, filename: str = "training_monitor_report.png"):
        """保存监控报告"""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📸 监控报告已保存: {filename}")


def main():
    """主函数"""
    print("🖥️  UR10e 训练实时监控器")
    print("=" * 40)

    monitor = TrainingMonitor()

    try:
        print("🎯 启动监控...")
        print("💡 提示: 关闭窗口停止监控")
        monitor.start_monitoring()

    except KeyboardInterrupt:
        print("\n⏹️ 监控被用户中断")
        monitor.stop_monitoring()

    except Exception as e:
        print(f"\n❌ 监控失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 保存最终报告
        monitor.save_monitoring_report()
        print("🏁 监控会话结束")


if __name__ == "__main__":
    main()