"""
UR10e PPO Inference with Stable-Baselines3

Inference script for testing trained models with pure RL control.
Supports visualization and performance analysis.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Set up environment variables for GPU 2 (server compatibility)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Stable-Baselines3 imports
from stable_baselines3 import PPO

# Import our custom environment
from ur10e_incremental_env import UR10eIncrementalEnv


class UR10eInference:
    """UR10e PPO inference class for model testing and visualization"""

    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize inference

        Args:
            model_path: Path to trained model (.zip file)
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path

        # Load model
        print(f"🤖 Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        print("✅ Model loaded successfully")

        # Create environment
        print(f"🏗️  Creating environment...")
        self.env = UR10eIncrementalEnv(config_path=config_path, num_envs=1)
        print("✅ Environment created")

        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_lengths = []
        self.success_episodes = 0
        self.total_episodes = 0

    def run_episode(self, render: bool = False, max_steps: int = None) -> Dict[str, Any]:
        """
        Run a single episode

        Args:
            render: Whether to render the episode
            max_steps: Maximum steps per episode (overrides config)

        Returns:
            Dictionary with episode statistics
        """
        obs, info = self.env.reset()
        done = False
        truncated = False

        episode_reward = 0
        episode_steps = 0
        episode_distances = []
        episode_torques = []

        # Override max steps if specified
        if max_steps:
            original_max_steps = self.env.max_steps
            self.env.max_steps = max_steps

        while not done and not truncated:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Track statistics
            episode_reward += reward
            episode_steps += 1
            episode_distances.append(info.get('distance', 1.0))
            episode_torques.append(info.get('torques', np.zeros(6)))

            done = terminated
            truncated = truncated

            # Optional rendering (simplified console output)
            if render and episode_steps % 50 == 0:
                print(f"   Step {episode_steps}: Reward={reward:.4f}, Distance={episode_distances[-1]:.4f}")

        # Restore original max steps
        if max_steps:
            self.env.max_steps = original_max_steps

        # Calculate episode statistics
        final_distance = episode_distances[-1] if episode_distances else 1.0
        success = final_distance < self.env.config.get('reward', {}).get('success_threshold', 0.05)

        episode_stats = {
            'reward': episode_reward,
            'length': episode_steps,
            'final_distance': final_distance,
            'success': success,
            'mean_distance': np.mean(episode_distances),
            'distance_progress': episode_distances,
            'torques': np.array(episode_torques),
            'mean_torque': np.mean(np.abs(episode_torques)),
            'max_torque': np.max(np.abs(episode_torques))
        }

        return episode_stats

    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate model over multiple episodes

        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            Dictionary with evaluation statistics
        """
        print(f"\n🧪 Evaluating model over {num_episodes} episodes...")

        episode_stats = []

        for episode in range(num_episodes):
            if render:
                print(f"\n🎬 Episode {episode + 1}/{num_episodes}")

            stats = self.run_episode(render=render)
            episode_stats.append(stats)

            # Update global tracking
            self.episode_rewards.append(stats['reward'])
            self.episode_distances.append(stats['final_distance'])
            self.episode_lengths.append(stats['length'])

            if stats['success']:
                self.success_episodes += 1
            self.total_episodes += 1

            # Print progress
            if (episode + 1) % 10 == 0:
                recent_success_rate = np.mean([s['success'] for s in episode_stats[-10:]]) * 100
                recent_reward = np.mean([s['reward'] for s in episode_stats[-10:]])
                print(f"   Episodes {episode-9}-{episode}: Success Rate={recent_success_rate:.1f}%, Reward={recent_reward:.4f}")

        # Calculate overall statistics
        rewards = [s['reward'] for s in episode_stats]
        distances = [s['final_distance'] for s in episode_stats]
        lengths = [s['length'] for s in episode_stats]
        successes = [s['success'] for s in episode_stats]

        evaluation_stats = {
            'total_episodes': num_episodes,
            'success_rate': np.mean(successes) * 100,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'best_episode_idx': np.argmax(rewards),
            'worst_episode_idx': np.argmin(rewards)
        }

        return evaluation_stats

    def print_evaluation_summary(self, stats: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n📊 Evaluation Summary:")
        print(f"   Episodes: {stats['total_episodes']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
        print(f"   Distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")
        print(f"   Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"   Best Episode: #{stats['best_episode_idx'] + 1} (Reward: {stats['max_reward']:.4f})")
        print(f"   Worst Episode: #{stats['worst_episode_idx'] + 1} (Reward: {stats['min_reward']:.4f})")

    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if not self.episode_rewards:
            print("⚠️ No episode data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('UR10e PPO Inference Performance', fontsize=16)

        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # Add moving average
        if len(self.episode_rewards) > 10:
            window_size = min(10, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'MA({window_size})')
            axes[0, 0].legend()

        # Plot 2: Success Rate
        if len(self.episode_rewards) > 0:
            success_window = 10
            success_rates = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - success_window + 1)
                successes = sum(1 for j in range(start_idx, i + 1) if self.episode_distances[j] < 0.05)
                success_rates.append(successes / (i - start_idx + 1) * 100)

            axes[0, 1].plot(success_rates)
            axes[0, 1].set_title('Rolling Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].grid(True)
            axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50%')
            axes[0, 1].legend()

        # Plot 3: Final Distances
        axes[1, 0].plot(self.episode_distances)
        axes[1, 0].set_title('Final Distances to Target')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Distance (m)')
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Success Threshold')
        axes[1, 0].legend()

        # Plot 4: Episode Lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.show()

    def analyze_best_episode(self, num_test_episodes: int = 100):
        """Analyze the best episode from recent test"""
        print(f"\n🔍 Analyzing best episode from {num_test_episodes} test episodes...")

        best_stats = None
        best_reward = -np.inf

        for episode in range(num_test_episodes):
            stats = self.run_episode(render=False)
            if stats['reward'] > best_reward:
                best_reward = stats['reward']
                best_stats = stats

        if best_stats:
            print(f"\n🏆 Best Episode Analysis:")
            print(f"   Reward: {best_stats['reward']:.4f}")
            print(f"   Length: {best_stats['length']} steps")
            print(f"   Final Distance: {best_stats['final_distance']:.4f}m")
            print(f"   Success: {best_stats['success']}")
            print(f"   Mean Torque: {best_stats['mean_torque']:.2f} N⋅m")
            print(f"   Max Torque: {best_stats['max_torque']:.2f} N⋅m")

            # Plot distance progress
            plt.figure(figsize=(10, 6))
            plt.plot(best_stats['distance_progress'])
            plt.title(f'Best Episode: Distance Progress Over Time (Reward: {best_stats["reward"]:.4f})')
            plt.xlabel('Step')
            plt.ylabel('Distance to Target (m)')
            plt.grid(True)
            plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Success Threshold')
            plt.legend()
            plt.show()

    def close(self):
        """Close environment"""
        if hasattr(self, 'env'):
            self.env.close()
        print("✅ Inference environment closed")


def main():
    parser = argparse.ArgumentParser(description='UR10e PPO Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes during evaluation')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training progress')
    parser.add_argument('--analyze-best', action='store_true',
                        help='Analyze best episode in detail')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save performance plot')

    args = parser.parse_args()

    print("🚀 Starting UR10e PPO Inference")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file {args.model} not found")
        sys.exit(1)

    # Initialize inference
    inference = UR10eInference(model_path=args.model, config_path=args.config)

    try:
        # Run evaluation
        stats = inference.evaluate(num_episodes=args.episodes, render=args.render)

        # Print summary
        inference.print_evaluation_summary(stats)

        # Plot progress if requested
        if args.plot:
            inference.plot_training_progress(save_path=args.save_plot)

        # Analyze best episode if requested
        if args.analyze_best:
            inference.analyze_best_episode(num_test_episodes=min(20, args.episodes))

    except KeyboardInterrupt:
        print("\n⏹️  Inference interrupted by user")

    except Exception as e:
        print(f"\n❌ Inference failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close environment
        inference.close()

    print(f"\n🎉 Inference completed!")


if __name__ == "__main__":
    main()