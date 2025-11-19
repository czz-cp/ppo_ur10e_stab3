"""
UR10e PPO Training with Stable-Baselines3

Pure RL control training script that replaces the RL-PID hybrid approach.
Uses incremental torque control (6D action space) with Isaac Gym integration.
"""

# Isaac Gym imports MUST be before any PyTorch imports
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")
    print("Please ensure Isaac Gym is properly installed")
    sys.exit(1)

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
from datetime import datetime

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Import our custom environment
from ur10e_incremental_env import UR10eIncrementalEnv

# Set up environment variables for GPU compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging CUDA errors

# Set device after Isaac Gym import
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU 2 becomes GPU 0
print(f"🔧 Using device: {device}")


class TrainingCallback(BaseCallback):
    """Custom callback for training progress monitoring"""

    def __init__(self, eval_freq: int = 10000, save_freq: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        self.start_time = time.time()

    def _on_rollout_start(self) -> None:
        """Called before collecting new rollout"""
        pass

    def _on_rollout_end(self) -> None:
        """Called after collecting rollout"""
        pass

    def _on_step(self) -> bool:
        """Called after each step"""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            rewards = []
            distances = []
            successes = 0

            for _ in range(10):  # 10 evaluation episodes
                obs = self.training_env.reset()
                done = False
                episode_reward = 0
                episode_distances = []

                while not done:
                    action, _ = self.model.policy.predict(obs, deterministic=True)
                    obs, reward, done, info = self.training_env.step(action)
                    episode_reward += reward
                    episode_distances.append(info.get('distance', 1.0))

                rewards.append(episode_reward)
                distances.append(np.mean(episode_distances))
                if episode_distances[-1] < 0.05:  # Success threshold
                    successes += 1

            mean_reward = np.mean(rewards)
            mean_distance = np.mean(distances)
            success_rate = successes / 10

            elapsed_time = time.time() - self.start_time

            print(f"\n📈 Evaluation at step {self.n_calls}:")
            print(f"   🎯 Mean Reward: {mean_reward:.4f}")
            print(f"   📏 Mean Distance: {mean_distance:.4f}m")
            print(f"   ✅ Success Rate: {success_rate*100:.1f}%")
            print(f"   ⏱️  Elapsed Time: {elapsed_time/60:.1f}min")

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"models/best_ur10e_ppo_{self.n_calls}")
                print(f"   💾 Saved new best model: reward={mean_reward:.4f}")

        if self.n_calls % self.save_freq == 0:
            self.model.save(f"models/ur10e_ppo_{self.n_calls}")
            print(f"💾 Checkpoint saved: ur10e_ppo_{self.n_calls}")

        return True


def create_single_env(config_path: str):
    """Create a single environment instance"""
    def _init():
        env = UR10eIncrementalEnv(config_path=config_path, num_envs=1)
        env = Monitor(env, filename="./monitor_logs/")
        return env
    return _init


def create_vec_env(config_path: str, num_envs: int):
    """Create vectorized environment for training"""
    # Check if visualization is enabled
    config = load_config(config_path)
    viz_enabled = config.get('visualization', {}).get('enable', False)

    if viz_enabled or num_envs == 1:
        # Use DummyVecEnv for visualization (single process)
        env_fns = [create_single_env(config_path) for _ in range(num_envs)]
        vec_env = DummyVecEnv(env_fns)
        print(f"🎬 Using DummyVecEnv for visualization (num_envs={num_envs})")
    else:
        # Use SubprocVecEnv for better parallelization (no visualization)
        env_fns = [create_single_env(config_path) for _ in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        print(f"🚀 Using SubprocVecEnv for parallel training (num_envs={num_envs})")

    return vec_env


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file {config_path} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration: {e}")
        sys.exit(1)


def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'logs', 'monitor_logs', 'tensorboard_logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created")


def validate_environment(env):
    """Validate environment compatibility with Stable-Baselines3"""
    try:
        check_env(env)
        print("✅ Environment validation passed")
        return True
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train UR10e PPO with Stable-Baselines3')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
                        help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=16,
                        help='Number of parallel environments')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Save frequency')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume training')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detect if not specified)')

    args = parser.parse_args()

    print("🚀 Starting UR10e PPO Training with Stable-Baselines3")
    print("=" * 60)

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    setup_directories()

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"🔧 Using device: {device}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(device.index)}")

    # Create environment
    print(f"\n🏗️  Creating environment...")
    num_envs = min(args.num_envs, config.get('env', {}).get('num_envs', 16))
    print(f"   Parallel environments: {num_envs}")

    # Create single environment first for validation
    single_env = UR10eIncrementalEnv(config_path=args.config, num_envs=1)

    # Skip validation for now to isolate CUDA issues
    # if not validate_environment(single_env):
    #     sys.exit(1)
    print("⚠️ Skipping environment validation to debug CUDA issue")

    # Print environment info
    print(f"   Action space: {single_env.action_space}")
    print(f"   Observation space: {single_env.observation_space}")

    # Test environment
    print(f"\n🧪 Testing environment...")
    obs, info = single_env.reset()
    action = single_env.action_space.sample()
    obs, reward, terminated, truncated, info = single_env.step(action)
    print(f"   Test step successful: reward={reward:.4f}")

    # Close single environment to avoid Isaac Gym conflicts
    print("🔄 Closing test environment to avoid conflicts...")
    single_env.close()

    # Create vectorized environment for training
    print(f"\n🔄 Creating vectorized environment...")
    vec_env = create_vec_env(args.config, num_envs)

    # Configure logger
    log_dir = "./tensorboard_logs/"
    logger = configure(log_dir, ["tensorboard"])

    # Load PPO configuration
    ppo_config = config.get('ppo', {})

    # Create PPO model
    print(f"\n🤖 Creating PPO model...")
    print(f"   Policy: {ppo_config.get('policy', 'MlpPolicy')}")
    print(f"   Learning rate: {ppo_config.get('learning_rate', 3e-4)}")
    print(f"   Total timesteps: {args.total_timesteps:,}")

    policy_kwargs = ppo_config.get('policy_kwargs', {})

    # Convert string activation function to actual function
    if 'activation_fn' in policy_kwargs:
        activation_fn_str = policy_kwargs['activation_fn']
        if activation_fn_str == 'relu':
            policy_kwargs['activation_fn'] = nn.ReLU
        elif activation_fn_str == 'tanh':
            policy_kwargs['activation_fn'] = nn.Tanh
        elif activation_fn_str == 'sigmoid':
            policy_kwargs['activation_fn'] = nn.Sigmoid
        else:
            print(f"⚠️ Unknown activation function: {activation_fn_str}, using ReLU")
            policy_kwargs['activation_fn'] = nn.ReLU

    model = PPO(
        policy=ppo_config.get('policy', 'MlpPolicy'),
        env=vec_env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.995),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        clip_range_vf=ppo_config.get('clip_range_vf', None),
        normalize_advantage=ppo_config.get('normalize_advantage', True),
        ent_coef=ppo_config.get('ent_coef', 0.02),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        use_sde=ppo_config.get('use_sde', False),
        sde_sample_freq=ppo_config.get('sde_sample_freq', -1),
        target_kl=ppo_config.get('target_kl', None),
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
        device=str(device)
    )

    # Resume training if specified
    if args.resume:
        print(f"📂 Resuming training from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=log_dir, device=device)

    # Create callback
    callback = TrainingCallback(
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        verbose=1
    )

    print(f"\n🎯 Starting training...")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Evaluation frequency: {args.eval_freq:,}")
    print(f"   Save frequency: {args.save_freq:,}")
    print(f"   TensorBoard logs: {log_dir}")

    start_time = time.time()

    try:
        # Start training
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )

        # Save final model
        model.save("models/ur10e_ppo_final")
        print("💾 Final model saved: ur10e_ppo_final")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        model.save("models/ur10e_ppo_interrupted")
        print("💾 Model saved: ur10e_ppo_interrupted")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        model.save("models/ur10e_ppo_error")
        print("💾 Error state model saved: ur10e_ppo_error")

    finally:
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total training time: {elapsed_time/60:.1f} minutes")

    # Test final model (disabled to avoid Isaac Gym multi-instance conflict)
    print(f"\n🧪 Testing final model...")
    print("⚠️ Testing disabled to avoid Isaac Gym multi-instance conflict")
    print("💡 To test the trained model, use: python inference.py --model models/ur10e_ppo_final")
    vec_env.close()

    print(f"\n🎉 Training completed successfully!")
    print(f"📂 Models saved in: ./models/")
    print(f"📊 TensorBoard logs: {log_dir}")


if __name__ == "__main__":
    main()