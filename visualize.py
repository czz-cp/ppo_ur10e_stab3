#!/usr/bin/env python3
"""
UR10e Visualization Test

Test script for visualizing the UR10e simulation without training.
Use this to verify that the simulation is working correctly before training.
"""

# Isaac Gym imports MUST be before any PyTorch imports
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")
    import sys
    sys.exit(1)

import os
import time
import numpy as np
import torch
import argparse

# Set up environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Import our environment
from ur10e_incremental_env import UR10eIncrementalEnv


def test_visualization(config_path: str = "config.yaml", duration: int = 60):
    """
    Test visualization with a single environment

    Args:
        config_path: Path to configuration file
        duration: Test duration in seconds
    """
    print("🎬 UR10e Visualization Test")
    print("=" * 50)

    try:
        # Create environment with visualization
        print("🏗️ Creating environment with visualization...")
        env = UR10eIncrementalEnv(config_path=config_path, num_envs=1)

        # Reset environment
        print("🔄 Resetting environment...")
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Initial distance to target: {info.get('distance', 'N/A')}")

        # Run visualization test
        print(f"\n🎬 Running visualization test for {duration} seconds...")
        print("   📹 Use mouse to rotate camera view")
        print("   🖱️  Scroll to zoom in/out")
        print("   ⌨️  Press ESC to close window early")
        print("   🔴 Red sphere: Target position")
        print("   🔵 Blue object: UR10e robot")

        start_time = time.time()
        step_count = 0

        while time.time() - start_time < duration:
            # Generate random action (small torques)
            action = np.random.uniform(-1, 1, 6) * 2.0  # Small random torques

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            step_count += 1

            # Print status every 100 steps
            if step_count % 100 == 0:
                distance = info.get('distance', 1.0)
                print(f"   Step {step_count}: Distance = {distance:.4f}m, Reward = {reward:.4f}")

            # Check if episode completed
            if terminated:
                print(f"   🎯 Episode completed at step {step_count}! Distance: {info.get('distance', 'N/A')}")
                obs, info = env.reset()
                print("   🔄 Environment reset for new episode")

        # Print final statistics
        print(f"\n📊 Visualization test completed:")
        print(f"   Total steps: {step_count}")
        print(f"   Duration: {time.time() - start_time:.1f} seconds")
        print(f"   Average steps per second: {step_count / (time.time() - start_time):.1f}")

        # Close environment
        env.close()
        print("✅ Environment closed successfully")

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        if 'env' in locals():
            env.close()

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.close()


def main():
    parser = argparse.ArgumentParser(description='UR10e Visualization Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds')

    args = parser.parse_args()

    print(f"🎬 Configuration: {args.config}")
    print(f"⏱️  Duration: {args.duration} seconds")

    # Check if visualization is enabled in config
    import yaml
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        viz_enabled = config.get('visualization', {}).get('enable', False)
        num_envs = config.get('env', {}).get('num_envs', 1)

        print(f"🎬 Visualization enabled: {viz_enabled}")
        print(f"🔢 Number of environments: {num_envs}")

        if not viz_enabled:
            print("\n⚠️  Visualization is disabled in config.yaml")
            print("   Set visualization.enable: true to enable visualization")
            return

        if num_envs > 1:
            print("\n⚠️  Multiple environments detected (num_envs > 1)")
            print("   Visualization works best with num_envs: 1")
            print("   Consider setting env.num_envs: 1 for visualization")

    except Exception as e:
        print(f"⚠️  Could not read config file: {e}")

    # Run visualization test
    test_visualization(args.config, args.duration)

    print(f"\n🎉 Visualization test completed!")
    print(f"💡 For training, use: python train.py --config {args.config}")
    print(f"💡 To disable visualization for faster training:")
    print(f"   Set visualization.enable: false in config.yaml")


if __name__ == "__main__":
    main()