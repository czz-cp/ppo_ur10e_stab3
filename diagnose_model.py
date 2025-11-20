#!/usr/bin/env python3
"""
UR10e Quick Control Check - 快速检查控制是否工作
"""



from ur10e_trajectory_env import UR10eTrajectoryEnv
import torch
import numpy as np
from stable_baselines3 import PPO

def quick_control_check():
    """快速控制检查"""
    print("🔧 Quick Control Check")
    print("=" * 40)
    
    # 加载模型
    model = PPO.load("models/trajectory_model_final.zip")
    print("✅ Model loaded")
    
    
    
    env = UR10eTrajectoryEnv()
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, info = obs
    else:
        info = {}
    
    print(f"Initial observation shape: {obs.shape}")
    
    # 测试5个步骤
    print("\nTesting 5 steps with model actions:")
    rewards = []
    
    for step in range(5):
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
        
        rewards.append(reward)
        print(f"Step {step}: reward = {reward:.3f}, action norm = {np.linalg.norm(action):.3f}")
        
        if done:
            break
    
    env.close()
    
    # 分析结果
    print(f"\n📊 Results:")
    print(f"Rewards: {[f'{r:.3f}' for r in rewards]}")
    
    if len(set([round(r, 2) for r in rewards])) == 1:
        print("❌ CONTROL NOT WORKING - All rewards identical!")
        print("   The robot is not moving despite action inputs")
    else:
        print("✅ Control is working - rewards are changing")
        print("   The robot is responding to actions")

if __name__ == "__main__":
    quick_control_check()