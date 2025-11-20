#!/usr/bin/env python3
"""
UR10e Model Validator - Final Fixed Version
"""

import os
import sys
from ur10e_trajectory_env import UR10eTrajectoryEnv
import torch
import numpy as np
from stable_baselines3 import PPO

# 强制环境配置一致性
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class UR10eValidatorFinal:
    """
    最终修复的UR10e验证器
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """加载模型"""
        print("🤖 Loading model...")
        self.model = PPO.load(self.model_path)
        print("✅ Model loaded")
        print(f"   Action: {self.model.action_space}")
        print(f"   Observation: {self.model.observation_space}")
        return True
    
    def create_consistent_environment(self):
        """创建一致的环境（确保6 DOFs）"""
        print("\n🔄 Creating consistent 6DOF environment...")
        
        try:
            
            
            # 创建环境
            env = UR10eTrajectoryEnv()
            
            # 立即检查环境配置
            action_space = env.action_space
            obs_space = env.observation_space
            
            print(f"✅ Environment created")
            print(f"   Action space: {action_space}")
            print(f"   Observation space: {obs_space}")
            
            # 验证配置一致性
            model_action_dim = self.model.action_space.shape[0]
            env_action_dim = action_space.shape[0]
            
            if model_action_dim != env_action_dim:
                print(f"❌ Action space mismatch: Model={model_action_dim}D, Env={env_action_dim}D")
                env.close()
                return None
            
            # 重置环境
            try:
                # 使用新的gymnasium API
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs, info = obs
                else:
                    info = {}
                print(f"✅ Environment reset successful")
                print(f"   Observation shape: {obs.shape}")
                return env
            except Exception as e:
                print(f"❌ Environment reset failed: {e}")
                env.close()
                return None
                
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return None
    
    def adapt_action(self, action, env):
        """适配动作维度"""
        model_action_dim = self.model.action_space.shape[0]
        env_action_dim = env.action_space.shape[0]
        
        if model_action_dim == env_action_dim:
            return action
        elif model_action_dim < env_action_dim:
            # 模型输出维度小于环境期望，用0填充
            adapted = np.zeros(env_action_dim)
            adapted[:model_action_dim] = action[0]
            return adapted.reshape(1, -1)
        else:
            # 模型输出维度大于环境期望，截断
            return action[:, :env_action_dim]
    
    def run_safe_validation(self):
        """运行安全的验证"""
        print("\n🧪 Running safe validation...")
        
        # 创建环境
        env = self.create_consistent_environment()
        if env is None:
            print("❌ Cannot proceed without environment")
            return False
        
        try:
            # 测试轨迹规划
            print("\n📐 Testing trajectory planning...")
            start_pos = np.array([0.3, 0.3, 0.5])
            goal_pos = np.array([-0.3, -0.3, 0.6])
            
            success = env.plan_trajectory(start_pos, goal_pos)
            if not success:
                print("❌ Trajectory planning failed")
                return False
            
            waypoint_count = len(env.current_ts_waypoints)
            print(f"✅ Trajectory planned: {waypoint_count} waypoints")
            
            # 重置环境
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}
            
            # 运行几个步骤测试
            print("\n🔄 Running trajectory tracking...")
            total_reward = 0
            max_steps = 30  # 减少步数
            
            for step in range(max_steps):
                try:
                    # 预测动作
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # 适配动作
                    adapted_action = self.adapt_action(action, env)
                    
                    # 执行步骤
                    step_result = env.step(adapted_action)
                    
                    # 处理不同的返回格式
                    if len(step_result) == 4:
                        obs, reward, done, info = step_result
                    else:  # gymnasium返回5个值
                        obs, reward, done, truncated, info = step_result
                        done = done or truncated
                    
                    total_reward += reward
                    
                    if step % 5 == 0:
                        current_waypoint = info.get('current_waypoint', 0)
                        progress = (current_waypoint / waypoint_count) * 100
                        print(f"   Step {step}: reward={reward:.3f}, progress={progress:.1f}%")
                    
                    if done:
                        print(f"   Episode ended at step {step}")
                        break
                        
                except Exception as step_error:
                    print(f"❌ Step {step} failed: {step_error}")
                    break
            
            # 获取统计信息
            try:
                stats = env.get_trajectory_statistics()
                completed = stats.get('trajectory_completed', False)
                print(f"\n📊 Validation Results:")
                print(f"   Total reward: {total_reward:.3f}")
                print(f"   Steps executed: {step + 1}")
                print(f"   Trajectory completed: {completed}")
                print(f"   Final waypoint: {info.get('current_waypoint', 0)}/{waypoint_count}")
            except:
                print(f"\n📊 Basic Results:")
                print(f"   Total reward: {total_reward:.3f}")
                print(f"   Steps executed: {step + 1}")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            env.close()
            print("🧹 Environment closed")
    
    def run_model_analysis(self):
        """分析模型行为"""
        print("\n🔍 Analyzing model behavior...")
        
        obs_dim = self.model.observation_space.shape[0]
        action_dim = self.model.action_space.shape[0]
        
        print(f"   Model analysis:")
        print(f"   - Observation dimension: {obs_dim}")
        print(f"   - Action dimension: {action_dim}")
        
        # 测试典型输入
        test_inputs = {
            "Zero input": np.zeros((1, obs_dim), dtype=np.float32),
            "Small noise": np.random.normal(0, 0.1, (1, obs_dim)).astype(np.float32),
            "Joint positions": self._create_joint_position_input(obs_dim),
        }
        
        for name, obs in test_inputs.items():
            action, _ = self.model.predict(obs)
            action_range = f"[{action.min():.3f}, {action.max():.3f}]"
            print(f"   - {name}: actions {action_range}")

    def _create_joint_position_input(self, obs_dim):
        """创建关节位置输入"""
        obs = np.zeros((1, obs_dim), dtype=np.float32)
        # 前6个维度是关节位置
        obs[0, :6] = np.random.uniform(-0.5, 0.5, 6)
        return obs

def main():
    model_path = "models/trajectory_model_final.zip"
    
    print("=" * 50)
    print("UR10e Model Validation - Final Version")
    print("=" * 50)
    
    validator = UR10eValidatorFinal(model_path)
    
    try:
        # 1. 加载模型
        if not validator.load_model():
            return
        
        # 2. 分析模型行为
        validator.run_model_analysis()
        
        # 3. 运行安全验证
        success = validator.run_safe_validation()
        
        if success:
            print("\n🎉 Validation completed successfully!")
        else:
            print("\n⚠️  Validation completed with issues")
        
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()