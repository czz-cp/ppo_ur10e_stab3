#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_trajectory.py
验证/评估训练好的 UR10e 轨迹跟踪 PPO 模型

用法：
python eval_trajectory.py \
    --model ./logs/trajectory_training_xxx/trajectory_model_final.zip \
    --config config.yaml \
    --episodes 20 \
    --max_steps 600 \
    --deterministic

说明：
- 每个 episode：
  1) reset 到 point_to_point 模式（避免 reset() 自动规划覆盖）
  2) 取当前 TCP 作 start
  3) 随机采样 goal
  4) 切回 trajectory_tracking 并 plan_trajectory(start, goal)
  5) 用模型 deterministic / stochastic 推理执行
- 输出成功率、平均回报、平均步数、最终距离等
"""

import os
import sys
import argparse
import time
import numpy as np

# ---------------- Isaac Gym 必须早于 torch ----------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
try:
    from isaacgym import gymapi, gymtorch, gymutil
    from isaacgym.torch_utils import *  # noqa
    print("✅ Isaac Gym imported successfully in eval_trajectory.py")
except Exception as e:
    print(f"❌ Failed to import Isaac Gym in eval_trajectory.py: {e}")
    sys.exit(1)

import torch
import yaml
from stable_baselines3 import PPO

from ur10e_trajectory_env import UR10eTrajectoryEnv


def load_config(config_path: str):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"✅ Config loaded: {config_path}")
        return cfg
    except FileNotFoundError:
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)


def sample_goal_from_config(cfg):
    ws_cfg = cfg.get("task_space", {}).get("workspace_bounds", {})
    def _axis(name, default):
        return ws_cfg.get(name, default)

    goal = np.array(
        [
            np.random.uniform(*_axis("x", [-0.6, 0.6])),
            np.random.uniform(*_axis("y", [-0.6, 0.6])),
            np.random.uniform(*_axis("z", [0.2, 0.8])),
        ],
        dtype=np.float32,
    )
    return goal


@torch.no_grad()
def get_current_tcp(env: UR10eTrajectoryEnv):
    tcp = env._forward_kinematics(env.joint_positions[0]).detach().cpu().numpy()
    return tcp.astype(np.float32)


def evaluate(model_path: str,
             config_path: str,
             episodes: int = 10,
             max_steps: int = 500,
             deterministic: bool = True,
             seed: int = 0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = load_config(config_path)

    # 评估只开 1 个 env 最稳
    env = UR10eTrajectoryEnv(config_path=config_path, num_envs=1, mode="point_to_point")

    # load 模型（绑定 env 以保证 action/obs space 一致）
    model = PPO.load(model_path, env=env)
    print(f"✅ Model loaded: {model_path}")

    success_list = []
    reward_list = []
    steps_list = []
    final_dist_list = []

    for ep in range(1, episodes + 1):
        print(f"\n================ Episode {ep}/{episodes} ================")

        # 1) reset 到 point_to_point 模式，避免 reset() 自动规划覆盖
        obs, info = env.reset(options={"mode": "point_to_point"})

        # 2) 当前 TCP 作为 start
        start_tcp = get_current_tcp(env)

        # 3) 随机采样 goal
        goal_tcp = sample_goal_from_config(cfg)

        # 4) 切回 trajectory_tracking 并规划
        env.set_mode("trajectory_tracking")
        ok = env.plan_trajectory(start_tcp, goal_tcp)
        if not ok:
            print("❌ Planning failed, skip this episode.")
            success_list.append(False)
            reward_list.append(0.0)
            steps_list.append(0)
            final_dist_list.append(float("inf"))
            continue

        # 规划后重新取 obs（19D 带 delta_to_waypoint/progress）
        obs = env.get_observation()

        done = False
        total_reward = 0.0
        steps = 0

        t0 = time.time()
        while (not done) and steps < max_steps:
            obs_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)

            if obs_np.ndim == 1:
                obs_np = obs_np[None, :]   # (1, obs_dim)

            action, _ = model.predict(obs_np, deterministic=deterministic)
            action = np.asarray(action).reshape(-1)  # (6,)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            done = bool(terminated or truncated)
            steps += 1

            if steps % 50 == 0:
                dist = info.get("distance_to_waypoint", None)
                print(f"[ep {ep}] step {steps:4d} | r_sum={total_reward:8.3f} | "
                      f"wp={info.get('current_waypoint', -1)}/{info.get('total_waypoints', -1)} | "
                      f"dist={dist}")

        dt = time.time() - t0

        # 5) 统计
        stats = env.get_trajectory_statistics()
        traj_completed = stats.get("trajectory_completed", False)
        if isinstance(traj_completed, (np.ndarray, list)):
            traj_completed = bool(traj_completed[0])

        # 最终距离：用 info 的 distance_to_waypoint（若已完成会接近 0），否则自己算到最后一个 wp
        if isinstance(info, dict) and "distance_to_waypoint" in info:
            final_dist = float(info["distance_to_waypoint"])
        else:
            final_tcp = get_current_tcp(env)
            if env.current_ts_waypoints:
                final_wp = env.current_ts_waypoints[-1].cartesian_position
                final_dist = float(np.linalg.norm(final_tcp - final_wp))
            else:
                final_dist = float("inf")

        print(f"Episode {ep} finished in {steps} steps, {dt:.2f}s")
        print(f"  total_reward = {total_reward:.3f}")
        print(f"  trajectory_completed = {traj_completed}")
        print(f"  final_dist_to_goal = {final_dist:.4f} m")
        print(f"  goal_tcp = {goal_tcp}")

        success_list.append(traj_completed)
        reward_list.append(total_reward)
        steps_list.append(steps)
        final_dist_list.append(final_dist)

    env.close()

    # 汇总
    success_rate = np.mean(success_list) * 100.0
    mean_reward = np.mean(reward_list)
    mean_steps = np.mean(steps_list)
    mean_final_dist = np.mean(final_dist_list)

    print("\n================== Evaluation Summary ==================")
    print(f"Episodes: {episodes}")
    print(f"Success rate: {success_rate:.1f}% ({sum(success_list)}/{episodes})")
    print(f"Mean total reward: {mean_reward:.3f}")
    print(f"Mean steps: {mean_steps:.1f}")
    print(f"Mean final dist: {mean_final_dist:.4f} m")
    print("========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("UR10e PPO Trajectory Model Evaluation")
    parser.add_argument("--model", type=str, default= "models/trajectory_model_final.zip",
                        help="path to SB3 PPO model .zip (e.g. trajectory_model_final.zip)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="path to config.yaml")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of test episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="max steps per episode")
    parser.add_argument("--deterministic", action="store_true",
                        help="use deterministic policy")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed
    )
