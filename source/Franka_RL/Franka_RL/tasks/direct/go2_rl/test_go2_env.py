#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
测试Go2 Direct环境是否正确设置
"""

import argparse

# 🔑 关键: 必须先启动Isaac Sim，再导入其他模块
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Test Go2 Direct environment")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")

# 添加AppLauncher参数 (会自动添加 --headless, --device 等参数)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 🔑 启动Isaac Sim应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 现在可以安全导入其他模块
import torch
import gymnasium as gym

# 注册Go2环境
import Franka_RL.tasks.direct.go2_rl


def test_go2_environment(num_steps: int = 100, num_envs: int = 16):
    """测试Go2环境"""
    
    print("=" * 80)
    print("Testing Go2 Direct Environment")
    print("=" * 80)
    
    # 导入配置类
    from Franka_RL.tasks.direct.go2_rl.go2_env_cfg import Go2EnvCfg
    
    # 创建环境配置
    env_cfg = Go2EnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # 创建环境
    print(f"\n🔄 Creating environment with {num_envs} parallel environments...")
    env = gym.make("Isaac-Go2-Direct-v0", cfg=env_cfg)
    
    print(f"\n✅ Environment created successfully!")
    print(f"  - Num envs: {env.unwrapped.num_envs}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Device: {env.unwrapped.device}")
    
    # 重置环境
    print(f"\n🔄 Resetting environment...")
    obs, info = env.reset()
    
    print(f"\n✅ Environment reset successfully!")
    print(f"  - Observation shape: {obs['policy'].shape}")
    print(f"  - Expected shape: ({num_envs}, 48)")
    
    # 检查观测值范围
    obs_policy = obs['policy']
    print(f"\n📊 Observation statistics:")
    print(f"  - Min: {obs_policy.min():.4f}")
    print(f"  - Max: {obs_policy.max():.4f}")
    print(f"  - Mean: {obs_policy.mean():.4f}")
    print(f"  - Std: {obs_policy.std():.4f}")
    
    # 运行随机策略
    print(f"\n🏃 Running {num_steps} steps with random policy...")
    
    total_reward = 0.0
    episode_count = 0
    step_rewards = []
    
    for step in range(num_steps):
        # 随机动作 [-1, 1]
        action = 2.0 * torch.rand(
            (num_envs, env.unwrapped.cfg.action_space),
            device=env.unwrapped.device
        ) - 1.0
        
        # 执行步骤
        obs, reward, terminated, truncated, info = env.step(action)
        
        mean_reward = reward.mean().item()
        total_reward += mean_reward
        step_rewards.append(mean_reward)
        
        # 打印进度
        if (step + 1) % 20 == 0:
            print(f"\n  📈 Step {step + 1}/{num_steps}")
            print(f"     - Avg reward: {mean_reward:.4f}")
            print(f"     - Min reward: {reward.min():.4f}")
            print(f"     - Max reward: {reward.max():.4f}")
            
            # 打印详细奖励分解
            if "log" in env.unwrapped.extras:
                log = env.unwrapped.extras["log"]
                print(f"     - Reward breakdown:")
                for key, value in sorted(log.items()):
                    if key.startswith("rewards/"):
                        reward_name = key.replace("rewards/", "")
                        print(f"       • {reward_name}: {value:.4f}")
        
        # 检查重置
        done = terminated | truncated
        if done.any():
            num_done = done.sum().item()
            episode_count += num_done
            if num_done > 0:
                print(f"     ⚠️  {num_done} environments reset")
    
    # 最终统计
    print(f"\n" + "=" * 80)
    print("📊 Final Statistics")
    print("=" * 80)
    print(f"  - Total episodes finished: {episode_count}")
    print(f"  - Average reward per step: {total_reward / num_steps:.4f}")
    print(f"  - Reward std: {torch.tensor(step_rewards).std():.4f}")
    print(f"  - Best step reward: {max(step_rewards):.4f}")
    print(f"  - Worst step reward: {min(step_rewards):.4f}")
    
    # 测试环境状态
    print(f"\n🔍 Environment State Check:")
    robot = env.unwrapped.robot
    print(f"  - Base position: {robot.data.root_state_w[0, :3]}")
    print(f"  - Base velocity: {robot.data.root_lin_vel_w[0]}")
    print(f"  - Joint positions (first 3): {robot.data.joint_pos[0, :3]}")
    print(f"  - Joint velocities (first 3): {robot.data.joint_vel[0, :3]}")
    
    # 关闭环境
    print(f"\n🔄 Closing environment...")
    env.close()
    print("✅ Environment closed successfully!")
    
    return True


def main():
    """Main function"""
    
    try:
        success = test_go2_environment(
            num_steps=args.num_steps,
            num_envs=args.num_envs
        )
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 All tests passed!")
            print("=" * 80)
            print("\n💡 Next steps:")
            print("  1. Train with PPO:")
            print("     python scripts/rsl_rl/train.py --task Isaac-Go2-Direct-v0")
            print("\n  2. Test with more environments:")
            print(f"     python {__file__} --num_envs 256 --num_steps 500")
            print("\n  3. Run in headless mode for faster testing:")
            print(f"     python {__file__} --headless --num_steps 1000")
            
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 关闭仿真应用
        simulation_app.close()


if __name__ == "__main__":
    main()