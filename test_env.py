#!/usr/bin/env python3
"""
简单的环境测试脚本
用于验证矿山环境是否能正常初始化和运行
"""
import gymnasium as gym
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.getcwd(), 'openmines'))

def test_single_env():
    """测试单个环境"""
    print("开始测试单个环境...")
    
    try:
        # 创建环境
        print("1. 创建环境...")
        env = gym.make(
            "mine/Mine-v1", 
            config_file="openmines/src/conf/north_pit_mine.json",
            use_enhanced_observation=True
        )
        print("   环境创建成功!")
        
        # 重置环境
        print("2. 重置环境...")
        obs, info = env.reset()
        print(f"   观察维度: {obs.shape}")
        print(f"   动作空间: {env.action_space}")
        print("   环境重置成功!")
        
        # 运行几步
        print("3. 运行几步测试...")
        for i in range(3):
            action = env.action_space.sample()  # 随机动作
            print(f"   步骤 {i+1}: 执行动作 {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")
            
            if terminated or truncated:
                print("   回合结束，重置环境...")
                obs, info = env.reset()
                break
        
        print("4. 关闭环境...")
        env.close()
        print("✅ 单环境测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 单环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_envs():
    """测试多个环境"""
    print("\n开始测试多个环境...")
    
    try:
        print("1. 创建2个并行环境...")
        envs = gym.vector.AsyncVectorEnv([
            lambda: gym.make("mine/Mine-v1", 
                           config_file="openmines/src/conf/north_pit_mine.json",
                           use_enhanced_observation=True)
            for _ in range(2)
        ])
        print("   多环境创建成功!")
        
        print("2. 重置多环境...")
        obs, info = envs.reset()
        print(f"   观察形状: {obs.shape}")
        print("   多环境重置成功!")
        
        print("3. 运行一步...")
        actions = [envs.single_action_space.sample() for _ in range(2)]
        obs, rewards, terminated, truncated, info = envs.step(actions)
        print(f"   奖励: {rewards}")
        
        print("4. 关闭多环境...")
        envs.close()
        print("✅ 多环境测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 多环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("矿山环境测试")
    print("=" * 50)
    
    # 测试单环境
    single_success = test_single_env()
    
    # 如果单环境成功，测试多环境
    if single_success:
        multi_success = test_multiple_envs()
    else:
        print("\n❌ 单环境测试失败，跳过多环境测试")
        multi_success = False
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"单环境测试: {'✅ 成功' if single_success else '❌ 失败'}")
    print(f"多环境测试: {'✅ 成功' if multi_success else '❌ 失败'}")
    
    if single_success and multi_success:
        print("\n🎉 所有测试通过！可以开始PPO训练。")
    else:
        print("\n⚠️  环境存在问题，需要先解决环境初始化问题。")
