#!/usr/bin/env python3
"""
增强观察功能使用示例
演示如何使用包含其他车辆详细信息的增强观察功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher, ObservationConfig
from openmines.src.dispatch_algorithms.ppo_dispatcher import PPODispatcher


def example_basic_usage():
    """基础使用示例（向后兼容）"""
    print("=== 基础模式示例 ===")
    
    # 使用原有功能，无需任何修改
    dispatcher = RLDispatcher("ShortestTripDispatcher", reward_mode="dense")
    print(f"调度器: {dispatcher.name}")
    print(f"使用增强观察: {dispatcher.observation_config.use_enhanced_observation}")
    print(f"最大跟踪车辆数: {dispatcher.observation_config.max_tracked_trucks}")
    print()


def example_enhanced_usage():
    """增强模式示例"""
    print("=== 增强模式示例 ===")
    
    # 使用增强观察功能
    enhanced_config = ObservationConfig.create_enhanced_config()
    enhanced_config.max_tracked_trucks = 15  # 自定义参数
    
    dispatcher = RLDispatcher("ShortestTripDispatcher", reward_mode="dense", 
                             observation_config=enhanced_config)
    print(f"调度器: {dispatcher.name}")
    print(f"使用增强观察: {dispatcher.observation_config.use_enhanced_observation}")
    print(f"最大跟踪车辆数: {dispatcher.observation_config.max_tracked_trucks}")
    print(f"包含车辆位置: {dispatcher.observation_config.include_truck_positions}")
    print(f"包含移动方向: {dispatcher.observation_config.include_movement_directions}")
    print(f"包含ETA预测: {dispatcher.observation_config.include_eta_predictions}")
    print()


def example_ppo_dispatcher():
    """PPO调度器使用示例"""
    print("=== PPO调度器示例 ===")
    
    # 基础PPO调度器
    try:
        ppo_basic = PPODispatcher(use_enhanced_observation=False)
        print(f"基础PPO调度器: {ppo_basic.name}")
        print(f"使用增强观察: {ppo_basic.use_enhanced_observation}")
    except Exception as e:
        print(f"基础PPO调度器创建失败: {e}")
    
    # 增强PPO调度器
    try:
        ppo_enhanced = PPODispatcher(use_enhanced_observation=True)
        print(f"增强PPO调度器: {ppo_enhanced.name}")
        print(f"使用增强观察: {ppo_enhanced.use_enhanced_observation}")
    except Exception as e:
        print(f"增强PPO调度器创建失败: {e}")
    print()


def example_custom_config():
    """自定义配置示例"""
    print("=== 自定义配置示例 ===")
    
    # 创建自定义配置
    custom_config = ObservationConfig()
    custom_config.use_enhanced_observation = True
    custom_config.max_tracked_trucks = 8
    custom_config.include_truck_positions = True
    custom_config.include_movement_directions = True
    custom_config.include_eta_predictions = False  # 不包含ETA预测
    custom_config.include_progress_states = True
    
    dispatcher = RLDispatcher("ShortestTripDispatcher", reward_mode="dense", 
                             observation_config=custom_config)
    
    print(f"调度器: {dispatcher.name}")
    print(f"使用增强观察: {dispatcher.observation_config.use_enhanced_observation}")
    print(f"最大跟踪车辆数: {dispatcher.observation_config.max_tracked_trucks}")
    print(f"包含车辆位置: {dispatcher.observation_config.include_truck_positions}")
    print(f"包含移动方向: {dispatcher.observation_config.include_movement_directions}")
    print(f"包含ETA预测: {dispatcher.observation_config.include_eta_predictions}")
    print(f"包含进度状态: {dispatcher.observation_config.include_progress_states}")
    print()


def example_feature_dimensions():
    """特征维度说明"""
    print("=== 特征维度说明 ===")
    print("基础观察维度: 194维")
    print("  - 订单状态: 6维")
    print("  - 当前卡车: 13维")
    print("  - 道路状态: ~100维")
    print("  - 目标状态: 75维")
    print()
    print("增强观察维度: +190维")
    print("  - 其他车辆信息: 10辆车 × 19维/车 = 190维")
    print("    - 每辆车特征: 位置(11维) + 方向(4维) + 数值(4维) = 19维")
    print()
    print("总维度: 194 + 190 = 384维")
    print()


if __name__ == "__main__":
    print("增强观察功能使用示例\n")
    
    example_basic_usage()
    example_enhanced_usage()
    example_ppo_dispatcher()
    example_custom_config()
    example_feature_dimensions()
    
    print("示例完成！")
    print("\n使用说明:")
    print("1. 基础模式: 与原有代码完全兼容")
    print("2. 增强模式: 提供其他车辆的详细位置和移动信息")
    print("3. 自定义配置: 可以灵活控制包含哪些特征")
    print("4. 自动特征处理: 系统会自动选择合适的处理函数")
