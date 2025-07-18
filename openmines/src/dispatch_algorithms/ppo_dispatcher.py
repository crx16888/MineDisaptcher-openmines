from __future__ import annotations
import os
import time
import numpy as np
from typing import Optional
import glob

import torch
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine
from openmines.src.truck import Truck

# 这个文件是将训练好的模型封装成一个调度器，用于实际调度

# 导入 rl_dispatch.py 中的 preprocess_observation 函数
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher

class PPODispatcher(BaseDispatcher):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.name = "PPODispatcher"
        self.checkpoints_dir = r"C:/Users/95718/Desktop/vscode/openmines/models/PPO_models"
        self.device = self._get_device()
        
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = self._find_latest_best_model()

        self.load_rl_model(self.model_path)
        self.rl_dispatcher_helper = RLDispatcher("NaiveDispatcher", reward_mode="dense")
        self.max_sim_time = 240
    
    def _find_latest_best_model(self):
        """自动查找最新的最佳模型文件"""
        if not os.path.exists(self.checkpoints_dir):
            raise FileNotFoundError(f"Checkpoints directory not found: {self.checkpoints_dir}")
        
        # 在指定目录中直接查找最佳模型文件
        model_files = [f for f in os.listdir(self.checkpoints_dir) 
                      if f.startswith('best_model_') and f.endswith('.pt')]
        
        if not model_files:
            raise FileNotFoundError(f"No best model files found in: {self.checkpoints_dir}")
        
        # 按文件名排序（通常包含步数或时间戳），选择最新的最佳模型
        model_files.sort(reverse=True)
        latest_model_file = model_files[0]
        model_path = os.path.join(self.checkpoints_dir, latest_model_file)
        
        print(f"自动找到最佳模型:")
        print(f"  模型目录: {self.checkpoints_dir}")
        print(f"  模型文件: {latest_model_file}")
        print(f"  完整路径: {model_path}")
        return model_path
        
    def _get_device(self):
        """
        确定使用的设备（CUDA/CPU）
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_rl_model(self, model_path: str):
        import __main__
        import openmines.test.cleanrl.ppo_single_net

        __main__.Args = openmines.test.cleanrl.ppo_single_net.Args

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        Args = __main__.Args
        Agent = openmines.test.cleanrl.ppo_single_net.Agent

        self.args = Args()
        # 确保 Agent 初始化时加载归一化参数
        self.agent = Agent(envs=-1, args=self.args, 
                         norm_path=os.path.join(os.path.dirname(__file__), "ppo_norm_params_dense.json"))

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # 从 checkpoint 中提取真正的模型权重
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        self.agent.load_state_dict(model_state_dict)
        self.agent.to(self.device)
        self.agent.eval()

    

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (initial loading).
        """
        return self._dispatch_action(truck, mine)

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (hauling).
        """
        return self._dispatch_action(truck, mine)

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (returning to charging or loading site).
        """
        return self._dispatch_action(truck, mine)

    def _dispatch_action(self, truck: Truck, mine: Mine) -> int:
        """
        Dispatch the truck to the next action based on model inference.
        """
        from openmines.src.utils.feature_processing import preprocess_observation

        current_observation_raw = self._get_raw_observation(truck, mine)
        processed_obs = preprocess_observation(current_observation_raw, self.max_sim_time)

        # 直接将原始数据转换为 tensor
        obs_tensor = torch.tensor(processed_obs, dtype=torch.float32).to(self.device)

        # 让 agent 自己处理归一化，这里传入未经归一化的 obs_tensor
        with torch.no_grad():
            action, logprob, _, value, _ = self.agent.get_action_and_value(
                obs_tensor, sug_action=None
            )

        return int(action.item())

    def _get_raw_observation(self, truck: Truck, mine: Mine):
        """
        获取原始的、未经预处理的观察值，直接复用 RLDispatcher 中的 _get_observation 方法
        """
        return self.rl_dispatcher_helper._get_observation(truck, mine)

# Example usage (for testing - you'd integrate this into your simulation):
if __name__ == "__main__":
    # This is a placeholder for a Mine and Truck object - you need to create
    # actual instances of Mine and Truck as defined in your openmines simulation.
    class MockLocation:
        def __init__(self, name):
            self.name = name
    class MockTruck:
        def __init__(self, name="Truck1", current_location_name="charging_site", truck_load=0, truck_capacity=40, truck_speed=40):
            self.name = name
            self.current_location = MockLocation(current_location_name)
            self.truck_load = truck_load
            self.truck_capacity = truck_capacity
            self.truck_speed = truck_speed
            self.truck_cycle_time = 0

        def get_status(self):
            return {} # Placeholder

    class MockMine:
        def __init__(self):
            self.env = MockEnv()
            self.load_sites = [MockLocation("load_site_1"), MockLocation("load_site_2")]
            self.dump_sites = [MockLocation("dump_site_1"), MockLocation("dump_site_2")]

        def get_status(self):
            return {} # Placeholder
    class MockEnv:
        def __init__(self):
            self.now = 10.0


    # dispatcher = PPODispatcher() # 默认使用最新最佳模型
    dispatcher = PPODispatcher(model_path="/home/chengrongxian/git/openmines/models/PPO_models/best_model_step00000500_tons7244.6_reward0.00.pt") # 只训了500步
    mock_mine = MockMine()
    mock_truck = MockTruck()

    # Example of getting orders:
    init_order = dispatcher.give_init_order(mock_truck, mock_mine)
    haul_order = dispatcher.give_haul_order(mock_truck, mock_mine)
    back_order = dispatcher.give_back_order(mock_truck, mock_mine)

    print(f"Init Order: {init_order}")
    print(f"Haul Order: {haul_order}")
    print(f"Back Order: {back_order}")