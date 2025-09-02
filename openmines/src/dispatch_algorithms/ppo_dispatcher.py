from __future__ import annotations
import os
import time
import numpy as np
from typing import Optional

import torch
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine
from openmines.src.truck import Truck

# ppo_norm_params_dense.json，每次训练前记得替换
# 导入 rl_dispatch.py 中的 preprocess_observation 函数
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher

class PPODispatcher(BaseDispatcher):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.name = "PPODispatcher"
        
        # 如果指定了模型路径，使用指定的模型
        if model_path is not None:
            self.model_path = model_path
            print(f"使用指定的模型: {model_path}")
        else:
            # 自动查找最新最好的模型
            self.model_path = self._find_latest_best_model()
        
        self.device = self._get_device()
        self.load_rl_model(self.model_path)
        self.rl_dispatcher_helper = RLDispatcher("ShortestTripDispatcher", reward_mode="dense")
        self.max_sim_time = 240

    def _find_latest_best_model(self):
        """自动查找最新的最佳模型文件"""
        # 检查模型目录
        checkpoints_dir = "/home/chengrongxian/git/openmines/checkpoints/mine"
        
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
        
        # 查找所有best_model_开头的.pt文件
        model_files = []
        for root, dirs, files in os.walk(checkpoints_dir):
            for file in files:
                if file.startswith('best_model_') and file.endswith('.pt'):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            raise FileNotFoundError(f"No best model files found in: {checkpoints_dir}")
        
        # 按训练时的吨数排序，选择性能最好的
        model_files.sort(key=lambda x: float(x.split('tons')[1].split('_')[0]), reverse=True)
        best_model = model_files[0]
        
        print(f"自动找到训练时性能最好的模型: {best_model}")
        return best_model

    def _get_device(self):
        """
        确定使用的设备（CUDA/CPU）
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    def load_rl_model(self, model_path: str):
        """
        Load an model for inference.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        from openmines.test.cleanrl.ppo_single_net import Agent, Args
        
        self.args = Args()
        # 此处是我修改的代码，可能会去掉
        self.args.r_mode = "none"  # 避免推理时错误的奖励处理
        self.agent = Agent(envs=None, args=self.args, 
                         norm_path="/home/chengrongxian/git/openmines/datasets/dispatch_data_20250719_122119/normalization_params.json") # 使用训练时的正则化参数
        
        # 加载模型时指定设备映射
        state_dict = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(state_dict)
        self.agent.to(self.device)  # 确保模型在正确的设备上
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
        processed_obs = torch.FloatTensor(
            preprocess_observation(current_observation_raw, self.max_sim_time)
        ).to(self.device)  # 确保输入数据在正确的设备上
        
        with torch.no_grad():  # 推理时不需要梯度
            action, logprob, _, value, _ = self.agent.get_action_and_value(
                processed_obs, sug_action=None # 决策时不需要专家算法
            )        

        return action

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

    dispatcher = PPODispatcher()
    mock_mine = MockMine()
    mock_truck = MockTruck()

    # Example of getting orders:
    init_order = dispatcher.give_init_order(mock_truck, mock_mine)
    haul_order = dispatcher.give_haul_order(mock_truck, mock_mine)
    back_order = dispatcher.give_back_order(mock_truck, mock_mine)

    print(f"Init Order: {init_order}")
    print(f"Haul Order: {haul_order}")
    print(f"Back Order: {back_order}")