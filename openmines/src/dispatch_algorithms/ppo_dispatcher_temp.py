from __future__ import annotations
import os
import time
import numpy as np
from typing import Optional

import torch
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine
from openmines.src.truck import Truck

# 导入 rl_dispatch.py 中的 preprocess_observation 函数
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher

class PPODispatcherTemp(BaseDispatcher):
    """
    专门使用指定模型的PPO调度器
    模型：best_model_step0070000_tons7340.5_reward0.00.pt
    """
    def __init__(self):
        super().__init__()
        self.name = "PPODispatcherTemp"
        
        # 指定使用特定的模型
        self.model_path = "/home/chengrongxian/git/openmines/checkpoints/mine/Mine-v1__ppo_single_net__s1__lr2.28e-03__e1.43e-02__g0.997__c0.20__l0.993__ep2__gr0.36__hs256__ns1400__ne50__mb4__rmreward_norm__t1753892143/best_model_step0070000_tons8118.8_reward0.00.pt"
        print(f"使用指定模型: {self.model_path}")
        print(f"模型信息: step70000, 性能: 7340.5吨")
        
        self.device = self._get_device()
        self.load_rl_model(self.model_path)
        self.rl_dispatcher_helper = RLDispatcher("ShortestTripDispatcher", reward_mode="dense")
        self.max_sim_time = 240

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
        # 此处是我修改的代码,推理时根本不需要对奖励归一化，只需要对状态正则化就好了
        self.args.r_mode = "none"  # 避免推理时错误的奖励处理
        self.agent = Agent(envs=-1, args=self.args, 
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

    dispatcher = PPODispatcherTemp()
    mock_mine = MockMine()
    mock_truck = MockTruck()

    # Example of getting orders:
    init_order = dispatcher.give_init_order(mock_truck, mock_mine)
    haul_order = dispatcher.give_haul_order(mock_truck, mock_mine)
    back_order = dispatcher.give_back_order(mock_truck, mock_mine)

    print(f"Init Order: {init_order}")
    print(f"Haul Order: {haul_order}")
    print(f"Back Order: {back_order}")