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

# 这个文件是将训练好的AM模型封装成一个调度器，用于实际调度

# 导入 rl_dispatch.py 中的 RLDispatcher
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher

class AMDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "AMDispatcher"
        # 关键修改：指向AM专用检查点目录
        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "mine_AM")
        self.model_path = self._find_latest_best_model()
        self.device = self._get_device()  # 获取可用设备
        self.load_am_model(self.model_path)
        self.rl_dispatcher_helper = RLDispatcher("NaiveDispatcher", reward_mode="dense")        
        self.max_sim_time = 240
    
    def _find_latest_best_model(self):
        """自动查找最新的最佳AM模型文件"""
        if not os.path.exists(self.checkpoints_dir):
            raise FileNotFoundError(f"AM Checkpoints directory not found: {self.checkpoints_dir}")
        
        # 查找所有AM实验目录
        exp_dirs = [d for d in os.listdir(self.checkpoints_dir) 
                   if os.path.isdir(os.path.join(self.checkpoints_dir, d))]
        
        if not exp_dirs:
            raise FileNotFoundError(f"No AM experiment directories found in: {self.checkpoints_dir}")
        
        # 按修改时间排序，选择最新的实验目录
        exp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoints_dir, x)), reverse=True)
        
        # 尝试找到包含最佳模型的最新实验目录
        for exp_dir in exp_dirs:
            latest_exp_dir = os.path.join(self.checkpoints_dir, exp_dir)
            
            # 在实验目录中查找最佳模型文件
            model_files = [f for f in os.listdir(latest_exp_dir) 
                          if f.startswith('best_model_') and f.endswith('.pt')]
            
            if model_files:
                # 按文件名排序，选择最新的最佳模型
                model_files.sort(reverse=True)
                model_path = os.path.join(latest_exp_dir, model_files[0])
                
                print(f"自动找到最佳AM模型:")
                print(f"  实验目录: {exp_dir}")
                print(f"  模型文件: {model_files[0]}")
                print(f"  完整路径: {model_path}")
                return model_path
        
        # 如果没有找到最佳模型，抛出异常
        raise FileNotFoundError(f"No best AM model files found in any experiment directory in: {self.checkpoints_dir}")
        
    def _get_device(self):
        """
        确定使用的设备（CUDA/CPU）
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    def load_am_model(self, model_path: str):
        """
        Load an AM model for inference.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AM Model file not found: {model_path}")
            
        # 导入AM模型类和参数
        from openmines.test.cleanrl.am_single_net import AttentionAgent, Args
        
        # 创建默认参数
        self.args = Args()
        
        # 创建AM智能体
        norm_path = os.path.join(os.path.dirname(__file__), "am_norm_params_dense.json")
        self.agent = AttentionAgent(args=self.args, norm_path=norm_path)
        
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
        Dispatch the truck to the next action based on AM model inference.
        """
        from openmines.src.utils.feature_processing import preprocess_observation 

        # 获取原始观察值
        current_observation_raw = self._get_raw_observation(truck, mine)
        
        # 预处理观察值
        processed_obs = torch.FloatTensor(
            preprocess_observation(current_observation_raw, self.max_sim_time)
        ).to(self.device)  # 确保输入数据在正确的设备上
        
        # 添加批次维度
        if len(processed_obs.shape) == 1:
            processed_obs = processed_obs.unsqueeze(0)
        
        with torch.no_grad():  # 推理时不需要梯度
            # 使用AM模型进行推理
            action, logprob, _, value, _ = self.agent.get_action_and_value(
                processed_obs, sug_action=None
            )        

        # 获取原始动作
        if torch.is_tensor(action):
            if action.dim() > 0:
                raw_action = int(action[0].item())
            else:
                raw_action = int(action.item())
        else:
            raw_action = int(action)
        
        # 将原始动作转换为站点索引
        site_index = self._convert_action_to_site_index(raw_action, truck, mine)
        
        print(f"AM推理: 原始动作={raw_action}, 映射后索引={site_index}, 卡车位置={truck.current_location.name}")
        
        return site_index

    def _get_raw_observation(self, truck: Truck, mine: Mine):
        """
        获取原始的、未经预处理的观察值，直接复用 RLDispatcher 中的 _get_observation 方法
        """
        return self.rl_dispatcher_helper._get_observation(truck, mine)

    def _convert_action_to_site_index(self, action: int, truck: Truck, mine: Mine) -> int:
        """
        将AM模型输出的动作转换为具体的站点索引
        
        AM模型输出的动作范围是0-10，对应11个节点：
        0: 充电站
        1-5: 装载点1-5
        6-10: 卸载点1-5
        
        需要根据卡车当前状态将其转换为实际的站点索引
        """
        # 获取当前卡车状态
        current_location = truck.current_location.name
        
        if "charging" in current_location.lower():
            # 卡车在充电站，需要去装载点 (action 1-5 对应装载点索引 0-4)
            if 1 <= action <= 5:
                return action - 1  # 转换为装载点索引 0-4
            else:
                # 如果动作不合法，默认去第一个装载点
                return 0
                
        elif "load" in current_location.lower():
            # 卡车在装载点，需要去卸载点 (action 6-10 对应卸载点索引 0-4)
            if 6 <= action <= 10:
                return action - 6  # 转换为卸载点索引 0-4
            else:
                # 如果动作不合法，默认去第一个卸载点
                return 0
                
        elif "dump" in current_location.lower():
            # 卡车在卸载点，需要返回装载点或充电站
            if 1 <= action <= 5:
                return action - 1  # 转换为装载点索引 0-4
            else:
                # 默认去第一个装载点
                return 0
        else:
            # 未知状态，默认动作
            return 0

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

    try:
        dispatcher = AMDispatcher()
        mock_mine = MockMine()
        mock_truck = MockTruck()

        # Example of getting orders:
        init_order = dispatcher.give_init_order(mock_truck, mock_mine)
        haul_order = dispatcher.give_haul_order(mock_truck, mock_mine)
        back_order = dispatcher.give_back_order(mock_truck, mock_mine)

        print(f"AM Init Order: {init_order}")
        print(f"AM Haul Order: {haul_order}")
        print(f"AM Back Order: {back_order}")
        
    except FileNotFoundError as e:
        print(f"AM Dispatcher 测试失败: {e}")
        print("请先训练AM模型或确保模型文件存在")
    except Exception as e:
        print(f"AM Dispatcher 遇到错误: {e}") 