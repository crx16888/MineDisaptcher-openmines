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
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher, ObservationConfig

class PPODispatcher(BaseDispatcher):
    def __init__(self, model_path: Optional[str] = None, use_enhanced_observation: bool = True):
        super().__init__()
        self.name = "PPODispatcher"
        
        # 先设置必要的属性
        self.use_enhanced_observation = use_enhanced_observation
        self.max_sim_time = 480
        
        # 创建观察配置（必须在load_rl_model之前，因为需要用于获取观察）
        if use_enhanced_observation:
            observation_config = ObservationConfig.create_enhanced_config()
        else:
            observation_config = ObservationConfig.create_basic_config()
        
        self.rl_dispatcher_helper = RLDispatcher("ShortestTripDispatcher", reward_mode="dense", observation_config=observation_config)
        
        # 如果指定了模型路径，使用指定的模型
        if model_path is not None:
            self.model_path = model_path
            print(f"使用指定的模型: {model_path}")
        else:
            # 自动查找最新最好的模型
            self.model_path = self._find_latest_best_model()
        
        self.device = self._get_device()
        self.load_rl_model(self.model_path)

    def _find_latest_best_model(self):
        """自动查找最新的最佳模型文件"""
        # 检查模型目录 - 使用相对路径，从项目根目录开始
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        checkpoints_base_dir = project_root / "checkpoints"
        
        if not checkpoints_base_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_base_dir}")
        
        # 查找所有可能包含模型的目录，按时间戳排序（目录名中的t后面的数字）
        model_dirs = []
        for item in checkpoints_base_dir.iterdir():
            if item.is_dir() and ('mine' in item.name.lower() or 'ppo' in item.name.lower()):
                model_dirs.append(item)
        
        # 按目录名中的时间戳排序（优先选择最新训练的模型）
        try:
            model_dirs.sort(key=lambda x: int(x.name.split('_t')[-1]) if '_t' in x.name else 0, reverse=True)
        except (IndexError, ValueError):
            # 如果无法按时间戳排序，按目录修改时间排序
            model_dirs.sort(key=lambda x: os.path.getmtime(str(x)), reverse=True)
        
        # 从最新的目录开始查找最佳模型
        model_files = []
        for model_dir in model_dirs:
            dir_models = []
            for root, dirs, files in os.walk(str(model_dir)):
                for file in files:
                    if file.startswith('model_') and file.endswith('.pt'):
                        dir_models.append(os.path.join(root, file))
            
            # 如果该目录有模型，按吨数排序选择最佳的
            if dir_models:
                try:
                    dir_models.sort(key=lambda x: float(x.split('tons')[1].split('_')[0]), reverse=True)
                    best_model = dir_models[0]
                    print(f"自动找到训练时性能最好的模型: {best_model}")
                    return best_model
                except (IndexError, ValueError):
                    # 如果无法按吨数排序，选择最新的文件
                    dir_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    best_model = dir_models[0]
                    print(f"自动找到最新的模型: {best_model}")
                    return best_model
        
        raise FileNotFoundError(f"No model files found in any checkpoints subdirectory: {checkpoints_base_dir}")

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
            
        from openmines.test.cleanrl.ppo_single_net import Agent, Args # 从训练文件中导入agent类，再从训练好的文件中导入agent
        
        self.args = Args()
        # 此处是我修改的代码，可能会去掉
        self.args.r_mode = "none"  # 避免推理时错误的奖励处理
        
        # 动态获取配置文件和正则化参数路径
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        
        # 设置mine_config路径
        self.args.mine_config = str(project_root / "openmines" / "src" / "conf" / "north_pit_mine.json")
        
        # 查找正则化参数文件 - 根据观察维度选择合适的文件
        norm_path = None
        if self.use_enhanced_observation:
            # 474维增强观察：优先查找474维正则化文件
            enhanced_norm_file = project_root / "normalization_params_474.json"
            if enhanced_norm_file.exists():
                norm_path = str(enhanced_norm_file)
                print(f"使用474维增强观察正则化参数: {norm_path}")
        else:
            # 194维基础观察：使用原有的194维正则化文件
            root_norm_file = project_root / "normalization_params.json"
            if root_norm_file.exists():
                norm_path = str(root_norm_file)
                print(f"使用194维基础观察正则化参数: {norm_path}")
        
        # 如果没有找到对应维度的文件，再检查datasets目录
        if norm_path is None:
            datasets_dir = project_root / "datasets"
            if datasets_dir.exists():
                for dataset_folder in datasets_dir.iterdir():
                    if dataset_folder.is_dir():
                        norm_file = dataset_folder / "normalization_params.json"
                        if norm_file.exists():
                            norm_path = str(norm_file)
                            break
        
        if norm_path is None:
            # 如果找不到正则化参数文件，使用默认路径
            norm_path = str(project_root / "datasets" / "dispatch_data_20250719_122119" / "normalization_params.json")
        
        # 由于Agent需要观察空间维度，我们需要创建一个具有正确观察空间的mock环境对象
        class MockEnv:
            def __init__(self, obs_dim):
                self.single_observation_space = type('MockSpace', (), {'shape': (obs_dim,)})()
        
        # 根据use_enhanced_observation设置正确的观察维度
        obs_dim = 474 if self.use_enhanced_observation else 194
        mock_env = MockEnv(obs_dim)
        
        # 创建Agent时传入mock环境
        self.agent = Agent(envs=mock_env, args=self.args, norm_path=norm_path)
        
        # 加载模型时指定设备映射，并设置weights_only=False以兼容旧版模型
        # 确保Args类在全局命名空间中可用
        import sys
        sys.modules['__main__'].Args = Args
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 从checkpoint中提取模型的state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # 如果直接保存的是state_dict
            state_dict = checkpoint
            
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
        from openmines.src.utils.feature_processing import preprocess_observation_auto

        current_observation_raw = self._get_raw_observation(truck, mine) # 去看这个函数就知道，从RLDispatcher这个类里面调的
        processed_obs = torch.FloatTensor(
            preprocess_observation_auto(current_observation_raw, self.max_sim_time)
        ).to(self.device)  # 确保输入数据在正确的设备上
        
        with torch.no_grad():  # 推理时不需要梯度
            action, logprob, _, value, _ = self.agent.get_action_and_value(
                processed_obs, sug_action=None # 决策时不需要专家算法
            )        

        return action

    def _get_raw_observation(self, truck: Truck, mine: Mine):
        """
        获取原始的、未经预处理的观察值，根据配置选择基础或增强观察
        """
        if self.use_enhanced_observation:
            return self.rl_dispatcher_helper._get_enhanced_observation(truck, mine)
        else:
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