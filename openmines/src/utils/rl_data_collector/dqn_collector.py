import os
import json
import time
import numpy as np
import gymnasium as gym
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from multiprocessing import Queue
import argparse
from openmines.src.utils.feature_processing import preprocess_observation_auto
import pathlib

# 导入环境以注册到gymnasium
import openmines.src.utils.gym.openmines_gym

# 这个脚本用于在不同调度器（dispatcher）配置下运行自定义 gym 环境（openmines 的 Mine 环境）若干回合，
# 从 environment 的 info（包含专家建议动作 sug_action 等）收集状态、动作、奖励与若干元信息，
# 把算有算法的数据汇总成normalization_params.json文件，并计算所有算法的状态/奖励的标准化参数（均值、方差），
# 同时为每个调度器保存单独的性能指标文件（metrics_*.json）

# ppo_norm_params_dense.json可以看出总样本数确实只有5288个，而且奖励标准差(15.29)相对于均值(2.80)来说比较大，
# 说明奖励分布的方差较大，这种情况下更需要大量样本来准确估计分布参数。
# 如增加数据收集量 ：将episodes从3增加到至少50-100个

class NumpyEncoder(json.JSONEncoder):
    """处理numpy数据类型的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DataCollector:
    def __init__(self, env_config, episodes=100, max_steps=1000, env_id="mine/Mine-v1", use_enhanced_observation=True):
        """初始化数据收集器
        
        Args:
            use_enhanced_observation: 是否使用384维增强观察（默认True）
        """
        self.env_config = env_config
        self.episodes = episodes
        self.max_steps = max_steps
        self.env_id = env_id
        self.use_enhanced_observation = use_enhanced_observation
        self.dataset = []
        self.all_states = []
        self.all_rewards = []  # 添加rewards列表用于存储所有奖励值
        
        # 读取配置文件
        with open(env_config, 'r') as f:
            self.config = json.load(f)
        
        # 根据配置设置观察维度（预处理后的实际维度）
        # 增强观察: 96基础维度 + (车辆数-1)*4维
        if use_enhanced_observation:
            # 动态计算车辆数
            self.total_trucks = sum(t['count'] for t in self.config['charging_site']['trucks'])
            other_trucks_dim = (self.total_trucks - 1) * 4
            self.obs_dim = 96 + other_trucks_dim  # 预处理后的实际维度
        else:
            self.total_trucks = sum(t['count'] for t in self.config['charging_site']['trucks'])
            self.obs_dim = 96  # 预处理后的实际维度
        
        # 获取配置文件名（不含路径和扩展名）
        self.config_name = pathlib.Path(env_config).stem
        
        print(f"数据收集器配置: {self.config_name} | {self.total_trucks}辆车 | {self.obs_dim}维{'增强观察' if use_enhanced_observation else '基础观察'}")
        
        self.dispatchers = self.config['dispatcher']['type']
        self.sim_time = self.config['sim_time']  # 获取模拟时间
        
        # 生成唯一的数据集ID
        self.run_id = self._generate_run_id()
        
        # 创建输出目录
        self.output_dir = os.path.join("datasets", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_run_id(self):
        """生成唯一的运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dispatch_data_{timestamp}"

    def collect_data(self):
        """对配置文件下的每个调度器收集调度决策数据；通常选择产量最高、最稳定的算法数据作为对比使用
        这只是收集专家数据用于所有rl算法的归一化（加速训练），而在真实决策时使用的是专家算法而不是专家数据
        通常模仿学习是通过收集专家数据训练一个专家模型，而此处我们已经有了专家算法"""
        for dispatcher in tqdm(self.dispatchers, desc="Processing dispatchers"):
            print(f"\n收集调度器 {dispatcher} 的数据...")
            
            # 更新环境配置中的调度器
            with open(self.env_config, 'r') as f:
                config = json.load(f)
            config['dispatcher']['type'] = [dispatcher]
            
            # 创建临时配置文件
            temp_config_path = os.path.join(self.output_dir, f"temp_{dispatcher}_config.json")
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            total_samples = 0
            metrics = defaultdict(list)
            
            # 使用gym.make创建环境，而不是直接使用MineEnv
            try:
                # 创建环境时传入配置文件和增强观察配置
                env = gym.make(
                    self.env_id, 
                    config_file=temp_config_path,
                    use_enhanced_observation=self.use_enhanced_observation
                ) # 默认是ShortestTripDispatcher算法作为指导算法
                
                for episode in tqdm(range(self.episodes), desc=f"Collecting episodes for {dispatcher}"):
                    observation, info = env.reset(seed=episode)
                    
                    episode_samples = 0
                    episode_reward = 0
                    last_production = 0
                    
                    for step in range(self.max_steps):
                        expert_action = info.get("sug_action", 0)  # 使用info中的建议动作
                        
                        # 直接使用原始观察数据（不预处理）
                        # PPO训练时会在Agent内部进行归一化处理
                        if isinstance(observation, dict):
                            # 如果是字典格式，提取state
                            state = observation.get('state', observation)
                            if isinstance(state, dict):
                                # 如果state还是字典，尝试提取observation
                                state = state.get('observation', state)
                        else:
                            # 如果是数组格式，直接使用
                            state = observation
                        
                        # 确保state是numpy数组
                        if not isinstance(state, np.ndarray):
                            state = np.array(state)
                        
                        next_observation, reward, done, truncated, info = env.step(expert_action)
                        self.all_rewards.append(reward)  # 收集reward
                        
                        # 获取产出信息
                        current_production = float(info.get('produce_tons', 0.0))
                        production_increase = current_production - last_production
                        last_production = current_production
                        
                        # 收集样本数据
                        data_sample = {
                            'dispatcher': dispatcher,
                            'episode': int(episode),
                            'step': int(step),
                            'state': [float(x) for x in state],
                            'action': int(expert_action),
                            'reward': float(reward),
                            'event_type': info.get('event_name', 'unknown'),
                            'delta_time': float(info.get('delta_time', 0.0)),
                            'location': info.get('truck_location', 'unknown'),
                            'truck_name': info.get('truck_name', 'unknown')
                        }
                        
                        self.dataset.append(data_sample)
                        self.all_states.append(state)
                        episode_samples += 1
                        
                        episode_reward += reward
                        
                        if done or truncated:
                            break
                            
                        observation = next_observation
                    
                    metrics['episode_samples'].append(int(episode_samples))
                    metrics['episode_rewards'].append(float(episode_reward))
                    metrics['total_production'].append(float(last_production))
                    total_samples += episode_samples
                
                env.close()
                
            except Exception as e:
                print(f"环境创建或运行时出错: {str(e)}")
                raise
            
            # 为每个调度器保存单独的指标
            dispatcher_metrics_path = os.path.join(self.output_dir, f"metrics_{dispatcher}.json")
            metrics_summary = {
                'dispatcher': dispatcher,
                'total_samples': int(total_samples),
                'total_episodes': int(self.episodes),
                'avg_samples_per_episode': float(np.mean(metrics['episode_samples'])),
                'avg_reward_per_episode': float(np.mean(metrics['episode_rewards'])),
                'avg_production_per_episode': float(np.mean(metrics['total_production'])),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(dispatcher_metrics_path, 'w') as f:
                json.dump(metrics_summary, f, indent=4, cls=NumpyEncoder)
            
            # 删除临时配置文件
            os.remove(temp_config_path)
        
        # 保存完整数据集
        self._save_dataset(len(self.dataset), metrics)

    def preprocess_features(self, observation):
        """使用导入的preprocess_observation_auto函数处理特征"""
        if hasattr(self, 'sim_time') and self.sim_time:
            return preprocess_observation_auto(observation, self.sim_time)
        else:
            # 尝试从observation中获取特征
            if isinstance(observation, dict) and 'state' in observation:
                return observation['state']
            return observation

    def _save_dataset(self, total_samples, metrics):
        """保存数据集和计算标准化参数"""
        # 保存数据集
        dataset_path = os.path.join(self.output_dir, "dispatch_dataset.jsonl")
        with open(dataset_path, 'w') as f:
            for sample in self.dataset:
                json.dump(sample, f, cls=NumpyEncoder)
                f.write('\n')
        
        # 计算并保存标准化参数
        states_array = np.array(self.all_states)
        rewards_array = np.array(self.all_rewards)
        
        state_mean = np.mean(states_array, axis=0)
        state_std = np.std(states_array, axis=0)
        reward_mean = np.mean(rewards_array)
        reward_std = np.std(rewards_array)
        
        normalization_params = {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "feature_dims": len(state_mean),
            "total_samples": total_samples,
            "dispatchers": self.dispatchers,
            "env_id": self.env_id,
            # 添加配置元数据
            "config_name": self.config_name,
            "total_trucks": self.total_trucks,
            "obs_dim": self.obs_dim,
            "use_enhanced_observation": self.use_enhanced_observation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 1. 保存到数据集目录
        params_path = os.path.join(self.output_dir, "normalization_params.json")
        with open(params_path, 'w') as f:
            json.dump(normalization_params, f, indent=4, cls=NumpyEncoder)
        
        # 2. 保存到项目根目录，文件名包含维度信息
        obs_mode = "enh" if self.use_enhanced_observation else "bas"
        root_params_filename = f"normalization_params_{self.config_name}_{self.total_trucks}t_{obs_mode}{self.obs_dim}d.json"
        root_params_path = pathlib.Path.cwd() / root_params_filename
        with open(root_params_path, 'w') as f:
            json.dump(normalization_params, f, indent=4, cls=NumpyEncoder)
        print(f"\n正则化参数已保存:")
        print(f"  数据集目录: {params_path}")
        print(f"  项目根目录: {root_params_path}")
        
        # 3. 同时保存一个通用名称的文件（用于默认使用）
        default_params_path = pathlib.Path.cwd() / "normalization_params.json"
        with open(default_params_path, 'w') as f:
            json.dump(normalization_params, f, indent=4, cls=NumpyEncoder)
        print(f"  默认文件: {default_params_path}")
        
        # 4. 复制配置文件到数据集目录
        import shutil
        config_copy_path = os.path.join(self.output_dir, "training_config.json")
        shutil.copy(self.env_config, config_copy_path)
        print(f"  配置文件副本: {config_copy_path}")
        
        print(f"\n数据收集完成!")
        print(f"总样本数: {total_samples}")
        print(f"数据集保存至: {dataset_path}")
        print(f"标准化参数保存至: {params_path}")
        print(f"\n特征统计信息:")
        print(f"特征维度: {len(state_mean)}")
        print(f"状态均值范围: [{min(state_mean):.3f}, {max(state_mean):.3f}]")
        print(f"状态标准差范围: [{min(state_std):.3f}, {max(state_std):.3f}]")
        print(f"奖励均值: {reward_mean:.3f}")
        print(f"奖励标准差: {reward_std:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect dispatch decision data")
    # parser.add_argument("--env_config", type=str,
    #                     default="/home/chengrongxian/git/openmines0/openmines/src/conf/north_pit_mine.json",
    #                     help="环境配置文件路径")
    parser.add_argument("--env_config", type=str,
                        default="src/conf/north_pit_mine.json",
                        help="环境配置文件路径")
    parser.add_argument("--episodes", type=int, default=50,
                        help="收集数据的回合数")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="每个回合的最大步数")
    parser.add_argument("--env_id", type=str, default="mine/Mine-v1-dense",
                        help="环境ID")
    parser.add_argument("--use_enhanced_observation", action="store_true", default=True,
                        help="是否使用增强观察（跟踪所有车辆，默认True）")
    parser.add_argument("--use_basic_observation", dest="use_enhanced_observation", action="store_false",
                        help="使用194维基础观察")

    args = parser.parse_args()

    collector = DataCollector(
        env_config=args.env_config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        env_id=args.env_id,
        use_enhanced_observation=args.use_enhanced_observation
    )
    collector.collect_data()