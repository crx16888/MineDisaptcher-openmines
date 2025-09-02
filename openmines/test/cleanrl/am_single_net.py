# am_single_net.py
# ------------------------------------------------
# Attention Model (AM) for Mine Truck Dispatch
# 基于论文 "Attention, Learn to Solve Routing Problems!"
# 使用 Transformer + Pointer Network 架构
# 训练方法：REINFORCE with greedy rollout baseline
# ------------------------------------------------

import os
import random
import time
import math
from dataclasses import dataclass
from typing import Optional, Dict
import json
import datetime
from pathlib import Path
import subprocess
import tempfile
import shutil
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# 感觉后面可以给调试信息关了，占速度
@dataclass
class Args:
    # 基础参数
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # 算法相关参数
    env_id: str = "mine/Mine-v1"
    mine_config: str = "/home/chengrongxian/git/openmines/openmines/src/conf/north_pit_mine.json"
    total_timesteps: int = 2000000 # 总训练的步数，智能体与环境交互的总次数

    # AM算法超参数
    learning_rate: float = 5e-4  # 提高学习率但保持在合理范围
    gamma: float = 0.997        # 使用与PPO相同的折扣因子
    baseline_lr_decay: float = 0.993  # 使用与PPO相同的学习率衰减
    
    # 网络架构参数
    embed_dim: int = 128     # 保持不变，这是Transformer的标准配置
    hidden_dim: int = 256    # 与PPO保持一致
    num_heads: int = 8       # 保持不变，这是标准配置
    num_layers: int = 3      # 保持不变，避免模型过于复杂
    ff_dim: int = 2048      # 保持不变，这是标准配置
    dropout: float = 0.1     # 保持不变，这是标准配置

    # 训练参数
    num_envs: int = 4       # 增加并行环境数量
    num_steps: int = 1000   # 增加每次更新的步数
    batch_size: int = 256   # 增加批次大小
    baseline_epochs: int = 2 # 增加基线训练轮数
    max_grad_norm: float = 0.5  # 降低梯度裁剪阈值，使训练更稳定

    # 检查点相关 - 使用AM专用目录
    checkpoint_dir: str = "/home/chengrongxian/git/openmines/checkpoints"
    save_interval: int = 10 # 保存间隔,意味着每几个iteration保存模型
    keep_checkpoint_max: int = 5 # 保留的最大检查点数量
    checkpoint_path: Optional[str] = None
    save_best_only: bool = True # 是否只保存最佳模型
    save_best_only_params: bool = True # 是否只保存最佳模型参数

    # 网络相关
    r_mode: str = "reward_norm" # 奖励模式
    norm_path: Optional[str] = "/home/chengrongxian/git/openmines/openmines/src/dispatch_algorithms/am_norm_params_dense.json" # 正则化参数路径


# 创建仿真环境，有点类似ppo，不过此处没有同时创建好几个环境
def make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        # 训练时使用简单的调度器作为建议动作，避免循环依赖
        sug_dispatcher = "ShortestTripDispatcher"  # 训练时使用固定的简单调度器
        
        print(f"环境创建: 使用配置文件 {args.mine_config}")
        print(f"环境创建: 建议调度器 {sug_dispatcher} (训练模式)")
        
        if capture_video and idx == 0:
            env = gym.make(env_id, config_file=args.mine_config, 
                          sug_dispatcher=sug_dispatcher, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, config_file=args.mine_config, 
                          sug_dispatcher=sug_dispatcher)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

# 多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out), attn_weights

# 标准transformer层，包含：- 多头意力层 + 残差连接
# - 前馈网络 + 残差连接
# - Layer Normalization
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.self_attn(x, x, x, mask) # x同时作为qkv先送进去多头注意力层，输出attn_out
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class AttentionEncoder(nn.Module): # 编码器同时嵌了好几个transformer，用于生成对于特征的更精确表示
    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x

# 指针解码器，用于输出最终决策结果
class PointerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 上下文编码
        self.context_embedding = nn.Linear(embed_dim * 3, embed_dim)  # [graph, first, last]
        
        # 指针注意力
        self.pointer_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.scale = 10.0  # 用于温度缩放
        
    def forward(self, encoder_outputs, context_vector, mask=None):
        """
        encoder_outputs: [batch_size, seq_len, embed_dim] - 编码器输出
        context_vector: [batch_size, embed_dim] - 上下文向量
        mask: [batch_size, seq_len] - 掩码，1表示可选，0表示不可选
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # 扩展上下文向量用作查询
        query = context_vector.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 使用注意力机制计算指针分布
        attn_output, attn_weights = self.pointer_attn(
            query, encoder_outputs, encoder_outputs
        )
        
        # 提取注意力权重并缩放
        logits = attn_weights.squeeze(1) * self.scale  # [batch_size, seq_len]
        
        # 应用掩码
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        
        return logits

# 注意力智能体类
class AttentionAgent(nn.Module):
    def __init__(self, args, norm_path=None):
        super().__init__()
        self.args = args
        
        # 加载正则化文件，此处应该和ppo一样，但我还不确定它能否加载专家算法一起计算出来的正则化文件
        # 可能没有写读取正则化参数的代码
        if norm_path is None:
            norm_path = manage_normalization_params(args=args)
        print(f"正在读取AM正则化参数文件: {norm_path}")
        
        try:
            with open(norm_path, "r") as f:
                normalization_params = json.load(f)
        except FileNotFoundError:
            # 如果文件不存在，使用默认维度创建正则化参数
            print(f"AM正则化参数文件不存在，创建默认参数: {norm_path}")
            default_obs_shape = 204  # 使用默认维度
            normalization_params = {
                "state_mean": [0.0] * default_obs_shape,
                "state_std": [1.0] * default_obs_shape,
                "reward_mean": 0.0,
                "reward_std": 1.0
            }
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            with open(norm_path, "w") as f:
                json.dump(normalization_params, f)
        
        # 从正则化参数中获取观察空间维度
        self.obs_shape = len(normalization_params["state_mean"])
        
        # 网络维度配置 - 从配置文件动态计算节点数量
        self.num_nodes = self._calculate_num_nodes(args.mine_config)  # 充电站(1) + 装载点(动态) + 卸载点(动态)
        self.node_feature_dim = self.obs_shape // self.num_nodes + 1  # 每个节点的特征维度
        # 节点的特征维度由状态信息维度除以节点数量得到，+1是因为每个节点还有个位置信息
        # 假设 obs_shape = 194，那么每个节点的特征维度就是 18 维（194/11 + 1）。
        # 对于一个装载点节点来说，这18维特征可能包含：
        # 位置信息（x, y坐标）：2维
        # 当前排队车辆数：1维
        # 装载能力/效率：1维
        # 当前状态（是否可用）：1维
        # 与其他站点的距离：10维（与其他站点的距离）
        # 历史统计信息（平均等待时间等）：2维
        # 其他相关属性：1维
        # 这样加起来就是18维的特征向量，用来描述这个节点（站点）的完整状态。每个节点（无论是充电站、装载点还是卸载点）都会被编码成相同维度（18维）的特征向量，这样便于神经网络处理。
        
        print(f"AM模型观察空间维度: {self.obs_shape}")
        print(f"AM模型节点特征维度: {self.node_feature_dim}")
        
        # 节点嵌入
        self.node_embedding = nn.Linear(self.node_feature_dim, args.embed_dim) # 这里只是定义线性层的形状，实际输入不只是维度信息更有实际的状态值信息，在底下forward里有
        
        # Transformer编码器
        self.encoder = AttentionEncoder(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout
        )
        
        # 指针解码器
        self.decoder = PointerDecoder(args.embed_dim, args.num_heads)
        
        # 价值网络，输入是矿山状态和矿车动作，输出是评估价值
        self.value_head = nn.Sequential(
            nn.Linear(args.embed_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
        # 注册正则化参数
        self.register_buffer(
            "obs_mean",
            torch.tensor(normalization_params["state_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "obs_std", 
            torch.tensor(normalization_params["state_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "reward_mean",
            torch.tensor(normalization_params["reward_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "reward_std",
            torch.tensor(normalization_params["reward_std"], dtype=torch.float32)
        )
        
        # 数值稳定性
        self.obs_std[self.obs_std < 1e-5] = 1.0
        if self.reward_std < 1e-5:
            self.reward_std = torch.tensor(1.0, device=self.reward_std.device)
    
    def _calculate_num_nodes(self, mine_config_path):
        """从配置文件动态计算节点数量"""
        import json
        
        try:
            with open(mine_config_path, 'r') as f:
                config = json.load(f)
            
            self.num_load_sites = len(config.get('load_sites', []))
            self.num_dump_sites = len(config.get('dump_sites', []))
            self.num_charging_sites = 1  # 假设只有一个充电站
            
            total_nodes = self.num_charging_sites + self.num_load_sites + self.num_dump_sites
            
            print(f"AM模型节点配置: 充电站={self.num_charging_sites}, 装载点={self.num_load_sites}, 卸载点={self.num_dump_sites}, 总计={total_nodes}")
            
            return total_nodes
            
        except Exception as e:
            print(f"警告：读取配置文件失败，使用默认节点数量11。错误：{e}")
            # 设置默认值
            self.num_load_sites = 5
            self.num_dump_sites = 5
            self.num_charging_sites = 1
            return 11  # 使用默认值作为后备

    def normalize_obs(self, obs):
        """对观察值进行正则化"""
        return (obs - self.obs_mean) / self.obs_std
    
    def normalize_reward(self, reward):
        """对奖励进行正则化"""
        if self.args.r_mode == "reward_norm":
            return (reward - self.reward_mean) / self.reward_std
        return reward
    
    def obs_to_graph(self, obs):
        # 矿山本就是一个图结构，转换后更能表示站点间的连接关系和车辆的移动可能性
        # 相当于数据结构的实体变成了nodes、edges等，graph
        """将观察值转换为图结构表示"""
        batch_size = obs.shape[0]
        obs_dim = obs.shape[1]  # 194
        
        # 计算每个节点应该分配的特征数量
        features_per_node = obs_dim // self.num_nodes  # 194 // 11 = 17
        remaining_features = obs_dim % self.num_nodes   # 194 % 11 = 7
        
        # 将观察空间分配给各个节点
        node_features_list = []
        start_idx = 0
        
        for i in range(self.num_nodes):
            # 为前几个节点分配额外的特征
            current_features = features_per_node + (1 if i < remaining_features else 0)
            end_idx = start_idx + current_features
            
            # 提取当前节点的特征
            node_feat = obs[:, start_idx:end_idx]  # [batch_size, current_features]
            
            # 如果特征维度不够，进行填充
            if node_feat.shape[1] < self.node_feature_dim:
                padding = torch.zeros(batch_size, 
                                    self.node_feature_dim - node_feat.shape[1],
                                    device=obs.device)
                node_feat = torch.cat([node_feat, padding], dim=1)
            # 如果特征维度过多，进行截断
            elif node_feat.shape[1] > self.node_feature_dim:
                node_feat = node_feat[:, :self.node_feature_dim]
            
            node_features_list.append(node_feat.unsqueeze(1))  # [batch_size, 1, node_feature_dim]
            start_idx = end_idx
        
        # 堆叠所有节点特征
        node_features = torch.cat(node_features_list, dim=1)  # [batch_size, num_nodes, node_feature_dim]
        
        return node_features
    
    def get_action_mask(self, obs, current_state):
        """
        生成动作掩码，确保只选择合法动作
        节点布局（动态）：
        0: 充电站
        1 到 num_load_sites: 装载点
        num_load_sites+1 到 num_load_sites+num_dump_sites: 卸载点
        """
        batch_size = obs.shape[0]
        mask = torch.zeros(batch_size, self.num_nodes, device=obs.device)
        
        # 严格限制动作范围，确保不会超出配置的站点数量
        # 从充电站出发 -> 只能去装载点
        load_start = 1
        load_end = min(1 + self.num_load_sites, self.num_nodes)  # 确保不超出节点总数
        if load_end > load_start:
            mask[:, load_start:load_end] = 1  # 装载点动作
        
        # 从装载点出发 -> 只能去卸载点  
        # 从卸载点出发 -> 只能去装载点
        # 由于无法从obs直接判断当前位置，我们允许所有合法动作
        # 实际的过滤会在RLDispatcher中进行
        dump_start = 1 + self.num_load_sites
        dump_end = min(dump_start + self.num_dump_sites, self.num_nodes)  # 确保不超出节点总数
        if dump_end > dump_start and dump_start < self.num_nodes:
            mask[:, dump_start:dump_end] = 1  # 卸载点动作
        
        # 确保至少有一个有效动作（安全措施）
        if mask.sum() == 0:
            # 如果没有有效动作，默认允许第一个装载点
            if self.num_load_sites > 0:
                mask[:, 1] = 1
            else:
                mask[:, 0] = 1  # 最后的安全措施
        
        return mask
    
    def forward(self, obs, action=None):
        """
        输入 obs: [batch_size, 194] 的张量，包含完整的矿山状态信息：
        
        1. 当前待调度的这一个车辆的状态信息 [0:5]：
            - truck_location_index: 车辆当前位置（充电站=0，装载点=1-5，卸载点=6-10）
            - truck_load: 当前载重量
            - truck_capacity: 车辆最大容量
            - truck_cycle_time: 完成一个装卸循环的时间
            - truck_speed: 车辆行驶速度
        
        2. 目标站点状态 [5:45]：
            对于每个站点（11个站点：1充电站 + 5装载点 + 5卸载点）：
            - queue_lengths: 各站点排队车辆数
            - capacities: 站点处理能力
            - est_wait: 预估等待时间
            - produced_tons: 已处理的矿石量
            - service_counts: 服务车辆次数
        
        3. 道路网络状态 [45:155]：
            对于每条道路（总共110条：5*5*2 + 5条路）：
            - oh_truck_count: 道路上的车辆数
            - oh_distances: 道路长度
            - oh_truck_jam_count: 道路拥堵程度
            - oh_repair_count: 道路维修次数
        
        4. 事件信息 [155:158]：
            - one-hot编码的事件类型：
              [1,0,0]: 初始化事件
              [0,1,0]: 装载事件
              [0,0,1]: 卸载事件
        
        5. 矿山整体状态 [158:194]：
            - truck_count: 系统中的车辆总数
            - total_production: 总产量
            - 其他全局统计信息
        """
        obs = self.normalize_obs(obs)  # 对所有特征进行标准化处理，使其均值为0，标准差为1
        # 不知道有没有正确读取正则化参数文件

        
        node_features = self.obs_to_graph(obs) # 转换为图结构，矿山模型本就是图结构表示，更利于网络处理
        # 将194维向量重组为 [batch_size, 11, 18] 的张量
        # 11：表示11个节点（1充电站 + 5装载点 + 5卸载点）
        # 18：每个节点的特征维度，包含该节点的所有相关信息
        
        # 节点嵌入
        embedded_nodes = self.node_embedding(node_features) # 节点嵌入，将每个节点的特征向量转换为嵌入向量
        
        # 编码器
        encoded_nodes = self.encoder(embedded_nodes)
        
        # 图级表示（平均池化）
        graph_embedding = encoded_nodes.mean(dim=1)
        
        # 生成动作掩码
        action_mask = self.get_action_mask(obs, None)
        
        # 解码器 - 生成动作分布
        logits = self.decoder(encoded_nodes, graph_embedding, action_mask)
        
        # 计算价值
        value = self.value_head(graph_embedding)
        
        # 创建动作分布
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample() # 根据动作分布选择动作，并且返回这次决策的价值
        
        # 安全检查：确保动作在有效范围内
        valid_actions = self._validate_actions(action)
        
        return valid_actions, probs.log_prob(valid_actions), probs.entropy(), value.squeeze(-1)

    def get_action_and_value(self, obs, action=None, sug_action=None):
        """兼容接口方法"""
        act, log_prob, entropy, value = self.forward(obs, action) # 使用模型的决策方法获取动作和价值
        
        sug_logprob = None
        if sug_action is not None:
            # 处理建议动作的log概率
            valid_mask = (sug_action >= 0)
            if valid_mask.any():
                # 重新计算用于建议动作
                _, _, _, _ = self.forward(obs)
                # 这里简化处理，实际需要计算建议动作的log概率
                sug_logprob = log_prob  # 占位符
        
        return act, log_prob, entropy, value, sug_logprob

    def get_value(self, obs): # 连续经过多个网络返回状态价值
        """获取状态价值"""
        obs = self.normalize_obs(obs)
        node_features = self.obs_to_graph(obs)
        embedded_nodes = self.node_embedding(node_features)
        encoded_nodes = self.encoder(embedded_nodes)
        graph_embedding = encoded_nodes.mean(dim=1)
        return self.value_head(graph_embedding).squeeze(-1)

    def _validate_actions(self, actions):
        """
        验证并修正动作，确保不会超出有效范围
        """
        # 计算有效的动作范围
        max_valid_action = self.num_nodes - 1
        
        # 将超出范围的动作限制在有效范围内
        valid_actions = torch.clamp(actions, 0, max_valid_action)
        
        # 进一步检查：确保动作对应有效的站点
        batch_size = actions.shape[0]
        for i in range(batch_size):
            action_val = valid_actions[i].item()
            
            # 检查是否是有效的装载点或卸载点
            if action_val == 0:  # 充电站
                continue
            elif 1 <= action_val <= self.num_load_sites:  # 装载点
                continue
            elif (1 + self.num_load_sites) <= action_val <= (self.num_load_sites + self.num_dump_sites):  # 卸载点
                continue
            else:
                # 无效动作，重定向到第一个装载点
                if self.num_load_sites > 0:
                    valid_actions[i] = 1
                else:
                    valid_actions[i] = 0
        
        return valid_actions
    
    def _convert_neural_action_to_env_action(self, neural_action):
        """
        将神经网络输出的动作转换为环境可接受的站点索引
        
        神经网络动作布局：
        0: 充电站
        1 到 num_load_sites: 装载点节点
        (num_load_sites+1) 到 (num_load_sites+num_dump_sites): 卸载点节点
        
        环境期望的站点索引：
        装载点索引: 0 到 (num_load_sites-1)
        卸载点索引: 0 到 (num_dump_sites-1)
        """
        if isinstance(neural_action, torch.Tensor):
            if neural_action.dim() > 0:
                neural_action = neural_action.cpu().numpy()
            else:
                neural_action = neural_action.item()
        
        # 处理批次或单个动作
        if isinstance(neural_action, np.ndarray):
            env_actions = []
            for action in neural_action:
                env_action = self._single_neural_to_env_action(int(action))
                env_actions.append(env_action)
            return np.array(env_actions)
        else:
            return self._single_neural_to_env_action(int(neural_action))
    
    def _single_neural_to_env_action(self, action: int) -> int:
        """将单个神经网络动作转换为环境站点索引"""
        if action == 0:  # 充电站
            # 充电站动作通常不会直接传递给环境，但为了安全返回0
            return 0
        elif 1 <= action <= self.num_load_sites:  # 装载点节点
            # 转换为装载点索引：1,2,3,4,5 -> 0,1,2,3,4
            env_index = action - 1
            # 边界检查
            if env_index >= self.num_load_sites:
                print(f"警告：装载点索引 {env_index} 超出范围，使用默认值0")
                return 0
            return env_index
        elif (1 + self.num_load_sites) <= action <= (self.num_load_sites + self.num_dump_sites):  # 卸载点节点
            # 转换为卸载点索引：6,7,8,9,10 -> 0,1,2,3,4
            env_index = action - 1 - self.num_load_sites
            # 边界检查
            if env_index >= self.num_dump_sites:
                print(f"警告：卸载点索引 {env_index} 超出范围，使用默认值0")
                return 0
            return env_index
        else:
            # 无效动作，返回默认装载点索引
            print(f"警告：无效的神经网络动作 {action}，使用默认值0")
            return 0
    
    @property
    def device(self):
        return next(self.parameters()).device


class CheckpointManager:
    def __init__(self, args: Args, exp_name: str):
        self.args = args
        self.exp_name = exp_name
        self.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        self.best_reward = float('-inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                        agent: nn.Module,
                        optimizer: optim.Optimizer,
                        iteration: int,
                        reward: float,
                        is_best: bool = False,
                        additional_info: Dict = None) -> str:
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': reward,
            'args': self.args,
            'info': additional_info or {}
        }

        produce_tons = additional_info.get('produce_tons', 0.0) if additional_info else 0.0
        global_step = additional_info.get('global_step', 0) if additional_info else 0
        
        # 使用AM专用目录：checkpoints/mine_AM/{exp_name}/
        run_dir = os.path.abspath(os.path.join(self.checkpoint_dir, "mine_AM", self.exp_name))
        os.makedirs(run_dir, exist_ok=True)

        if is_best:
            # 先清理旧的最佳模型
            self._cleanup_old_best_models(run_dir)
            
            filename = f'best_model_step{global_step:08d}_tons{produce_tons:.1f}_reward{reward:.2f}.pt'
            best_path = os.path.abspath(os.path.join(run_dir, filename))
            
            if self.args.save_best_only_params:
                torch.save(agent.state_dict(), best_path)
            else:
                torch.save(checkpoint, best_path)
            
            print(f"新的最佳AM模型已保存到: {best_path}")
            print(f"  - 步数: {global_step}, 产量: {produce_tons:.1f}, 奖励: {reward:.2f}")
            
            return best_path
        else:
            if not self.args.save_best_only:
                filename = f'model_step{global_step:08d}_tons{produce_tons:.1f}_reward{reward:.2f}.pt'
                model_path = os.path.abspath(os.path.join(run_dir, filename))
                torch.save(checkpoint, model_path)
                return model_path

        self._cleanup_old_checkpoints()
        return ""

    def load_checkpoint(self,
                        agent: nn.Module,
                        optimizer: Optional[optim.Optimizer] = None,
                        path: Optional[str] = None) -> tuple:
        if path is None:
            checkpoints = self._get_checkpoints()
            if not checkpoints:
                return 0, float('-inf')
            path = checkpoints[-1]

        try:
            print(f"Loading AM checkpoint from {path}")
            checkpoint = torch.load(path, map_location=agent.device)
            agent.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['iteration'], checkpoint['reward']
        except Exception as e:
            print(f"Error loading AM checkpoint: {e}")
            return 0, float('-inf')

    def _cleanup_old_checkpoints(self):
        if self.args.save_best_only:
            return
        
        run_dir = os.path.join(self.checkpoint_dir, "mine_AM", self.exp_name)
        if not os.path.exists(run_dir):
            return
            
        files = [f for f in os.listdir(run_dir)
                if f.startswith('model_') and f.endswith('.pt')]
        files = [os.path.join(run_dir, f) for f in files]
        files = sorted(files)
        
        if len(files) > self.args.keep_checkpoint_max:
            for ckpt in files[:-self.args.keep_checkpoint_max]:
                os.remove(ckpt)

    def _get_checkpoints(self):
        run_dir = os.path.join(self.checkpoint_dir, "mine_AM", self.exp_name)
        if not os.path.exists(run_dir):
            return []
            
        files = [f for f in os.listdir(run_dir)
                if (f.startswith('best_model_') or f.startswith('model_')) and f.endswith('.pt')]
        files = [os.path.join(run_dir, f) for f in files]
        return sorted(files)
    
    def _get_latest_best_checkpoint(self):
        """查找最新的最佳模型检查点"""
        # 查找所有AM实验目录
        if not os.path.exists(os.path.join(self.checkpoint_dir, "mine_AM")):
            return None
            
        am_dirs = [d for d in os.listdir(os.path.join(self.checkpoint_dir, "mine_AM"))
                  if os.path.isdir(os.path.join(self.checkpoint_dir, "mine_AM", d))]
        
        if not am_dirs:
            return None
        
        # 按修改时间排序，选择最新的实验目录
        am_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, "mine_AM", x)), reverse=True)
        
        # 在最新的实验目录中查找最佳模型
        for exp_dir in am_dirs:
            exp_path = os.path.join(self.checkpoint_dir, "mine_AM", exp_dir)
            best_files = [f for f in os.listdir(exp_path) 
                         if f.startswith('best_model_') and f.endswith('.pt')]
            
            if best_files:
                # 返回最新的最佳模型文件路径
                best_files.sort(reverse=True)  # 按文件名降序排列
                return os.path.join(exp_path, best_files[0])
        
        return None

    def _cleanup_old_best_models(self, run_dir):
        """清理旧的最佳模型，确保只保留一个最新的最佳模型"""
        files = [f for f in os.listdir(run_dir) 
                if f.startswith('best_model_') and f.endswith('.pt')]
        
        # 删除所有旧的最佳模型文件（因为即将保存新的）
        for old_file in files:
            old_path = os.path.join(run_dir, old_file)
            os.remove(old_path)
            print(f"已删除旧的AM最佳模型: {old_file}")


class GreedyBaseline:
    """贪婪基线用于减少REINFORCE的方差"""
    def __init__(self, agent):
        self.agent = agent
        
    def evaluate(self, obs):
        """评估状态价值作为基线"""
        with torch.no_grad():
            return self.agent.get_value(obs) # 评估状态的价值


def manage_normalization_params(args: Optional[Args] = None) -> str:
    """管理AM算法的正则化参数文件"""
    if args and args.norm_path:
        if os.path.isfile(args.norm_path):
            print(f"使用指定的AM正则化参数文件: {args.norm_path}")
            return args.norm_path
        else:
            print(f"警告: 指定的AM参数文件不存在: {args.norm_path}")
    
    # 默认路径
    default_path = "/home/chengrongxian/git/openmines/openmines/src/dispatch_algorithms/am_norm_params_dense.json"
    return default_path


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # 计算批次大小
    args.batch_size = args.num_envs * args.num_steps # 并行环境的数量*单个环境收集的步数=每次更新总收集的步数
    args.num_iterations = args.total_timesteps // args.batch_size # 总训练的步数/每次更新总收集的步数=更新次数

    # 生成实验名称
    run_name = f"AM-v1_reinforce_s{args.seed}_lr{args.learning_rate:.2e}_" \
               f"g{args.gamma:.3f}_emb{args.embed_dim}_" \
               f"hd{args.hidden_dim}_t{int(time.time())}"
    
    print(f"使用的AM实验名称: {run_name}")

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 创建环境
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "only discrete action space is supported"

    # 创建AM智能体
    norm_path = manage_normalization_params(args=args) # 管理正则化参数，返回其路径
    agent = AttentionAgent(args=args, norm_path=norm_path).to(device) # 输入是矿山整体状态信息（包含待调度的车辆、站点等），输出是车辆的动作，同时输出动作的价值估计用于训练
    
    # 优化器
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    
    # 检查点管理器和基线
    checkpoint_manager = CheckpointManager(args, run_name)
    
    # 尝试加载已有的最佳模型继续训练
    start_iteration = 0
    if args.checkpoint_path:
        # 如果指定了检查点路径，加载指定的模型
        start_iteration, best_reward = checkpoint_manager.load_checkpoint(agent, optimizer, args.checkpoint_path)
        print(f"从指定检查点继续训练: iteration={start_iteration}, reward={best_reward}")
    else:
        # 尝试自动查找最新的最佳模型
        try:
            latest_checkpoint = checkpoint_manager._get_latest_best_checkpoint()
            if latest_checkpoint:
                start_iteration, best_reward = checkpoint_manager.load_checkpoint(agent, optimizer, latest_checkpoint)
                print(f"找到已有最佳模型，继续训练: iteration={start_iteration}, reward={best_reward}")
            else:
                print("未找到已有模型，从头开始训练")
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            print("从头开始训练")
    
    baseline = GreedyBaseline(agent) # 对每个状态估计一个基准价值作为比较基点，实际获得的奖励与基线的差值用于更新策略；训练更稳定更有效率

    # 训练数据存储 - 从agent获取观察空间维度
    obs_shape = agent.obs_shape  # 动态获取观察空间维度
    print(f"训练使用的观察空间维度: {obs_shape}")
    
    obs = torch.zeros((args.num_steps, args.num_envs, obs_shape), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)



    # 训练变量
    global_step = 0
    start_time = time.time()
    best_reward = float('-inf')
    latest_produce_tons = 0.0

    # 初始化环境
    env_seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(args.num_envs)]
    next_obs, infos = envs.reset(seed=env_seeds)
    next_obs = torch.FloatTensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    # 初始化episode追踪
    if not hasattr(envs, 'episode_rewards'):
        envs.episode_rewards = [0.0] * args.num_envs
        envs.episode_counts = [0] * args.num_envs
    if not hasattr(envs, 'episode_produce_tons'):
        envs.episode_produce_tons = []

    print("开始AM算法训练...")

    for iteration in range(start_iteration + 1, args.num_iterations + 1): # 对于每一轮更新的轮数
        # 数据收集阶段 - 记录每次迭代的基础信息
        current_raw_reward = 0
        
        # 每次迭代都记录基础训练指标
        writer.add_scalar("charts/iteration", iteration, global_step)
        
        if "episode" in infos and "r" in infos["episode"]:
            current_raw_reward = infos["episode"]["r"]
            current_normalized_reward = agent.normalize_reward(
                torch.tensor(current_raw_reward, device=device)
            ).item()
            
            writer.add_scalar("charts/episodic_return_raw", current_raw_reward, global_step)
            writer.add_scalar("charts/episodic_return_normalized", current_normalized_reward, global_step)

        # 检查是否是最佳模型
        is_best = current_raw_reward > best_reward
        if is_best:
            best_reward = current_raw_reward

        # 保存检查点
        if iteration % args.save_interval == 0 or is_best:
            checkpoint_manager.save_checkpoint(
                agent,
                optimizer,
                iteration,
                current_raw_reward,
                is_best,
                additional_info={
                    'global_step': global_step,
                    'time_elapsed': time.time() - start_time,
                    'produce_tons': latest_produce_tons,
                }
            )

        # 收集轨迹数据
        for step in range(args.num_steps): # 直到这一轮步数收集满，收集有状态数据、动作相关数据、奖励数据、终止数据等
            # 典型的在线学习，每一步的交互数据立即用于更新策略，而不是存放在回放缓冲区中
            global_step += args.num_envs
            obs[step] = next_obs # 更新数据并送入数组存储，后面用于训练
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            
            # 记录每个step的统计信息（类似PPO）
            if step % 10 == 0:  # 每10步记录一次以避免过多日志
                avg_value = value.mean().item()
                writer.add_scalar("charts/avg_value", avg_value, global_step)

            # 环境步进 - 添加动作转换和安全检查
            action_numpy = action.cpu().numpy()
            
            # 将神经网络动作转换为环境可接受的站点索引
            env_action_numpy = agent._convert_neural_action_to_env_action(action_numpy)
            
            # 最终安全检查：确保所有动作都在有效范围内
            max_load_sites = agent.num_load_sites
            max_dump_sites = agent.num_dump_sites
            for i in range(len(env_action_numpy)):
                if env_action_numpy[i] < 0:
                    print(f"警告：检测到负数动作 {env_action_numpy[i]}，重置为0")
                    env_action_numpy[i] = 0
                elif env_action_numpy[i] >= max(max_load_sites, max_dump_sites):
                    print(f"警告：检测到过大动作 {env_action_numpy[i]}，重置为0")
                    env_action_numpy[i] = 0
            
            # 只在第一步或动作转换发生变化时打印调试信息
            if step == 0 or not np.array_equal(action_numpy, env_action_numpy):
                print(f"调试：神经网络动作 {action_numpy} -> 环境动作 {env_action_numpy}")
            
            next_obs_np, reward, terminations, truncations, infos = envs.step(env_action_numpy)
            next_obs = torch.FloatTensor(next_obs_np).to(device)
            next_done = torch.FloatTensor(np.logical_or(terminations, truncations)).to(device)
            
            # 奖励正则化
            reward_tensor = torch.tensor(reward, device=device).view(-1)
            rewards[step] = agent.normalize_reward(reward_tensor)

            # 更新episode统计，将每个环境的奖励累加，并记录每个环境的步数
            for idx in range(args.num_envs):
                envs.episode_rewards[idx] += reward[idx]
                envs.episode_counts[idx] += 1

            # 记录episode结束时的统计 terminations: 表示环境自然结束（比如完成任务），truncations: 表示环境被强制截断（比如达到最大步数）
            # 当在num_steps里收集完一个回合的数据都会被保存到这里来
            for idx, (term, trunc) in enumerate(zip(terminations, truncations)):
                if term or trunc:  # 当episode因为终止(termination)或截断(truncation)结束时
                    produce_tons = sum(np.exp(infos["final_observation"][idx][-5:]) - 1) # 记录产量数据
                    envs.episode_produce_tons.append(produce_tons)
                    
                    raw_episode_reward = envs.episode_rewards[idx]  # 获取累积总奖励
                    episode_length = envs.episode_counts[idx]       # 获取episode总步数
                    
                    print(f"AM Episode完成: global_step={global_step}, env_id={idx}") 
                    print(f"  produce_tons={produce_tons:.2f}")
                    print(f"  raw_reward={raw_episode_reward:.2f}")
                    
                    writer.add_scalar("charts/episodic_return", raw_episode_reward, global_step) # 记录指标到TensorBoard
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    writer.add_scalar("charts/produce_tons", produce_tons, global_step)
                        
                    envs.episode_rewards[idx] = 0.0  # 重置奖励计数器
                    envs.episode_counts[idx] = 0     # 重置步数计数器

            # 计算平均产量 num_steps 是每个num_steps结束时候的统计数据
            if step == args.num_steps - 1:
                if len(envs.episode_produce_tons) > 0:
                    avg_tons = sum(envs.episode_produce_tons) / len(envs.episode_produce_tons)
                    writer.add_scalar("charts/avg_produce_tons_per_rollout", avg_tons, global_step)
                    envs.episode_produce_tons = []
                    latest_produce_tons = avg_tons

        # REINFORCE训练 收集完一个episode数据就开始训练
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            returns = torch.zeros_like(rewards, device=device)
            
            # 记录数据收集阶段的统计信息
            writer.add_scalar("charts/avg_reward_per_step", rewards.mean().item(), global_step)
            writer.add_scalar("charts/avg_value_per_step", values.mean().item(), global_step)
            
            # 计算回报
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                returns[t] = rewards[t] + args.gamma * nextvalues * nextnonterminal # R(t) = r(t) + γV(s(t+1))

        # 展平批次
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) # 指收集数据阶段的num_steps数据，而非之前缓冲区的所有数据
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 1.计算优势函数
        b_advantages = b_returns - b_values # 实际获得的回报与预测值之间的差异，用于指导策略更新 这是一个向量
        
        # 记录训练前的统计信息（类似PPO）
        writer.add_scalar("charts/advantages_mean", b_advantages.mean().item(), global_step)
        writer.add_scalar("charts/advantages_std", b_advantages.std().item(), global_step)
        writer.add_scalar("charts/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("charts/returns_std", b_returns.std().item(), global_step)
        
        # REINFORCE更新
        optimizer.zero_grad()
        
        # 2.重新计算当前策略的log概率，用当前策略重新评估所有状态-动作对
        # 为什么要这一步？到底有没有用新θ
        _, new_logprobs, entropy, new_values, _ = agent.get_action_and_value(b_obs, b_actions.long())
        
        # 策略损失
        policy_loss = -(new_logprobs * b_advantages).mean() # ∇J(θ) = E[∇log π(a|s) * A(s,a)] ∇log π(a|s) 是策略梯度，A(s,a) 是优势函数
        # 负号是因为我们要最大化目标，而优化器默认最小化损失
        # 如果动作比预期好（优势＞0）我们鼓励增加在这个状态之下选择这个动作的概率（调整θ使得这个状态-动作对的概率增加），鼓励梯度影响模型参数往这个方向去走
        
        # 价值损失
        value_loss = ((new_values - b_returns) ** 2).mean() # ∇J(θ) = E[(V(s) - R(s))^2]，我们希望值函数网络的预测越来越接近实际能获得的回报
        
        # 熵损失
        entropy_loss = entropy.mean() # H(π) = -Σ π(a|s)log π(a|s) 鼓励探索，避免过早收敛
        
        # 总损失
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss # 总损失是策略损失、价值损失和熵损失的加权和
        
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # 记录日志
        explained_var = 0.0
        with torch.no_grad():
            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        print(f"AM Training - Iteration: {iteration}, SPS: {sps}, Loss: {loss.item():.4f}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # 最终保存
    checkpoint_manager.save_checkpoint(
        agent,
        optimizer,
        args.num_iterations,
        best_reward,
        additional_info={'final': True}
    )

    envs.close()
    writer.close()

    print("\nAM算法训练完成!")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"模型保存在: {checkpoint_manager.checkpoint_dir}/mine_AM/{run_name}/") 


# import torch
# print(torch.__version__)