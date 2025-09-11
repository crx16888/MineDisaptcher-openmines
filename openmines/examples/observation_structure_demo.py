#!/usr/bin/env python3
"""
展示修改后送入PPO网络的384维观察向量结构
"""

import numpy as np
import json
from typing import Dict, List

def create_sample_raw_observation() -> Dict:
    """创建一个示例的原始观察字典"""
    return {
        "truck_name": "Truck1",
        "event_name": "haul",  # init/haul/unhaul
        "info": {
            "produce_tons": 1250.5,
            "time": 120.5,
            "delta_time": 2.3,
            "load_num": 5,
            "unload_num": 5
        },
        "the_truck_status": {
            "truck_location": "load_site_2",
            "truck_location_onehot": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 11维：1充电站+5装载点+5卸载点
            "truck_load": 35.0,
            "truck_capacity": 40.0,
            "truck_cycle_time": 15.2,
            "truck_speed": 25.0
        },
        "target_status": {
            "queue_lengths": [2, 1, 3, 0, 1, 1, 2, 0, 1, 3],  # 5装载点+5卸载点
            "capacities": [10, 8, 12, 9, 11, 15, 12, 18, 14, 16],
            "est_wait": [5.2, 3.1, 8.7, 0.5, 4.2, 2.1, 6.3, 1.2, 3.8, 7.5],
            "single_est_wait": [6.1, 4.2, 9.8, 1.3, 5.1],  # 当前事件类型的精确等待时间
            "service_ratio": [0.9, 1.0, 0.8, 1.0, 0.95, 1.0, 0.85, 1.0, 0.92, 0.88],
            "produced_tons": [120, 89, 156, 67, 134, 89, 112, 145, 98, 167],
            "service_counts": [45, 32, 58, 23, 49, 34, 41, 52, 38, 61]
        },
        "cur_road_status": {
            "truck_counts": [2, 1, 0, 3, 1],  # 从当前位置到各目标的道路卡车数
            "distances": [3.2, 4.1, 2.8, 3.9, 5.1],  # 距离
            "truck_jam_counts": [0, 1, 0, 2, 0],  # 拥堵数
            "oh_truck_count": [3, 2, 1, 4, 2, 1, 0, 2, 1, 3, 2, 1, 0, 1, 2] + [0] * 40,  # 所有道路卡车数（55维）
            "oh_distances": [1.5, 2.3, 1.8, 2.1, 1.9, 3.2, 4.1, 2.8, 3.9] + [0] * 46,  # 所有道路距离（55维）
            "oh_truck_jam_count": [0, 1, 0, 2, 0, 1, 0, 1, 0, 2] + [0] * 45,  # 所有道路拥堵（55维）
            "oh_repair_count": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1] + [0] * 45  # 所有道路维修（55维）
        },
        "mine_status": {
            "truck_count": 10,
            "total_production": 1250.5,
            # ... 其他矿山状态
        },
        # 🆕 增强观察：其他车辆详细信息
        "other_trucks_detailed": {
            "detailed_positions": [
                {
                    "current_location_name": "dump_site_1",
                    "target_location_name": "load_site_3",
                    "current_location_type": "dump",
                    "target_location_type": "load",
                    "status": "moving",
                    "load_ratio": 0.0
                },
                {
                    "current_location_name": "charging_site",
                    "target_location_name": "load_site_1",
                    "current_location_type": "charging",
                    "target_location_type": "load",
                    "status": "moving",
                    "load_ratio": 0.0
                },
                {
                    "current_location_name": "load_site_1",
                    "target_location_name": None,
                    "current_location_type": "load",
                    "target_location_type": "unknown",
                    "status": "loading",
                    "load_ratio": 0.6
                },
                # ... 更多车辆（最多10辆）
            ],
            "movement_directions": ["unhaul", "init", "stationary"],  # 对应上面的车辆
            "progress_states": [
                {"progress_ratio": 0.3, "remaining_distance": 2.1, "total_distance": 3.0},
                {"progress_ratio": 0.7, "remaining_distance": 0.6, "total_distance": 2.0},
                {"progress_ratio": 0.0, "remaining_distance": 0.0, "total_distance": 0.0}
            ],
            "eta_predictions": [
                {"eta_minutes": 5.04, "eta_absolute_time": 125.54},
                {"eta_minutes": 1.44, "eta_absolute_time": 121.94},
                {"eta_minutes": 0.0, "eta_absolute_time": 120.5}
            ]
        }
    }

def demonstrate_feature_processing():
    """演示特征预处理过程"""
    print("=== 384维观察向量结构演示 ===\n")
    
    # 创建示例原始观察
    raw_obs = create_sample_raw_observation()
    
    print("1. 原始观察字典结构：")
    print(f"   - truck_name: {raw_obs['truck_name']}")
    print(f"   - event_name: {raw_obs['event_name']}")
    print(f"   - 包含增强观察: {'other_trucks_detailed' in raw_obs}")
    print(f"   - 其他车辆数量: {len(raw_obs['other_trucks_detailed']['detailed_positions'])}")
    print()
    
    # 模拟特征预处理过程
    processed_features = simulate_feature_processing(raw_obs)
    
    print("2. 处理后的384维向量结构：")
    print(f"   - 总维度: {len(processed_features)}")
    print(f"   - 数据类型: {processed_features.dtype}")
    print(f"   - 数值范围: [{processed_features.min():.3f}, {processed_features.max():.3f}]")
    print()
    
    # 详细展示各部分特征
    show_feature_breakdown(processed_features)
    
    return processed_features

def simulate_feature_processing(raw_obs: Dict) -> np.ndarray:
    """模拟特征预处理过程"""
    max_sim_time = 480.0  # 8小时
    
    # 1. 订单状态 (6维)
    event_name = raw_obs['event_name']
    if event_name == "init":
        event_type = [1, 0, 0]
    elif event_name == "haul":
        event_type = [0, 1, 0]
    else:
        event_type = [0, 0, 1]
    
    time_delta = float(raw_obs['info']['delta_time'])
    time_now = float(raw_obs['info']['time']) / max_sim_time
    time_left = 1 - time_now
    order_state = np.array([event_type[0], event_type[1], event_type[2], time_delta, time_now, time_left])
    
    # 2. 车辆自身状态 (13维)
    truck_location_onehot = np.array(raw_obs["the_truck_status"]["truck_location_onehot"])
    truck_features = np.array([
        np.log(raw_obs['the_truck_status']['truck_load'] + 1),
        np.log(raw_obs['the_truck_status']['truck_cycle_time'] + 1),
    ])
    truck_self_state = np.concatenate([truck_location_onehot, truck_features])
    
    # 3. 道路相关状态 (约100维) - 简化版
    travel_time = np.array(raw_obs['cur_road_status']['distances']) * 60 / 25
    truck_counts = np.array(raw_obs['cur_road_status']['truck_counts']) / 11  # 正则化
    road_dist = np.array(raw_obs['cur_road_status']['oh_distances'][:50])  # 取前50维
    road_jam = np.array(raw_obs['cur_road_status']['oh_truck_jam_count'][:35])  # 取前35维
    road_states = np.concatenate([travel_time, truck_counts, road_dist, road_jam])
    
    # 4. 目标点状态 (75维) - 简化版
    est_wait = np.log(np.array(raw_obs['target_status']['single_est_wait']) + 1)
    tar_wait_time = np.log(np.array(raw_obs['target_status']['est_wait']) + 1)
    queue_lens = np.array(raw_obs['target_status']['queue_lengths']) / 11
    tar_capa = np.log(np.array(raw_obs['target_status']['capacities']) + 1)
    ability_ratio = np.array(raw_obs['target_status']['service_ratio'])
    produced_tons = np.log(np.array(raw_obs['target_status']['produced_tons']) + 1)
    tar_state = np.concatenate([est_wait, tar_wait_time, queue_lens, tar_capa, ability_ratio, produced_tons])
    
    # 基础194维特征
    base_features = np.concatenate([order_state, truck_self_state, road_states, tar_state])
    # 补齐到194维
    if len(base_features) < 194:
        base_features = np.pad(base_features, (0, 194 - len(base_features)), 'constant')
    else:
        base_features = base_features[:194]
    
    # 5. 🆕 其他车辆详细信息 (190维)
    other_trucks_features = process_other_trucks_info(raw_obs['other_trucks_detailed'])
    
    # 合并所有特征
    final_features = np.concatenate([base_features, other_trucks_features])
    
    return final_features.astype(np.float32)

def process_other_trucks_info(other_trucks_info: Dict) -> np.ndarray:
    """处理其他车辆信息，返回190维特征"""
    max_tracked_trucks = 10
    detailed_positions = other_trucks_info.get("detailed_positions", [])
    movement_directions = other_trucks_info.get("movement_directions", [])
    progress_states = other_trucks_info.get("progress_states", [])
    eta_predictions = other_trucks_info.get("eta_predictions", [])
    
    all_truck_features = []
    
    for i in range(max_tracked_trucks):
        if i < len(detailed_positions):
            # 编码单个车辆特征 (19维)
            position_info = detailed_positions[i]
            direction = movement_directions[i] if i < len(movement_directions) else "stationary"
            progress = progress_states[i] if i < len(progress_states) else {"progress_ratio": 0.0}
            eta = eta_predictions[i] if i < len(eta_predictions) else {"eta_minutes": 0.0}
            
            # 位置编码 (11维 one-hot)
            location_onehot = encode_location_name(position_info["current_location_name"])
            
            # 方向编码 (4维 one-hot)
            direction_onehot = encode_movement_direction(direction)
            
            # 数值特征 (4维)
            numerical_features = [
                position_info["load_ratio"],
                progress["progress_ratio"],
                eta["eta_minutes"] / 60.0,  # 标准化为小时
                1.0 if position_info["status"] == "moving" else 0.0
            ]
            
            truck_features = np.concatenate([location_onehot, direction_onehot, numerical_features])
        else:
            # 填充零向量 (19维)
            truck_features = np.zeros(19)
        
        all_truck_features.extend(truck_features)
    
    return np.array(all_truck_features)

def encode_location_name(location_name: str) -> np.ndarray:
    """位置名称编码"""
    onehot = np.zeros(11)  # 1充电站+5装载点+5卸载点
    
    if "charging" in location_name.lower():
        onehot[0] = 1.0
    elif "load_site" in location_name.lower():
        try:
            site_num = int(location_name.split('_')[-1])
            if 1 <= site_num <= 5:
                onehot[site_num] = 1.0
        except:
            pass
    elif "dump_site" in location_name.lower():
        try:
            site_num = int(location_name.split('_')[-1])
            if 1 <= site_num <= 5:
                onehot[5 + site_num] = 1.0
        except:
            pass
    
    return onehot

def encode_movement_direction(direction: str) -> np.ndarray:
    """移动方向编码"""
    direction_map = {"init": 0, "haul": 1, "unhaul": 2, "stationary": 3}
    onehot = np.zeros(4)
    if direction in direction_map:
        onehot[direction_map[direction]] = 1.0
    return onehot

def show_feature_breakdown(features: np.ndarray):
    """详细展示特征分解"""
    print("3. 384维向量详细分解：")
    
    # 基础特征 (194维)
    base_features = features[:194]
    print(f"   📊 基础观察 (0-193维, 共194维):")
    print(f"      - 订单状态 (0-5维): {base_features[:6]}")
    print(f"      - 车辆状态 (6-18维): {base_features[6:19]}")
    print(f"      - 道路状态 (19-118维): {base_features[19:119]}")
    print(f"      - 目标状态 (119-193维): {base_features[119:194]}")
    print()
    
    # 增强特征 (190维)
    enhanced_features = features[194:]
    print(f"   🚛 增强观察 (194-383维, 共190维):")
    print(f"      - 其他车辆信息: 10辆车 × 19维/车 = 190维")
    
    for i in range(10):
        start_idx = i * 19
        end_idx = start_idx + 19
        truck_features = enhanced_features[start_idx:end_idx]
        
        if np.any(truck_features != 0):  # 只显示非零的车辆
            location_onehot = truck_features[:11]
            direction_onehot = truck_features[11:15]
            numerical = truck_features[15:19]
            
            # 解码位置
            location_idx = np.argmax(location_onehot) if np.any(location_onehot) else -1
            location_names = ["charging_site", "load_site_1", "load_site_2", "load_site_3", 
                            "load_site_4", "load_site_5", "dump_site_1", "dump_site_2", 
                            "dump_site_3", "dump_site_4", "dump_site_5"]
            location = location_names[location_idx] if 0 <= location_idx < len(location_names) else "unknown"
            
            # 解码方向
            direction_idx = np.argmax(direction_onehot) if np.any(direction_onehot) else -1
            directions = ["init", "haul", "unhaul", "stationary"]
            direction = directions[direction_idx] if 0 <= direction_idx < len(directions) else "unknown"
            
            print(f"      - 车辆{i+1} ({194+start_idx}-{194+end_idx-1}维):")
            print(f"        位置: {location}")
            print(f"        方向: {direction}")
            print(f"        载重比例: {numerical[0]:.2f}")
            print(f"        进度: {numerical[1]:.2f}")
            print(f"        ETA(小时): {numerical[2]:.2f}")
            print(f"        移动中: {'是' if numerical[3] > 0.5 else '否'}")
        else:
            print(f"      - 车辆{i+1} ({194+start_idx}-{194+end_idx-1}维): 空位 (全零)")
    print()

def show_network_input_format(features: np.ndarray):
    """展示送入网络的格式"""
    print("4. 送入PPO网络的格式：")
    print(f"   - torch.FloatTensor shape: {features.shape}")
    print(f"   - device: cuda/cpu")
    print(f"   - 网络输入层: Linear(384, hidden_size)")
    print()
    
    print("5. 网络处理流程：")
    print("   384维向量 → 隐藏层1 → 隐藏层2 → Actor/Critic分支")
    print("   Actor输出: 动作概率分布")
    print("   Critic输出: 状态价值估计")
    print()

def save_sample_to_file(features: np.ndarray):
    """保存示例到文件"""
    sample_data = {
        "description": "384维PPO网络输入向量示例",
        "total_dimensions": len(features),
        "feature_breakdown": {
            "basic_observation": {
                "dimensions": "0-193 (194 dims)",
                "components": ["order_state(6)", "truck_self_state(13)", "road_states(~100)", "target_state(75)"]
            },
            "enhanced_observation": {
                "dimensions": "194-383 (190 dims)",
                "components": ["other_trucks_info: 10_trucks × 19_dims_per_truck"]
            }
        },
        "sample_values": features.tolist(),
        "non_zero_indices": np.where(features != 0)[0].tolist(),
        "statistics": {
            "mean": float(features.mean()),
            "std": float(features.std()),
            "min": float(features.min()),
            "max": float(features.max()),
            "non_zero_count": int(np.count_nonzero(features))
        }
    }
    
    with open("openmines/examples/sample_384dim_vector.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("6. 示例数据已保存到: openmines/examples/sample_384dim_vector.json")

if __name__ == "__main__":
    print("🚛 矿山调度PPO网络输入向量结构演示\n")
    
    # 演示完整流程
    processed_features = demonstrate_feature_processing()
    show_network_input_format(processed_features)
    save_sample_to_file(processed_features)
    
    print("✅ 演示完成！")
    print("\n📝 总结:")
    print("- 基础观察: 194维 (订单+卡车+道路+目标)")
    print("- 增强观察: +190维 (10辆其他车辆 × 19维/车)")
    print("- 总维度: 384维")
    print("- 数据类型: torch.FloatTensor")
    print("- 网络输入: Linear(384, hidden_size)")
