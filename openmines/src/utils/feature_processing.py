import numpy as np

def preprocess_observation(observation, max_sim_time):
    # 用于将原始的状态信息处理为194维度矿区状态向量
    """特征预处理逻辑"""
    """
    0.订单信息
    """
    # 1.订单类型,时间信息
    event_name = observation['event_name']
    if event_name == "init":
        event_type = [1, 0, 0]
        action_space_n = observation['info']['load_num']
    elif event_name == "haul":
        event_type = [0, 1, 0]
        action_space_n = observation['info']['unload_num']
    else:
        event_type = [0, 0, 1]
        action_space_n = observation['info']['load_num']
    # 2.当前订单时间绝对位置和相对位置
    # 添加调试信息和错误处理
    delta_time_raw = observation['info']['delta_time']
    time_raw = observation['info']['time']
    
    try:
        time_delta = float(delta_time_raw)  # 距离上次调度的时间
    except (TypeError, ValueError) as e:
        print(f"Error converting delta_time to float: {delta_time_raw}, type: {type(delta_time_raw)}")
        # 如果转换失败，使用默认值
        time_delta = 0.0
    
    try:
        time_now = float(time_raw) / max_sim_time  # 当前时间(正则化）
    except (TypeError, ValueError) as e:
        print(f"Error converting time to float: {time_raw}, type: {type(time_raw)}")
        # 如果转换失败，使用默认值
        time_now = 0.0
    time_left = 1 - time_now  # 距离结束时间
    order_state = np.array([event_type[0], event_type[1], event_type[2], time_delta, time_now, time_left])

    """
    1.车辆自身信息
    """
    # 矿山总卡车数目（用于正则化）
    truck_num = observation['mine_status']['truck_count']
    # 4.车辆当前位置One-hot编码
    truck_location_onehot = np.array(observation["the_truck_status"]["truck_location_onehot"])
    # 车辆装载量，车辆循环时间（正则化）
    truck_features = np.array([
        np.log(observation['the_truck_status']['truck_load'] + 1),
        np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
    ])
    truck_self_state = np.concatenate([truck_location_onehot, truck_features])

    """
    2.道路相关信息
    """
    # 车预期行驶时间
    travel_time = np.array(observation['cur_road_status']['distances']) * 60 / 25
    # 道路上卡车数量
    truck_counts = np.array(observation['cur_road_status']['truck_counts']) / (truck_num + 1e-8)
    # 道路距离信息
    road_dist = np.array(observation['cur_road_status']['oh_distances'])
    # 道路拥堵信息
    road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'])

    road_states = np.concatenate([travel_time, truck_counts, road_dist, road_jam])

    """
    3.目标点相关信息
    """
    # 预期等待时间
    est_wait = np.log(observation['target_status']['single_est_wait'] + 1)  # 包含了路上汽车+队列汽车的目标装载点等待时间
    tar_wait_time = np.log(np.array(observation['target_status']['est_wait']) + 1)  # 不包含路上汽车
    # 队列长度（正则化）
    queue_lens = np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8)
    # 装载量
    tar_capa = np.log(np.array(observation['target_status']['capacities']) + 1)
    # 各个目标点当前的产能系数(维护导致的产能下降）
    ability_ratio = np.array(observation['target_status']['service_ratio'])
    # 已经生产的矿石量（正则化）
    produced_tons = np.log(np.array(observation['target_status']['produced_tons']) + 1)

    tar_state = np.concatenate([est_wait, tar_wait_time, queue_lens, tar_capa, ability_ratio, produced_tons])

    state = np.concatenate([order_state, truck_self_state, road_states, tar_state])
    assert not np.isnan(state).any(), f"NaN detected in state: {state}"
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    return state.astype(np.float32)


def preprocess_enhanced_observation(observation, max_sim_time, max_tracked_trucks=10):
    """
    处理包含其他车辆详细信息的增强观察
    :param observation: 增强观察字典
    :param max_sim_time: 最大仿真时间
    :param max_tracked_trucks: 最大跟踪车辆数
    :return: 扩展的特征向量
    """
    # 获取基础特征（原有的194维）
    base_features = preprocess_observation(observation, max_sim_time)
    
    # 处理其他车辆详细信息
    other_trucks_features = _process_other_trucks_detailed(
        observation.get("other_trucks_detailed", {}), 
        max_tracked_trucks
    )
    
    # 合并特征
    enhanced_features = np.concatenate([base_features, other_trucks_features])
    return enhanced_features.astype(np.float32)


def _process_other_trucks_detailed(other_trucks_info: dict, max_tracked_trucks: int) -> np.ndarray:
    """处理其他车辆的详细信息"""
    detailed_positions = other_trucks_info.get("detailed_positions", [])
    movement_directions = other_trucks_info.get("movement_directions", [])
    progress_states = other_trucks_info.get("progress_states", [])
    eta_predictions = other_trucks_info.get("eta_predictions", [])
    
    all_truck_features = []
    
    for i in range(max_tracked_trucks):
        if i < len(detailed_positions):
            # 处理单个车辆的特征
            truck_features = _encode_single_truck_features(
                detailed_positions[i],
                movement_directions[i],
                progress_states[i],
                eta_predictions[i]
            )
        else:
            # 填充零向量
            truck_features = np.zeros(19)  # 每辆车19维特征
        
        all_truck_features.extend(truck_features)
    
    return np.array(all_truck_features)


def _encode_single_truck_features(position_info: dict, direction: str, progress: dict, eta: dict) -> np.ndarray:
    """编码单个车辆的特征"""
    # 位置编码 (11维 one-hot)
    location_onehot = _encode_location_name(position_info["current_location_name"])
    
    # 方向编码 (4维 one-hot)
    direction_onehot = _encode_movement_direction(direction)
    
    # 数值特征 (4维)
    numerical_features = [
        position_info["load_ratio"],
        progress["progress_ratio"],
        eta["eta_minutes"] / 60.0,  # 标准化为小时
        1.0 if position_info["status"] == "moving" else 0.0
    ]
    
    return np.concatenate([location_onehot, direction_onehot, numerical_features])


def _encode_location_name(location_name: str) -> np.ndarray:
    """将位置名称编码为one-hot向量"""
    onehot = np.zeros(11)  # 1充电站+5装载点+5卸载点
    
    # 处理道路上的情况：road_to_destination
    if location_name.startswith("road_to_"):
        destination = location_name.replace("road_to_", "")
        location_name = destination
    
    if "charging" in location_name.lower():
        onehot[0] = 1.0
    elif "load_site" in location_name.lower():
        # 提取装载点编号
        try:
            site_num = int(location_name.split('_')[-1])
            if 1 <= site_num <= 5:
                onehot[site_num] = 1.0
        except:
            pass
    elif "dump_site" in location_name.lower():
        # 提取卸载点编号
        try:
            site_num = int(location_name.split('_')[-1])
            if 1 <= site_num <= 5:
                onehot[5 + site_num] = 1.0
        except:
            pass
    
    return onehot


def _encode_movement_direction(direction: str) -> np.ndarray:
    """编码移动方向"""
    direction_map = {"init": 0, "haul": 1, "unhaul": 2, "stationary": 3}
    onehot = np.zeros(4)
    if direction in direction_map:
        onehot[direction_map[direction]] = 1.0
    return onehot


def preprocess_observation_auto(observation, max_sim_time):
    """
    自动选择处理函数的包装器
    """
    if "other_trucks_detailed" in observation:
        return preprocess_enhanced_observation(observation, max_sim_time)
    else:
        return preprocess_observation(observation, max_sim_time) 