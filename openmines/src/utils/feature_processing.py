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


def preprocess_enhanced_observation(observation, max_sim_time, max_tracked_trucks=None):
    """
    处理包含其他车辆详细信息的增强观察
    :param observation: 增强观察字典
    :param max_sim_time: 最大仿真时间
    :param max_tracked_trucks: 最大跟踪车辆数，如果为None则追踪所有车辆
    :return: 扩展的特征向量
    """
    # 获取基础特征（原有的154维）
    base_features = preprocess_observation(observation, max_sim_time)
    
    # 处理其他车辆简化信息
    other_trucks_features = _process_other_trucks_simplified(
        observation.get("other_trucks_detailed", {}), 
        max_tracked_trucks
    )
    
    # 合并特征
    enhanced_features = np.concatenate([base_features, other_trucks_features])
    return enhanced_features.astype(np.float32)


def _process_other_trucks_simplified(other_trucks_info: dict, max_tracked_trucks=None) -> np.ndarray:
    """处理其他车辆的简化信息 - 支持所有车辆跟踪"""
    detailed_positions = other_trucks_info.get("detailed_positions", [])
    movement_directions = other_trucks_info.get("movement_directions", [])
    progress_states = other_trucks_info.get("progress_states", [])
    eta_predictions = other_trucks_info.get("eta_predictions", [])
    
    # 如果没有指定最大跟踪数量，则处理所有可用的车辆
    if max_tracked_trucks is None:
        actual_truck_count = len(detailed_positions)
    else:
        actual_truck_count = min(max_tracked_trucks, len(detailed_positions))
    
    all_truck_features = []
    
    # 处理实际存在的车辆
    for i in range(actual_truck_count):
        truck_features = _encode_single_truck_simplified(
            detailed_positions[i],
            movement_directions[i],
            progress_states[i],
            eta_predictions[i]
        )
        all_truck_features.extend(truck_features)
    
    # 如果指定了最大跟踪数量且实际车辆数少于该数量，则用零向量填充
    if max_tracked_trucks is not None and actual_truck_count < max_tracked_trucks:
        for i in range(max_tracked_trucks - actual_truck_count):
            truck_features = np.zeros(4)  # 每辆车4维特征
            all_truck_features.extend(truck_features)
    
    return np.array(all_truck_features)




def _encode_single_truck_simplified(position_info: dict, direction: str, progress: dict, eta: dict) -> np.ndarray:
    """编码单个车辆的精确道路特征 - 包含具体的起点和终点信息"""
    
    current_location = position_info["current_location_name"]
    target_location = position_info.get("target_location_name", "")
    
    # 获取起点和终点的编号
    start_id, end_id = _get_road_endpoints(current_location, target_location, direction)
    
    # 道路编码 (3维) - 起点ID, 终点ID, 道路类型
    road_features = [
        start_id / 10.0,      # 起点ID，正则化到[0,1]
        end_id / 10.0,        # 终点ID，正则化到[0,1] 
        _get_road_type_id(direction) / 3.0  # 道路类型ID，正则化到[0,1]
    ]
    
    # 进度信息 (1维)
    progress_feature = [progress["progress_ratio"]]  # 当前道路行驶进度
    
    return np.concatenate([road_features, progress_feature])


def _get_road_endpoints(current_location: str, target_location: str, direction: str) -> tuple:
    """获取道路的起点和终点ID"""
    
    # 站点名称到ID的映射
    def get_site_id(site_name: str) -> int:
        if not site_name:
            return 0
        
        # 充电站
        if "charging" in site_name.lower():
            return 0
        
        # 装载点 (ID: 1-5)
        if "loadsite1" in site_name.lower() or site_name == "LoadSite1":
            return 1
        elif "loadsite2" in site_name.lower():
            return 2
        elif "loadsite3" in site_name.lower():
            return 3
        elif "loadsite4" in site_name.lower():
            return 4
        elif "loadsite5" in site_name.lower():
            return 5
        
        # 卸载点 (ID: 6-10)
        elif "dumpsite1" in site_name.lower():
            return 6
        elif "dumpsite2" in site_name.lower():
            return 7
        elif "dumpsite3" in site_name.lower():
            return 8
        elif "dumpsite4" in site_name.lower():
            return 9
        elif "dumpsite5" in site_name.lower():
            return 10
        
        return 0  # 未知
    
    # 如果车辆在路上，从location_name和direction解析
    if current_location.startswith("road_to_"):
        # 在路上，target就是终点
        end_id = get_site_id(target_location)
        
        if direction == "init":
            start_id = 0  # 从充电站出发
        elif direction == "haul":
            # 从装载点出发到卸载点，可以从current_location反推起点
            # 例如：如果去DumpSite3，可能从某个装载点出发
            # 这里简化：根据终点推断可能的起点
            if end_id >= 6 and end_id <= 10:  # 确实是去卸载点
                # 简化假设：平均分配装载点
                start_id = ((end_id - 6) % 5) + 1  # 卸载点1->装载点1, 卸载点2->装载点2...
            else:
                start_id = 1  # 默认装载点1
        elif direction == "unhaul":
            # 从卸载点出发到装载点
            if end_id >= 1 and end_id <= 5:  # 确实是去装载点
                # 简化假设：根据终点推断起点
                start_id = (end_id - 1) + 6  # 装载点1->卸载点1, 装载点2->卸载点2...
            else:
                start_id = 6  # 默认卸载点1
        else:
            start_id = 0  # 未知，默认充电站
            
    else:
        # 车辆在站点，起点就是当前位置
        start_id = get_site_id(current_location)
        end_id = get_site_id(target_location)
    
    return start_id, end_id


def _get_road_type_id(direction: str) -> int:
    """获取道路类型ID"""
    if direction == "init":
        return 1  # 充电站->装载点
    elif direction == "haul":
        return 2  # 装载点->卸载点
    elif direction == "unhaul":
        return 3  # 卸载点->装载点
    else:
        return 0  # 未知


def preprocess_observation_auto(observation, max_sim_time):
    """
    自动选择处理函数的包装器
    """
    if "other_trucks_detailed" in observation:
        return preprocess_enhanced_observation(observation, max_sim_time)
    else:
        return preprocess_observation(observation, max_sim_time) 