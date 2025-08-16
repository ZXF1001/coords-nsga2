import numpy as np


def fast_non_dominated_sort(objectives):
    """
    输入：objectives 为形状为 (n_objectives, pop_size) 的目标函数值数组；
    输出：返回一个列表，列表中的每个元素是一个列表，表示一个前沿
    """
    # 初始化数据结构
    n_objectives, num_population = objectives.shape
    dominated_solutions = [[] for _ in range(num_population)]
    domination_count = np.zeros(num_population)
    ranks = np.zeros(num_population)
    fronts = [[]]

    # 确定支配关系
    for p in range(num_population):
        for q in range(num_population):
            if p == q:
                continue
            # p 支配 q：p 在所有目标上都不差于 q，且至少在一个目标上优于 q
            p_dominates_q = True
            q_dominates_p = True
            p_better_in_at_least_one = False
            q_better_in_at_least_one = False

            for obj_func in objectives:
                if obj_func[p] > obj_func[q]:
                    q_dominates_p = False
                    p_better_in_at_least_one = True
                elif obj_func[p] < obj_func[q]:
                    p_dominates_q = False
                    q_better_in_at_least_one = True

            if p_dominates_q and p_better_in_at_least_one:
                dominated_solutions[p].append(q)
            elif q_dominates_p and q_better_in_at_least_one:
                domination_count[p] += 1

        # 如果没有解支配 p，则 p 属于第一个前沿
        if domination_count[p] == 0:
            fronts[0].append(p)

    # 按前沿层次进行排序
    current_rank = 0
    while fronts[current_rank]:
        next_front = []
        for p in fronts[current_rank]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = current_rank + 1
                    next_front.append(q)
        current_rank += 1
        fronts.append(next_front)

    # 去掉最后一个空层
    fronts.pop()
    return fronts


def crowding_distance(objectives):
    """
    计算 NSGA-II 中的拥挤度距离

    参数:
    value1 (numpy.ndarray): 第一个目标函数值的数组
    value2 (numpy.ndarray): 第二个目标函数值的数组

    返回:
    numpy.ndarray: 拥挤度距离数组
    """
    n_objectives, n_individuals = objectives.shape
    
    # 初始化拥挤度距离
    crowding_distances = np.zeros(n_individuals)
    
    # 处理每个目标
    for obj_values in objectives:
        # 排序
        sorted_indices = np.argsort(obj_values)
        sorted_values = obj_values[sorted_indices]
        
        # 计算范围
        obj_range = sorted_values[-1] - sorted_values[0]
        
        if obj_range == 0:
            continue
        
        # 创建距离贡献数组
        distances = np.zeros(n_individuals)
        
        # 设置边界为无穷大
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # 向量化计算中间点的距离
        if n_individuals > 2:
            # 计算相邻点的差值
            next_values = sorted_values[2:]
            prev_values = sorted_values[:-2]
            contributions = (next_values - prev_values) / obj_range
            
            # 将贡献值赋给对应的个体
            middle_indices = sorted_indices[1:-1]
            distances[middle_indices] = contributions
        
        # 累加到总拥挤度距离
        crowding_distances += distances
    
    return crowding_distances