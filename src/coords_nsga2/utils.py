import numpy as np


def fast_non_dominated_sort(values1, values2):
    """
    输入：values1, values2 为两个目标函数的值列表；
    输出：返回一个列表，列表中的每个元素是一个列表，表示一个前沿
    """
    assert len(values1) == len(values2)
    # 初始化数据结构
    num_population = len(values1)
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
            if (values1[p] > values1[q] and values2[p] >= values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]):
                dominated_solutions[p].append(q)
            elif (values1[q] > values1[p] and values2[q] >= values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]):
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


def crowding_distance(value1, value2):
    """
    计算 NSGA-II 中的拥挤度距离
    
    参数:
    value1 (numpy.ndarray): 第一个目标函数值的数组
    value2 (numpy.ndarray): 第二个目标函数值的数组
    
    返回:
    numpy.ndarray: 拥挤度距离数组
    """
    n = len(value1)
    value1 = np.array(value1)
    value2 = np.array(value2)
    
    # 对 value1 进行排序，并记录原始索引
    sorted_idx = np.argsort(value1)
    sorted_value1 = value1[sorted_idx]
    sorted_value2 = value2[sorted_idx]
    
    # 初始化拥挤度距离数组
    crowding_dist = np.zeros(n)
    
    # 边界点的拥挤度距离为无穷大
    crowding_dist[0] = float('inf')
    crowding_dist[-1] = float('inf')
    
    # 计算归一化因子
    norm_factor1 = sorted_value1.max() - sorted_value1.min()
    norm_factor2 = sorted_value2.max() - sorted_value2.min()
    
    # 避免除零错误
    if norm_factor1 == 0:
        norm_factor1 = 1
    if norm_factor2 == 0:
        norm_factor2 = 1
    
    # 矢量化计算中间点的拥挤度距离
    if n > 2:
        value1_diff = np.abs(sorted_value1[2:] - sorted_value1[:-2]) / norm_factor1
        value2_diff = np.abs(sorted_value2[2:] - sorted_value2[:-2]) / norm_factor2
        crowding_dist[1:-1] = value1_diff + value2_diff
    # 将拥挤度距离按照原始索引还原
    result = np.zeros(n)
    result[sorted_idx] = crowding_dist
    
    return result