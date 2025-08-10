import numpy as np


def fast_non_dominated_sort(values1, values2):
    """
    输入：values1, values2 为两个目标函数的值列表；
    输出：返回一个列表，列表中的每个元素是一个列表，表示一个前沿
    """
    # 初始化数据结构
    num_population = len(values1)
    dominated_solutions = [[] for _ in range(num_population)]
    domination_count = [0] * num_population
    ranks = [0] * num_population
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


def sort_by_values(idx_lst, values_lst):
    # 根据values对list进行排序
    idx_arr = np.array(idx_lst)
    values_arr = np.array(values_lst)[idx_arr]

    # 根据values进行排序
    sorted_idx_arr = idx_arr[np.argsort(values_arr)]
    return sorted_idx_arr

def crowding_distance(values1, values2, front):
    """
    Calculate crowding distance for solutions in a Pareto front.
    
    Args:
        values1, values2: Lists/arrays of objective function values
        front: List of indices representing solutions in the current front
        
    Returns:
        ndarray: Crowding distances for solutions in the front
    """
    if len(front) <= 2:
        return np.full(len(front), np.inf)

    # Convert inputs to numpy arrays
    values = np.column_stack((np.array(values1)[front], np.array(values2)[front]))
    
    # Initialize distances
    distances = np.zeros(len(front))
    
    # Handle case where all values are identical
    if np.all(values[0] == values):
        return distances
        
    # Calculate normalized crowding distance for each objective
    for i in range(2):
        # Sort values and get indices
        idx = values[:, i].argsort()
        sorted_values = values[idx, i]
        
        # Set boundary points to infinity
        distances[idx[0]] = distances[idx[-1]] = np.inf
        
        # Normalize and accumulate distances
        norm = sorted_values[-1] - sorted_values[0]
        if norm > 0:
            distances[idx[1:-1]] += (sorted_values[2:] - sorted_values[:-2]) / norm
            
    return distances
