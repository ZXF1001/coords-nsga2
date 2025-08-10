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
    needed_values = [values_lst[i] for i in idx_lst]
    sorted_list = [x for _, x in sorted(zip(needed_values, idx_lst))]
    return sorted_list


def crowding_distance(values1, values2, front):
    """
    输入：values1, values2 为两个目标函数的值列表；front 为一个前沿中的解的索引列表
    输出：返回一个列表，列表中的每个元素是一个解的拥挤距离
    """
    # 初始化拥挤距离
    distance = [0.0] * len(front)

    # 对每个目标进行排序（根据values的值给front里的index排序）
    sorted1 = sort_by_values(front, values1)
    sorted2 = sorted1[::-1]
    # 计算每个解的拥挤距离
    # ! 注意：这里min的对象是整个种群，因此min可能会受到惩罚函数较大系数的影响。最好的解决方案是惩罚函数乘一个适中的系数，避免把拥挤度距离小到抹去。
    min1, max1 = min(values1), max(values1)
    min2, max2 = min(values2), max(values2)
    if (min1 == max1) and (min2 == max2):
        return distance

    for i, ind in enumerate(front):
        idx_in_sorted1 = sorted1.index(ind)
        if idx_in_sorted1 == 0 or idx_in_sorted1 == len(front) - 1:
            distance[i] = float('inf')
        # * 如果和上一个点重合，就是0
        elif values1[sorted1[idx_in_sorted1]] == values1[sorted1[idx_in_sorted1 - 1]]:
            distance[i] += 0
        else:
            distance[i] += (values1[sorted1[idx_in_sorted1 + 1]] -
                            values1[sorted1[idx_in_sorted1 - 1]]) / (max1 - min1)

        idx_in_sorted2 = sorted2.index(ind)
        if idx_in_sorted2 == 0 or idx_in_sorted2 == len(front) - 1:
            distance[i] = float('inf')
        # * 如果和**下**一个点重合，就是0
        elif values2[sorted2[idx_in_sorted2]] == values2[sorted2[idx_in_sorted2 + 1]]:
            distance[i] += 0
        else:
            distance[i] += (values2[sorted2[idx_in_sorted2 + 1]] -
                            values2[sorted2[idx_in_sorted2 - 1]]) / (max2 - min2)

    return distance
