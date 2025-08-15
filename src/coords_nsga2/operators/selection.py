import numpy as np

from ..utils import crowding_distance, fast_non_dominated_sort


def coords_selection(population, values_P, tourn_size=3):
    # 锦标赛选择，选择的依据是快速非支配排序的结果和拥挤度
    pop_size = len(population)
    # 1. 先把所有的解进行快速非支配排序和拥挤度计算
    population_sorted_in_fronts = fast_non_dominated_sort(values_P)
    crowding_distances = [crowding_distance(
        values_P[:, front]) for front in population_sorted_in_fronts]
    # 将这两个结果组成一个列表，方便后续比较：第一列为index，第二列为前沿等级，第三列为拥挤度
    compare_table = []
    for i, front in enumerate(population_sorted_in_fronts):
        for j, idx in enumerate(front):
            compare_table.append([idx, i, crowding_distances[i][j]])
    # 按照index排序
    compare_table = np.array(compare_table)
    compare_table = compare_table[compare_table[:, 0].argsort()]

    # 2. 生成[0,self.pop_size)的随机数，形状为(self.pop_size, tourn_size)
    aspirants_idx = np.random.randint(
        pop_size, size=(pop_size, tourn_size))

    # 3. 选择每组中最前沿（前沿等级相同时选择拥挤度最高）的解
    candidates = compare_table[aspirants_idx]
    sorted_indices = np.lexsort((-candidates[..., 2], candidates[..., 1]))
    Q_idx = aspirants_idx[np.arange(pop_size), sorted_indices[:, 0]]
    return population[Q_idx]
