import numpy as np


def coords_crossover(population, prob_crs):
    n_points = population.shape[1]
    # 坐标的交叉算子
    for i in range(0, len(population), 2):
        if np.random.rand() < prob_crs:
            cross_num = np.random.randint(1, n_points)
            cross_idx = np.random.choice(
                n_points, cross_num, replace=False)
            population[i:i+2, cross_idx] = population[i:i+2, cross_idx][::-1]
    return population


def region_crossover(population_list, prob_crs):
    """
    实现了随机选出区域，交换区域内所有的点的功能，由于这样的话点数不固定了，population_list为长度为n_pop的数组，每个对象为一个长度为当前解包含点数的np.array。代码可以进一步简化/优化
    """
    # 找到坐标范围
    x_min = np.min([points[:, 0].min() for points in population_list])
    x_max = np.max([points[:, 0].max() for points in population_list])
    y_min = np.min([points[:, 1].min() for points in population_list])
    y_max = np.max([points[:, 1].max() for points in population_list])

    # 判断是否碰上交叉
    for i in range(0, len(population_list), 2):
        if np.random.rand() < prob_crs:
            while True:
                # 随机选择xy范围
                chosen_x_minmax = np.random.uniform(x_min, x_max, 2)
                chosen_x_min = chosen_x_minmax.min()
                chosen_x_max = chosen_x_minmax.max()

                chosen_y_minmax = np.random.uniform(y_min, y_max, 2)
                chosen_y_min = chosen_y_minmax.min()
                chosen_y_max = chosen_y_minmax.max()
                
                parent_1 = population_list[i].copy()
                parent_2 = population_list[i+1].copy()
                mask_1 = ((parent_1[:, 0] >= chosen_x_min) & (parent_1[:, 0] <= chosen_x_max) &
                          (parent_1[:, 1] >= chosen_y_min) & (parent_1[:, 1] <= chosen_y_max))
                mask_2 = ((parent_2[:, 0] >= chosen_x_min) & (parent_2[:, 0] <= chosen_x_max) &
                          (parent_2[:, 1] >= chosen_y_min) & (parent_2[:, 1] <= chosen_y_max))
                if mask_1.any() or mask_2.any():
                    crossover_1 = parent_1[mask_1]
                    crossover_2 = parent_2[mask_2]

                    new_1 = np.delete(parent_1, np.where(mask_1), axis=0)
                    new_1 = np.concatenate([new_1, crossover_2])
                    new_2 = np.delete(parent_2, np.where(mask_2), axis=0)
                    new_2 = np.concatenate([new_2, crossover_1])
                    population_list[i] = new_1
                    population_list[i+1] = new_2
                    break
    return population_list


if __name__ == "__main__":
    np.random.seed(42)
    population_list = [
        np.array([[0.37454012, 0.95071431],
                  [0.73199394, 0.59865848],
                  [0.15601864, 0.15599452],
                  [0.05808361, 0.86617615]]),
        np.array([[0.60111501, 0.70807258],
                  [0.02058449, 0.96990985]])
    ]
    res = region_crossover(population_list, 1)
    print(res)