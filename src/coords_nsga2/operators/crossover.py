import numpy as np


def coords_crossover(population, prob_crs):
    """坐标交叉算子"""
    n_points = population.shape[1]
    for i in range(0, len(population), 2):
        if np.random.rand() < prob_crs:
            cross_num = np.random.randint(1, n_points)
            cross_idx = np.random.choice(n_points, cross_num, replace=False)
            population[i:i+2, cross_idx] = population[i:i+2, cross_idx][::-1]
    return population


def region_crossover(population_list, prob_crs, n_points_min, n_points_max, max_attempts=100):
    """
    区域交叉算子    
    Args:
        population_list: 长度为n_pop的列表，每个元素为包含坐标点的np.array
        prob_crs: 交叉概率
        max_attempts: 最大尝试次数，防止无限循环

    Returns:
        交叉后的种群列表
    """

    # 使用向量化操作一次性计算全局边界，比原来的列表推导式更高效
    all_points = np.vstack(population_list)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    # 创建副本避免修改原始数据
    result = [individual.copy() for individual in population_list]

    # 成对处理个体进行交叉
    for i in range(0, len(result), 2):
        if np.random.rand() < prob_crs:
            parent1, parent2 = result[i], result[i + 1]

            # 尝试找到有效的交叉区域
            for attempt in range(max_attempts):
                # 使用np.sort直接生成有序坐标，避免min/max调用
                x_coords = np.sort(np.random.uniform(x_min, x_max, 2))
                y_coords = np.sort(np.random.uniform(y_min, y_max, 2))
                region_x_min, region_x_max = x_coords[0], x_coords[1]
                region_y_min, region_y_max = y_coords[0], y_coords[1]

                # 创建区域掩码
                mask1 = ((parent1[:, 0] >= region_x_min) & (parent1[:, 0] <= region_x_max) &
                         (parent1[:, 1] >= region_y_min) & (parent1[:, 1] <= region_y_max))
                mask2 = ((parent2[:, 0] >= region_x_min) & (parent2[:, 0] <= region_x_max) &
                         (parent2[:, 1] >= region_y_min) & (parent2[:, 1] <= region_y_max))

                # 如果至少有一个个体在区域内有点，则进行交叉
                if mask1.any() or mask2.any():
                    # 提取区域内的点
                    region1 = parent1[mask1]
                    region2 = parent2[mask2]

                    result_1 = np.vstack(
                        [parent1[~mask1], region2]) if region2.size > 0 else parent1[~mask1]
                    result_2 = np.vstack(
                        [parent2[~mask2], region1]) if region1.size > 0 else parent2[~mask2]
                    # 限制点的范围
                    if len(result_1) >= n_points_min and len(result_1) <= n_points_max and len(result_2) >= n_points_min and len(result_2) <= n_points_max:
                        result[i] = result_1
                        result[i + 1] = result_2
                        break

    return result


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
    res = region_crossover(population_list, 1, 1, 5)
    print(res)
