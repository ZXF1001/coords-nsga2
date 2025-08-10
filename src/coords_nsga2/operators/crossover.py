import numpy as np


def coords_crossover(population, prob_crs, n_points):
    # 坐标的交叉算子
    for i in range(0, len(population), 2):
        if np.random.rand() < prob_crs:
            cross_num = np.random.randint(1, n_points)
            cross_idx = np.random.choice(
                n_points, cross_num, replace=False)
            temp = population[i][cross_idx]
            population[i][cross_idx] = population[i+1][cross_idx]
            population[i+1][cross_idx] = temp
    return population
