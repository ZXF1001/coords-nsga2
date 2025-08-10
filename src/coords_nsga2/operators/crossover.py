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
