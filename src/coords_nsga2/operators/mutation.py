import numpy as np
from ..spatial import create_point_in_polygon

def coords_mutation(population, prob_mut, n_points, polygons, is_int=False):
    # 坐标的变异算子，这里做小修改没有把整个个体变异，而是对每个坐标进行变异，因为这样才可以逐渐产生满足约束的解，如果对整个个体变异大概率会违反约束
    for i in range(len(population)):
        for j in range(n_points):
            if np.random.rand() < prob_mut:
                x, y = create_point_in_polygon(polygons, is_int)
                population[i][j] = np.array([x, y])
    return population
