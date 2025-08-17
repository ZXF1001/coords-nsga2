import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points


def test_main():

    # 创建边界
    region = region_from_points([
        [0, 0],
        [2, 0],
        [2, 1.5],
        [1, 2],
        [0, 1.5],
    ])

    # Define multiple objective functions
    def objective_1(coords):
        """Maximize sum of x and y coordinates (prefer upper-right)"""
        return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

    def objective_2(coords):
        """Maximize layout dispersion"""
        return np.std(coords[:, 0]) + np.std(coords[:, 1])

    def objective_3(coords):
        """Minimize distance to center"""
        center = np.array([1.0, 1.0])  # Region center
        distances = np.linalg.norm(coords - center, axis=1)
        return -np.mean(distances)  # Negative for maximization

    def objective_4(coords):
        """Maximize minimum distance between points"""
        if len(coords) < 2:
            return 0
        dist_matrix = distance.pdist(coords)
        return np.min(dist_matrix)

    min_spacing = 0.1  # 间距限制

    def constraint_spacing(coords):
        """Minimum spacing constraint between points"""
        if len(coords) < 2:
            return 0
        dist_list = distance.pdist(coords)
        violations = min_spacing - dist_list[dist_list < min_spacing]
        return np.sum(violations)

    problem = Problem(objectives=[objective_1, objective_2, objective_3, objective_4],
                      n_points=10,
                      region=region,
                      constraints=[constraint_spacing])

    optimizer = CoordsNSGA2(problem=problem,
                            pop_size=20,
                            prob_crs=0.7,
                            prob_mut=0.1)

    result = optimizer.run(100)
    # 断言result存在
    assert len(result) == 20

    # 1. Pareto Front Visualizations
    optimizer.plot.pareto_front(obj_indices=[0, 1], is_show=False)  # 2D
    optimizer.plot.pareto_front(obj_indices=[0, 1, 2], is_show=False)  # 3D

    # 4. Objective Optimal Layouts
    optimizer.plot.optimal_coords(obj_indices=0, is_show=False)


if __name__ == '__main__':
    test_main()
