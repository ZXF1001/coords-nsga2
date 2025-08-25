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
    def objectives(coords):
        center = np.array([1.0, 1.0])  # Region center
        distances = np.linalg.norm(coords - center, axis=1)
        dist_matrix = distance.pdist(coords)

        obj_1 = np.sum(coords[:, 0]) + np.sum(coords[:, 1])
        obj_2 = np.std(coords[:, 0]) + np.std(coords[:, 1])
        obj_3 = -np.mean(distances)  # Negative for maximization
        obj_4 = np.min(dist_matrix)
        return [obj_1, obj_2, obj_3, obj_4]

    def constraints(coords):
        min_spacing = 0.1  # 间距限制
        """Minimum spacing constraint between points"""
        if len(coords) < 2:
            return 0
        dist_list = distance.pdist(coords)
        p1 = np.sum(min_spacing - dist_list[dist_list < min_spacing])
        x_coords = coords[:, 0]
        p2 = np.sum(0.5-x_coords[x_coords < 0.5])
        return [p1, p2]

    problem = Problem(objectives=objectives,
                      n_points=10,
                      region=region,
                      constraints=constraints)

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


if __name__ == '__main__':
    test_main()
