import numpy as np
from scipy.spatial import distance
from shapely.geometry import MultiPolygon

from coords_nsga2 import coords_nsga2
from coords_nsga2.spatial import region_from_points


def test_main():

    # 创建边界
    polygon = region_from_points([
        [0, 0],
        [1, 0],
        [2, 1],
        [1, 1],
    ])
    multi_polygon = MultiPolygon([polygon])

    # 定义目标函数1：更靠近右上方
    def objective_1(coords):
        return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

    # 定义目标函数2：布局更分散
    def objective_2(coords):
        return np.std(coords[:, 0]) + np.std(coords[:, 1])

    spacing = 0.05  # 间距限制

    def constraint_1(coords):
        dist_list = distance.pdist(coords)
        penalty_list = spacing-dist_list[dist_list < spacing]
        penalty_sum = np.sum(penalty_list)
        return penalty_sum

    optimizer = coords_nsga2(func1=objective_1,
                             func2=objective_2,
                             pop_size=20,
                             n_points=10,
                             prob_crs=0.5,
                             prob_mut=0.1,
                             polygons=multi_polygon,  # todo: 这里自动判定是单个多边形还是多个多边形
                             constraints=[constraint_1],
                             random_seed=10,
                             is_int=False)
    result = optimizer.run(100)
    # 断言result存在
    assert len(result) == 20


if __name__ == '__main__':
    test_main()
