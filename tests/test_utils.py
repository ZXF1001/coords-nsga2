import numpy as np
from shapely.geometry import Polygon

from coords_nsga2.spatial import create_points_in_polygon
from coords_nsga2.utils import crowding_distance, fast_non_dominated_sort

values_array = np.array([
    [-1, 3, -4.2, 8.2, -2, 5, 10.1, 0.2],
    [12.1, 7, 18.75, 8, 10, 11, 1.9, 5.1]])


def test_fast_non_dominated_sort():
    res = fast_non_dominated_sort(values_array)
    assert res == [[0, 2, 3, 5, 6], [1, 4], [7]]


def test_crowding_distance_1():
    front_idx = [0, 2, 3, 5, 6]
    res = crowding_distance(values_array[:, front_idx])
    assert np.allclose(res, [1.1032973, np.inf, 0.8967027, 0.88668009, np.inf])


def test_crowding_distance_2():
    res = crowding_distance(np.array([[1, 2], [2, 0]]))
    assert np.all(res == [np.inf, np.inf])


# def test_crowding_distance_3():
#     res = crowding_distance([2, 1, 3, 1, 2, 1, 3],
#                             [0, 1, -1, 1, 0, 1, -1])
#     assert np.all(res == [1, np.inf, 1, 0, 1, 1, np.inf])


def test_create_points_in_polygon():
    polygon = Polygon([[0, 0],
                       [2, 0],
                       [1, 3]])
    points = create_points_in_polygon(polygon, 10)
    assert len(points) == 10


if __name__ == '__main__':
    test_fast_non_dominated_sort()
    test_crowding_distance_1()
    test_crowding_distance_2()
    # test_crowding_distance_3()
    test_create_points_in_polygon()
