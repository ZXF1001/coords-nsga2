import numpy as np

from coords_nsga2.utils import crowding_distance, fast_non_dominated_sort

v1_array = np.array([-1, 3, -4.2, 8.2, -2, 5, 10.1, 0.2])
v2_array = np.array([12.1, 7, 18.75, 8, 10, 11, 1.9, 5.1])


def test_fast_non_dominated_sort():
    res = fast_non_dominated_sort(v1_array, v2_array)
    assert res == [[0, 2, 3, 5, 6], [1, 4], [7]]


def test_crowding_distance_1():
    front_idx = [0, 2, 3, 5, 6]
    res = crowding_distance(v1_array[front_idx], v2_array[front_idx])
    assert np.allclose(res, [1.1032973, np.inf, 0.8967027, 0.88668009, np.inf])


def test_crowding_distance_2():
    res = crowding_distance([1, 2], [2, 0])
    assert np.all(res == [np.inf, np.inf])

def test_crowding_distance_3():
    res = crowding_distance([2, 1, 3, 1, 2, 1, 3], 
                            [0, 1, -1,1, 0, 1, -1])
    assert np.all(res == [1, np.inf, 1, 0, 1, 1, np.inf])

    

if __name__ == '__main__':
    test_fast_non_dominated_sort()
    test_crowding_distance_1()
    test_crowding_distance_2()
    test_crowding_distance_3()
