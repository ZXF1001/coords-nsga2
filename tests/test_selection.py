import numpy as np

from coords_nsga2.operators.selection import coords_selection


def test_coords_selection_no_variable():
    population = np.random.random((10, 5, 2))
    values_P = np.random.random((3, 10))
    result = coords_selection(population, values_P)
    assert len(result) == len(population)
