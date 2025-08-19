import numpy as np

from coords_nsga2.operators.crossover import coords_crossover, region_crossover


def test_coords_crossover():
    population = np.random.random((10, 5, 2))
    result = coords_crossover(population, 1)
    assert result.shape == population.shape


def test_region_crossover():
    population_list = [
        np.array([[0.37454012, 0.95071431],
                  [0.73199394, 0.59865848],
                  [0.15601864, 0.15599452],
                  [0.05808361, 0.86617615]]),
        np.array([[0.60111501, 0.70807258],
                  [0.02058449, 0.96990985]])
    ]
    n_points_min = 1
    n_points_max = 5
    result = region_crossover(population_list, 1, n_points_min, n_points_max)
    assert len(result) == len(population_list)
    assert np.min([len(q) for q in result]) >= n_points_min \
        and np.max([len(q) for q in result]) <= n_points_max
