from typing import List, Union

import numpy as np

from ..utils import crowding_distance, fast_non_dominated_sort


def coords_selection(
    population: Union[List[np.ndarray], np.ndarray], 
    objective_values: np.ndarray, 
    tourn_size: int = 3
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Performs tournament selection based on non-dominated sorting and crowding distance.

    Args:
        population: The current population of solutions. Can be a list of NumPy arrays
                    (for variable number of points) or a single NumPy array
                    (for fixed number of points).
        objective_values: A NumPy array of objective values for the population,
                          with shape (n_objectives, pop_size).
        tourn_size: Tournament size. Defaults to 3.

    Returns:
        The selected offspring population.
    """
    pop_size = len(population)
    
    # 1. Perform fast non-dominated sorting and crowding distance calculation
    population_sorted_in_fronts = fast_non_dominated_sort(objective_values)
    crowding_distances = [crowding_distance(
        objective_values[:, front]) for front in population_sorted_in_fronts]
    
    # Create a comparison table: [original_index, front_rank, crowding_distance]
    compare_table = []
    for i, front in enumerate(population_sorted_in_fronts):
        for j, idx in enumerate(front):
            compare_table.append([idx, i, crowding_distances[i][j]])
    
    # Sort the comparison table by original index
    compare_table = np.array(compare_table)
    compare_table = compare_table[compare_table[:, 0].argsort()]

    # 2. Generate random indices for tournament aspirants
    aspirants_idx = np.random.randint(
        pop_size, size=(pop_size, tourn_size))

    # 3. Select the best solution from each tournament
    # The best solution is determined by front rank (lower is better),
    # then by crowding distance (higher is better if ranks are equal).
    candidates = compare_table[aspirants_idx]
    # np.lexsort sorts by the last column first, then the second to last, etc.
    # We want to sort by front rank (ascending) then crowding distance (descending).
    # So, we use -candidates[..., 2] for descending crowding distance.
    sorted_indices = np.lexsort((-candidates[..., 2], candidates[..., 1]))
    
    # Get the index of the best candidate within each tournament
    Q_idx = aspirants_idx[np.arange(pop_size), sorted_indices[:, 0]]

    # Return the selected individuals
    return [population[idx] for idx in Q_idx] if isinstance(population, list) else population[Q_idx]


if __name__ == "__main__":
    np.random.seed(42)
    population_list = [
        np.array([[0.37454012, 0.95071431],
                  [0.73199394, 0.59865848],
                  [0.15601864, 0.15599452],
                  [0.05808361, 0.86617615]]),
        np.array([[0.60111501, 0.70807258],
                  [0.02058449, 0.96990985]])
    ]
    values_P = np.array([[4, 5], [2, 3], [3, 6]])
    res = coords_selection(population_list, values_P)
    print(res)
