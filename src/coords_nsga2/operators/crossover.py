from typing import List

import numpy as np


def coords_crossover(population: np.ndarray, prob_crs: float) -> np.ndarray:
    """
    Performs coordinate-based crossover on a population with a fixed number of points.
    
    Args:
        population: A NumPy array of shape (pop_size, n_points, 2) representing the population.
        prob_crs: Crossover probability.
        
    Returns:
        The population after applying crossover.
    """
    n_points = population.shape[1]
    for i in range(0, len(population), 2):
        if np.random.rand() < prob_crs:
            cross_num = np.random.randint(1, n_points)
            cross_idx = np.random.choice(n_points, cross_num, replace=False)
            population[i:i+2, cross_idx] = population[i:i+2, cross_idx][::-1]
    return population


def region_crossover(
    population_list: List[np.ndarray], 
    prob_crs: float, 
    n_points_min: int, 
    n_points_max: int, 
    max_attempts: int = 100
) -> List[np.ndarray]:
    """
    Performs region-based crossover on a population with a variable number of points.
    
    Args:
        population_list: A list of NumPy arrays, where each array represents an individual's
                         coordinate points.
        prob_crs: Crossover probability.
        n_points_min: Minimum number of points allowed for an individual.
        n_points_max: Maximum number of points allowed for an individual.
        max_attempts: Maximum number of attempts to find a valid crossover region.
        
    Returns:
        The population list after applying region crossover.
    """
    all_points = np.vstack(population_list)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    result = [individual.copy() for individual in population_list]

    for i in range(0, len(result), 2):
        if np.random.rand() < prob_crs:
            parent1, parent2 = result[i], result[i + 1]

            for attempt in range(max_attempts):
                x_coords = np.sort(np.random.uniform(x_min, x_max, 2))
                y_coords = np.sort(np.random.uniform(y_min, y_max, 2))
                region_x_min, region_x_max = x_coords[0], x_coords[1]
                region_y_min, region_y_max = y_coords[0], y_coords[1]

                mask1 = ((parent1[:, 0] >= region_x_min) & (parent1[:, 0] <= region_x_max) &
                         (parent1[:, 1] >= region_y_min) & (parent1[:, 1] <= region_y_max))
                mask2 = ((parent2[:, 0] >= region_x_min) & (parent2[:, 0] <= region_x_max) &
                         (parent2[:, 1] >= region_y_min) & (parent2[:, 1] <= region_y_max))

                if mask1.any() or mask2.any():
                    region1 = parent1[mask1]
                    region2 = parent2[mask2]

                    result_1 = np.vstack(
                        [parent1[~mask1], region2]) if region2.size > 0 else parent1[~mask1]
                    result_2 = np.vstack(
                        [parent2[~mask2], region1]) if region1.size > 0 else parent2[~mask2]
                    
                    if len(result_1) >= n_points_min and len(result_1) <= n_points_max and \
                       len(result_2) >= n_points_min and len(result_2) <= n_points_max:
                        result[i] = result_1
                        result[i + 1] = result_2
                        break

    return result


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
    res = region_crossover(population_list, 1, 1, 5)
    print(res)
