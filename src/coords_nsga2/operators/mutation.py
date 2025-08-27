from typing import List

import numpy as np
from shapely.geometry import Polygon

from ..spatial import create_points_in_polygon


def coords_mutation(
    population: np.ndarray, 
    prob_mut: float, 
    region: Polygon
) -> np.ndarray:
    """
    Coordinate mutation operator that mutates individual coordinates within region.

    Args:
        population: NumPy array of shape (n_individuals, n_points, 2).
        prob_mut: Mutation probability for each coordinate. If -1, auto-set as 1/n_points.
        region: Shapely Polygon defining valid regions.

    Returns:
        Mutated population array.
    """
    if prob_mut == -1:
        prob_mut_actual = 1/population.shape[1]
    else:
        prob_mut_actual = prob_mut
    
    # Generate mutation mask
    mutation_mask = np.random.random(population.shape[:-1]) < prob_mut_actual

    # Count mutations needed
    n_mutations = int(np.sum(mutation_mask))

    if n_mutations > 0:
        # Generate all new points at once
        new_points = create_points_in_polygon(region, n_mutations)

        # Apply mutations using mask
        population[mutation_mask] = new_points

    return population


def variable_mutation(
    population: List[np.ndarray], 
    prob_mut: float, 
    region: Polygon, 
    n_points_min: int, 
    n_points_max: int
) -> List[np.ndarray]:
    """
    Variable mutation operator for populations with variable number of points.
    
    Each point can undergo one of three mutations:
    1. Point removal (if above minimum)
    2. Point replacement 
    3. Point replacement with addition (if below maximum)

    Args:
        population: List of NumPy arrays, each representing an individual's coordinate points.
        prob_mut: Mutation probability for each point. If -1, auto-set as 1/average_n_points.
        region: Shapely Polygon defining valid regions.
        n_points_min: Minimum number of points allowed for an individual.
        n_points_max: Maximum number of points allowed for an individual.

    Returns:
        Mutated population list.
    """
    if prob_mut == -1:
        total_points = sum(len(ind) for ind in population)
        avg_points = total_points / len(population) if len(population) > 0 else 1
        prob_mut = 1 / avg_points

    new_population = []
    for ind in population:
        current_n_points = len(ind)
        temp_ind_points = []
        
        for point in ind:
            random_status = np.random.random()
            if random_status < 1/3 * prob_mut:
                # Point removal (1/3 probability)
                if current_n_points > n_points_min:
                    current_n_points -= 1
                    # Point is removed (not added to temp_ind_points)
                else:
                    # Cannot remove, keep original point
                    temp_ind_points.append(point)
            elif random_status < 2/3 * prob_mut:
                # Point replacement (1/3 probability)
                new_point = create_points_in_polygon(region, 1)
                temp_ind_points.append(new_point[0])
            elif random_status < prob_mut:
                # Point replacement with addition (1/3 probability)
                if current_n_points < n_points_max:
                    current_n_points += 1
                    new_points = create_points_in_polygon(region, 2).tolist()
                    temp_ind_points += new_points
                else:
                    # Cannot add, just replace
                    new_point = create_points_in_polygon(region, 1)
                    temp_ind_points.append(new_point[0])
            else:
                # No mutation
                temp_ind_points.append(point)
                
        new_population.append(np.array(temp_ind_points))
    return new_population


if __name__ == "__main__":
    from coords_nsga2.spatial import region_from_range
    np.random.seed(42)
    population_list = [
        np.array([[0.37454012, 0.95071431],
                  [0.73199394, 0.59865848],
                  [0.15601864, 0.15599452],
                  [0.05808361, 0.86617615]]),
        np.array([[0.60111501, 0.70807258],
                  [0.02058449, 0.96990985]])
    ]
    res = variable_mutation(
        population_list, 1, region_from_range(0, 1, 0, 1), 1, 4)
    print(res)
