import numpy as np

from ..spatial import create_points_in_polygon


def coords_mutation(population, prob_mut, region):
    """Coordinate mutation operator that mutates individual coordinates within region.

    Args:
        population: numpy array of shape (n_individuals, n_points, 2)
        prob_mut: mutation probability for each coordinate (-1 is auto set as 1/N_points)
        region: region defining valid regions

    Returns:
        Mutated population array
    """
    if prob_mut == -1:
        prob_mut = 1/population.shape[1]
    # Generate mutation mask
    mutation_mask = np.random.random(population.shape[:-1]) < prob_mut

    # Count mutations needed
    n_mutations = np.sum(mutation_mask)

    if n_mutations > 0:
        # Generate all new points at once
        new_points = create_points_in_polygon(region, n_mutations)

        # Apply mutations using mask
        population[mutation_mask] = new_points

    return population
