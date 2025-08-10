import numpy as np
from ..spatial import create_point_in_polygon

def coords_mutation(population, prob_mut, polygons, is_int=False):
    """Coordinate mutation operator that mutates individual coordinates within polygons.
    
    Args:
        population: numpy array of shape (n_individuals, n_points, 2)
        prob_mut: mutation probability for each coordinate
        polygons: list of polygons defining valid regions
        is_int: whether to generate integer coordinates
    
    Returns:
        Mutated population array
    """
    # Generate mutation mask
    mutation_mask = np.random.random(population.shape[:-1]) < prob_mut
    
    # Count mutations needed
    n_mutations = np.sum(mutation_mask)
    
    if n_mutations > 0:
        # Generate all new points at once
        new_points = np.array([create_point_in_polygon(polygons, is_int) 
                             for _ in range(n_mutations)])
        
        # Apply mutations using mask
        population[mutation_mask] = new_points

    return population
