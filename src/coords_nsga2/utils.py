from typing import List

import numpy as np


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Performs fast non-dominated sorting on a population's objective values.

    Args:
        objectives: A NumPy array of objective values with shape (n_objectives, pop_size).

    Returns:
        A list of lists, where each inner list represents a Pareto front
        and contains the indices of the solutions belonging to that front.
    """
    n_objectives, num_population = objectives.shape
    dominated_solutions = [[] for _ in range(num_population)]
    domination_count = np.zeros(num_population)
    ranks = np.zeros(num_population)
    fronts = [[]]

    for p in range(num_population):
        for q in range(num_population):
            if p == q:
                continue
            
            p_dominates_q = True
            q_dominates_p = True
            p_better_in_at_least_one = False
            q_better_in_at_least_one = False

            for obj_func_idx in range(n_objectives):
                if objectives[obj_func_idx, p] > objectives[obj_func_idx, q]:
                    q_dominates_p = False
                    p_better_in_at_least_one = True
                elif objectives[obj_func_idx, p] < objectives[obj_func_idx, q]:
                    p_dominates_q = False
                    q_better_in_at_least_one = True

            if p_dominates_q and p_better_in_at_least_one:
                dominated_solutions[p].append(q)
            elif q_dominates_p and q_better_in_at_least_one:
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    current_rank = 0
    while fronts[current_rank]:
        next_front = []
        for p in fronts[current_rank]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = current_rank + 1
                    next_front.append(q)
        current_rank += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """
    Calculates the crowding distance for solutions within a Pareto front.

    Args:
        objectives: A NumPy array of objective values for the solutions in a front,
                    with shape (n_objectives, n_individuals).

    Returns:
        A NumPy array of crowding distances for each individual in the front.
    """
    n_objectives, n_individuals = objectives.shape
    
    crowding_distances = np.zeros(n_individuals)
    
    for obj_values in objectives:
        sorted_indices = np.argsort(obj_values)
        sorted_values = obj_values[sorted_indices]
        
        obj_range = sorted_values[-1] - sorted_values[0]
        
        if obj_range == 0:
            continue
        
        distances = np.zeros(n_individuals)
        
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        if n_individuals > 2:
            contributions = (sorted_values[2:] - sorted_values[:-2]) / obj_range
            middle_indices = sorted_indices[1:-1]
            distances[middle_indices] = contributions
        
        crowding_distances += distances
    
    return crowding_distances
