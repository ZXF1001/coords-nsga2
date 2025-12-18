"""
Coordinate-based NSGA-II algorithm implementation for multi-objective optimization.

This module provides the core classes for solving multi-objective optimization problems
specifically designed for coordinate point layouts using the NSGA-II algorithm.
"""

import pickle
from typing import Callable, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from shapely.geometry import Polygon
from tqdm import trange

from .operators.crossover import coords_crossover, region_crossover
from .operators.mutation import coords_mutation, variable_mutation
from .operators.selection import coords_selection
from .spatial import create_points_in_polygon
from .utils import crowding_distance, fast_non_dominated_sort
from .visualization import Plotting


class Problem:
    """
    Problem definition for coordinate-based multi-objective optimization.
    
    This class encapsulates the problem definition including objectives, constraints,
    number of points, and the optimization region.
    """
    
    def __init__(
        self, 
        objectives: Union[List[Callable], Callable], 
        n_points: Union[int, List[int]], 
        region: Polygon, 
        constraints: Union[List[Callable], Callable, List] = [], 
        penalty_weight: float = 1e6
    ):
        """
        Initialize the optimization problem.
        
        Args:
            objectives: List of objective functions or single function returning multiple values.
                       Each function should take coordinates array (n_points, 2) as input.
            n_points: Fixed number of points (int) or range [min, max] for variable points.
            region: of shape (n_vertices, 2).
            constraints: List of constraint functions or single function returning multiple values.
                        Each function should return penalty value (>0 for violation, 0 for satisfaction).
            penalty_weight: Weight for constraint penalties in objective function.
            
        Returns:
            None
        """
        self.objectives = objectives
        self.n_points = n_points
        self.region = region
        self.constraints = constraints
        self.penalty_weight = penalty_weight

        if isinstance(self.n_points, int):
            self.variable_n_points = False
            self.n_points_fixed = self.n_points  # Store as separate attribute for type safety
        else:
            assert len(self.n_points) == 2, "n_points range must contain exactly 2 values [min, max]"
            self.variable_n_points = True
            self.n_points_min = self.n_points[0]
            self.n_points_max = self.n_points[1]

    def _sample_population(self, pop_size: int) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Sample initial population layouts.
        
        Args:
            pop_size: Population size.
            
        Returns:
            Population as array (pop_size, n_points, 2) for fixed points or 
            list of arrays for variable points.
        """
        if self.variable_n_points:
            n_points_list = np.random.randint(
                self.n_points_min, self.n_points_max + 1, pop_size)
            coords = [create_points_in_polygon(
                self.region, n_p) for n_p in n_points_list]
            return coords
        else:
            # For fixed n_points, use the type-safe attribute
            n_points = self.n_points_fixed
            coords = create_points_in_polygon(
                self.region, pop_size * n_points)
            return coords.reshape(pop_size, n_points, 2)

    def _evaluate_individual(self, individual: np.ndarray) -> np.ndarray:
        """
        Evaluate objective functions for a single individual with constraint penalties.
        
        Args:
            individual: Coordinate array of shape (n_points, 2).
            
        Returns:
            Array of objective values with constraint penalties applied.
        """
        if isinstance(self.objectives, list):
            obj_values = np.array([obj(individual) for obj in self.objectives])
        else:
            obj_values = np.array(self.objectives(individual))
            
        if self.constraints:
            if isinstance(self.constraints, list):
                penalty = self.penalty_weight * \
                    np.sum([c(individual) for c in self.constraints])
            else:
                penalty = self.penalty_weight * \
                    np.sum(self.constraints(individual))
            obj_values -= penalty
        return obj_values

    def evaluate(self, population: Union[np.ndarray, List[np.ndarray]], n_jobs: int = 1) -> np.ndarray:
        """
        Evaluate objective functions for entire population.

        Args:
            population: Population array of shape (pop_size, n_points, 2) or list of arrays.
            n_jobs: Number of parallel jobs (1 for serial computation).

        Returns:
            Objective values array of shape (n_objectives, pop_size).
        """
        if n_jobs == 1:
            results = [self._evaluate_individual(ind) for ind in population]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_individual)(ind) for ind in population
            )

        return np.array(results).T


class CoordsNSGA2:
    """
    Coordinate-based NSGA-II optimizer for multi-objective optimization.
    
    This class implements the NSGA-II algorithm specifically designed for optimizing
    coordinate point layouts with support for variable number of points and parallel computation.
    """
    
    def __init__(
        self, 
        problem: Problem, 
        pop_size: int, 
        prob_crs: float, 
        prob_mut: float, 
        random_seed: int = 42, 
        n_jobs: int = 1
    ):
        """
        Initialize the NSGA-II optimizer.
        
        Args:
            problem: Problem instance defining objectives, constraints, and region.
            pop_size: Population size (must be even number).
            prob_crs: Crossover probability.
            prob_mut: Mutation probability.
            random_seed: Random seed for reproducibility.
            n_jobs: Number of parallel jobs for evaluation (1 for serial).
            
        Returns:
            None
        """
        self.problem = problem
        self.variable_n_points = self.problem.variable_n_points
        self.pop_size = pop_size
        self.prob_crs = prob_crs
        self.prob_mut = prob_mut
        self.n_jobs = n_jobs

        np.random.seed(random_seed)
        assert pop_size % 2 == 0, "pop_size must be even number"
        
        self.P = self.problem._sample_population(pop_size)
        self.values_P = self.problem.evaluate(self.P, n_jobs=self.n_jobs)
        self.P_history = [self.P]
        self.values_history = [self.values_P]

        self.plot = Plotting(self)

        self.selection = coords_selection
        if self.variable_n_points:
            self.n_points_min = problem.n_points_min
            self.n_points_max = problem.n_points_max
            self.crossover = region_crossover
            self.mutation = variable_mutation
        else:
            self.crossover = coords_crossover
            self.mutation = coords_mutation

    def _get_next_population(
        self, 
        R: Union[np.ndarray, List[np.ndarray]],
        population_sorted_in_fronts: List[List[int]],
        crowding_distances: List[np.ndarray]
    ) -> Union[Tuple[np.ndarray, List[int]], Tuple[List[np.ndarray], List[int]]]:
        """
        Select next generation population based on Pareto fronts and crowding distance.
        
        Args:
            R: Combined population from current and offspring generations.
            population_sorted_in_fronts: Indices of solutions grouped by Pareto front.
            crowding_distances: Crowding distances for each front.
            
        Returns:
            Tuple containing:
            - Selected population for next generation
            - Indices of selected individuals in R
        """
        new_idx = []
        for i, front in enumerate(population_sorted_in_fronts):
            remaining_size = self.pop_size - len(new_idx)
            
            if len(front) < remaining_size:
                new_idx.extend(front)
            elif len(front) == remaining_size:
                new_idx.extend(front)
                break
            else:
                crowding_dist = np.array(crowding_distances[i])
                sorted_front_idx = np.argsort(crowding_dist)[::-1]
                sorted_front = np.array(front)[sorted_front_idx]
                new_idx.extend(sorted_front[:remaining_size])
                break
        
        selected_pop = [R[i] for i in new_idx] if self.variable_n_points else R[new_idx]
        return selected_pop, new_idx

    def run(self, gen: int = 1000, verbose: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Execute the NSGA-II optimization algorithm.
        
        Args:
            gen: Number of generations to run.
            verbose: Whether to display progress bar.
            
        Returns:
            Final population of optimal solutions.
        """
        if verbose:
            iterator = trange(gen)
        else:
            iterator = range(gen)

        for _ in iterator:
            Q = self.selection(self.P, self.values_P)
            
            if self.variable_n_points:
                Q = self.crossover(
                    Q, self.prob_crs, self.n_points_min, self.n_points_max)
                Q = self.mutation(
                    Q, self.prob_mut, self.problem.region, self.n_points_min, self.n_points_max)
                assert np.min([len(q) for q in Q]) >= self.n_points_min \
                    and np.max([len(q) for q in Q]) <= self.n_points_max
            else:
                Q = self.crossover(Q, self.prob_crs)
                Q = self.mutation(Q, self.prob_mut, self.problem.region)

            values_Q = self.problem.evaluate(Q, n_jobs=self.n_jobs)

            R = self.P + Q if self.variable_n_points \
                else np.concatenate([self.P, Q])
            values_R = np.concatenate([self.values_P, values_Q], axis=1)

            population_sorted_in_fronts = fast_non_dominated_sort(values_R)
            crowding_distances = [crowding_distance(
                values_R[:, front]) for front in population_sorted_in_fronts]

            self.P, p_idx = self._get_next_population(R,
                                              population_sorted_in_fronts, crowding_distances)

            self.values_P = values_R[:, p_idx]

            self.P_history.append(self.P)
            self.values_history.append(self.values_P)
            
        return self.P

    def save(self, filepath: str) -> None:
        """
        Save the optimizer instance to file using pickle.
        
        Args:
            filepath: Path to save the optimizer instance.
            
        Returns:
            None
            
        Note:
            Ensure that all objective and constraint functions are self-contained
            and do not rely on external global variables for successful loading.
        """
        print("Please ensure that all objective and constraint functions are self-contained. "
              "Functions relying on external global variables may fail to load properly!")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"CoordsNSGA2 instance successfully saved to {filepath}")
        except Exception as e:
            print("Warning: Failed to save CoordsNSGA2 instance. This may be due to unpickleable objects (e.g., lambda functions or nested functions).")
            print(f"Error details: {e}")
            raise

    @classmethod
    def load(cls, filepath: str) -> 'CoordsNSGA2':
        """
        Load optimizer instance from file.
        
        Args:
            filepath: Path to the saved optimizer instance.
            
        Returns:
            Loaded CoordsNSGA2 instance.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        print(f"CoordsNSGA2 instance successfully loaded from {filepath}")
        return instance
