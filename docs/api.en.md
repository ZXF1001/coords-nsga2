# API Reference

> **⚠️ Important Notice**: This document is AI-generated based on source-code analysis. Although we strive for accuracy, inconsistencies or issues may still exist. We are actively improving and validating all content. If you encounter any problems, please report them promptly.

## Core Classes

### Problem

Class that defines a multi-objective optimisation problem.

**Constructor:**
```python
Problem(objectives, n_points, region, constraints=[], penalty_weight=1e6)
```

**Parameters:**
- `objectives` (list[callable]): List of objective functions. Each function takes coords (n_points, 2) and returns a scalar.
- `n_points` (int): Number of coordinate points to optimise.
- `region` (shapely.geometry.Polygon): Shapely polygon defining the feasible region.
- `constraints` (list, optional): List of constraint functions. Default is an empty list.
- `penalty_weight` (float, optional): Weight applied to constraint violations. Default is 1e6.

**Methods:**

#### sample_population(pop_size)
Generate an initial population.

**Parameters:**
- `pop_size` (int): Population size.

**Returns:**
- `numpy.ndarray`: Population array with shape`(pop_size, n_points, 2)`.

#### evaluate(population)
Evaluate the objective values for all individuals in a population.

**Parameters:**
- `population` (numpy.ndarray): Population with shape`(pop_size, n_points, 2)`.

**Returns:**
- `numpy.ndarray`: Array of objective values with shape `(n_objectives, pop_size)`.

**Example:**
```python
from coords_nsga2 import Problem
from coords_nsga2.spatial import region_from_points

# Define objective functions
def obj1(coords):
    return np.sum(coords[:, 0])

def obj2(coords):
    return np.sum(coords[:, 1])

# Create a problem
region = region_from_points([[0, 0], [1, 0], [1, 1], [0, 1]])
problem = Problem(objectives=[obj1, obj2], n_points=5, region=region)

# Generate an initial population
population = problem.sample_population(10)
print(f"Population shape: {population.shape}")  # (10, 5, 2)

# Evaluate the population
values = problem.evaluate(population)  # shape: (n_objectives, pop_size)
print(f"Objective 1 values: {values[0]}")
print(f"Objective 2 values: {values[1]}")
```

### CoordsNSGA2

NSGA-II optimiser for coordinate problems.

**Constructor:**
```python
CoordsNSGA2(problem, pop_size, prob_crs, prob_mut, random_seed=42)
```

**Parameters:**
- `problem` (Problem): Problem instance.
- `pop_size` (int): Population size (must be even).
- `prob_crs` (float): Crossover probability in the range 0-1.
- `prob_mut` (float): Mutation probability in the range 0-1 or -1.
- `random_seed` (int, optional): Random seed. Default is 42.

**Attributes:**
- `P`: Current population, shape `(pop_size, n_points, 2)`.
- `values_P`: Objective values of the current population, shape `(n_objectives, pop_size)`.
- `P_history`: Population history.
- `values_history`: List of objective-value arrays per generation, each with shape `(n_objectives, pop_size)`.

**Methods:**

#### run(generations, verbose=True)
Run the optimization algorithm.

**Parameters:**
- `generations` (int): Number of generations.
- `verbose` (bool, optional): Show progress bar if `True`. Default is `True`.

**Returns:**
- `numpy.ndarray`: Final population.

#### save(path)
Save the optimiser state to a file.

**Parameters:**
- `path` (str): File path to save to.s

#### load(path)
Load the optimiser state from a file.

**Parameters:**
- `path` (str): File path to load from.

**Example:**
```python
from coords_nsga2 import CoordsNSGA2, Problem

# Create an optimiser
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# Run optimisation
result = optimizer.run(1000)

# Save results
optimizer.save("optimization_result.npz")

# Load results
optimizer.load("optimization_result.npz")
```

## Spatial Utilities

### region_from_points(points)

Create a polygon region from a list of points.s

**Parameters:**
- `points` (list): List of points in the form `[[x1, y1], [x2, y2], ...]`.

**Returns:**
- `shapely.geometry.Polygon`: Shapely polygon object.

**Example:**
```python
from coords_nsga2.spatial import region_from_points

# Create a triangular region
points = [[0, 0], [1, 0], [0.5, 1]]
region = region_from_points(points)
print(f"Area: {region.area}")
```

### region_from_range(x_min, x_max, y_min, y_max)

Create a rectangular region from coordinate bounds.

**Parameters:**
- `x_min` (float): Minimum x-coordinate.
- `x_max` (float): Maximum x-coordinate.
- `y_min` (float): Minimum y-coordinate.
- `y_max` (float): Maximum y-coordinate.

**Returns:**
- `shapely.geometry.Polygon`: Shapely rectangle object.

**Example:**
```python
from coords_nsga2.spatial import region_from_range

# Create a rectangular region
region = region_from_range(0, 10, 0, 5)
print(f"Bounds: {region.bounds}")
```

### create_points_in_polygon(polygon, n)

Generate `n` random points inside a polygon.

**Parameters:**
- `polygon` (shapely.geometry.Polygon): Target polygon.
- `n` (int): Number of points to generate.

**Returns:**
- `numpy.ndarray`: Array of coordinates with shape `(n, 2)`.

**Example:**
```python
from coords_nsga2.spatial import create_points_in_polygon, region_from_points

# Create a region
region = region_from_points([[0, 0], [1, 0], [1, 1], [0, 1]])

# Generate random points
points = create_points_in_polygon(region, 10)
print(f"Generated points: {points}")
```

## Genetic Operators

### coords_crossover(population, prob_crs)

Coordinate-specific crossover operator that exchanges subsets of points between parents.

**Parameters:**
- `population` (numpy.ndarray): Population with shape `(pop_size, n_points, 2)`.
- `prob_crs` (float): Crossover probability.

**Returns:**
- `numpy.ndarray`: Population after crossover.

**Algorithm:**
- For each parent pair, perform crossover with probability `prob_crs`.
- Randomly select between 1 and `n_points-1` points to swap.
- Population size remains unchanged.

**Example:**
```python
from coords_nsga2.operators.crossover import coords_crossover

# Perform crossover
new_population = coords_crossover(population, prob_crs=0.5)
```

### coords_mutation(population, prob_mut, region)

Coordinate-specific mutation operator that relocates points randomly within the region.

**Parameters:**
- `population` (numpy.ndarray): Population with shape `(pop_size, n_points, 2)`.
- `prob_mut` (float): Mutation probability.
- `region` (shapely.geometry.Polygon): Feasible region.

**Returns:**
- `numpy.ndarray`: Population after mutation.

**Algorithm:**
- For each coordinate point, mutate with probability `prob_mut`.
- During mutation, generate a new random position within the region.
- Ensure mutated points remain inside the feasible region.

**Example:**
```python
from coords_nsga2.operators.mutation import coords_mutation

# Perform mutation
new_population = coords_mutation(population, prob_mut=0.1, region=region)
```

### coords_selection(population, values_P, tourn_size=3)

Tournament selection based on non-dominated sorting and crowding distance.

**Parameters：**
- `population` (numpy.ndarray): Population with shape `(pop_size, n_points, 2)`.
- `values_P` (numpy.ndarray): Objective values with shape `(n_objectives, pop_size)`.
- `tourn_size` (int, optional): Tournament size. Default is `3`

**Returns:**
- `numpy.ndarray`: Selected population.

**Algorithm:**
- Use fast non-dominated sorting to assign front ranks.
- Compute crowding distance within each front.
- Apply tournament selection, preferring individuals in lower fronts.
- When ranks tie, choose individuals with larger crowding distance.

**Example:**
```python
from coords_nsga2.operators.selection import coords_selection

# Perform selection
selected_population = coords_selection(population, values_P, tourn_size=3)
```

## Utility Functions

### fast_non_dominated_sort(objectives)

Fast non-dominated sorting algorithm.

**Parameters:**
- `objectives` (numpy.ndarray): Objective values with shape `(n_objectives, pop_size)`.

**Returns:**
- `list`: List of fronts; each front contains the indices of individuals in that front.

**Algorithm:**
- Count how many times each individual is dominated.
- Record which individuals each one dominates.
- Return indices grouped by front.
- Sort individuals by front rank.

**Example:**
```python
from coords_nsga2.utils import fast_non_dominated_sort

# Perform non-dominated sorting
fronts = fast_non_dominated_sort(values)
print(f"Number of fronts: {len(fronts)}")
for i, front in enumerate(fronts):
    print(f"Front {i}: {front}")
```

### crowding_distance(objectives)

Compute crowding distance.

**Parameters:**
- `objectives` (numpy.ndarray): Objective values with shape `(n_objectives, pop_size)`.

**Returns:**
- `numpy.ndarray`: Array of crowding distances.

**Algorithm**
- Sort individuals for each objective.
- Set boundary points’ crowding distance to infinity.
- For interior points, sum the normalised differences of adjacent objective values.
- Return distances in the original order.

**Example:**
```python
from coords_nsga2.utils import crowding_distance

# Compute crowding distance (pass a single front when needed)
distances = crowding_distance(values)
print(f"Crowding distances: {distances}")
```

## Complete Example

```python
import numpy as np
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points
from coords_nsga2.utils import fast_non_dominated_sort, crowding_distance

# 1. Define the problem
def objective_1(coords):
    """Maximise the sum of x coordinates"""
    return np.sum(coords[:, 0])

def objective_2(coords):
    """Maximise the sum of y coordinates"""
    return np.sum(coords[:, 1])

def constraint(coords):
    """Minimum spacing constraint"""
    dist_list = distance.pdist(coords)
    penalty = np.sum(0.1 - dist_list[dist_list < 0.1])
    return penalty

# 2. Create region and problem
region = region_from_points([[0, 0], [1, 0], [1, 1], [0, 1]])
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=5,
    region=region,
    constraints=[constraint]
)

# 3. Create the optimiser
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 4. Run optimisation
result = optimizer.run(100)

# 5. Analyse results
print(f"Final population shape: {result.shape}")
print(f"Objective 1 values: {optimizer.values_P[0]}")
print(f"Objective 2 values: {optimizer.values_P[1]}")

# 6. Find the Pareto front
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_front = result[fronts[0]]  # The first front is the Pareto front
print(f"Number of Pareto-optimal solutions: {len(pareto_front)}")
```
