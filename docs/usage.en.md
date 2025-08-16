# Usage Guide

> **⚠️ Important Notice**: This document is AI-generated based on source-code analysis. Although we strive for accuracy, inconsistencies or issues may still exist. We are actively improving and validating all content. If you encounter any problems, please report them promptly.

## English Usage Guide

### Basic Concepts

The core concepts of Coords-NSGA2 library include:

1. **Problem**: Defines the optimization problem's objective functions, constraints, and search region
2. **CoordsNSGA2**: The optimizer that executes the NSGA-II algorithm
3. **Region**: Defines the valid search space for coordinate points
4. **Constraints**: Conditions that limit the feasibility of solutions

### Quick Start Example

Here's a complete usage example demonstrating how to optimize the layout of 10 coordinate points:

```python
import numpy as np
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 1. Define optimization region (polygon)
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# 2. Define objective functions
def objective_1(coords):
    """First objective: maximize coordinate sum"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """Second objective: maximize point spread"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# 3. Define constraints
spacing = 0.05  # minimum spacing
def constraint_1(coords):
    """Constraint: minimum spacing between points"""
    dist_list = distance.pdist(coords)
    penalty_list = spacing - dist_list[dist_list < spacing]
    return np.sum(penalty_list)

# 4. Create problem instance (supports arbitrary number of objectives)
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

# 5. Create optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 6. Run optimization
result = optimizer.run(1000)

# 7. View results
print(f"Optimization complete! Result shape: {result.shape}")
print(f"Population size: {len(result)}")
print(f"Points per solution: {result.shape[1]}")
```

### Region Definition

#### Create polygon region from point list

```python
from coords_nsga2.spatial import region_from_points

# Define polygon vertices
points = [
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
]
region = region_from_points(points)
```

#### Create rectangular region from coordinate bounds

```python
from coords_nsga2.spatial import region_from_range

# Define rectangle bounds
region = region_from_range(x_min=0, x_max=10, y_min=0, y_max=5)
```

### Objective Function Definition

Objective functions should accept a numpy array of shape `(n_points, 2)` as input and return a scalar value:

```python
def my_objective(coords):
    """
    Parameters:
        coords: numpy array of shape (n_points, 2)
                each row is a coordinate point [x, y]
    
    Returns:
        float: objective function value
    """
    # Example: calculate average distance to origin
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    return np.mean(distances)
```

### Constraint Definition

Constraint functions should return penalty values for constraint violations. Return 0 if no constraints are violated:

```python
def my_constraint(coords):
    """
    Parameters:
        coords: numpy array of shape (n_points, 2)
    
    Returns:
        float: penalty value for constraint violation (0 means no violation)
    """
    # Example: ensure all points are within unit circle
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    violations = distances[distances > 1] - 1
    return np.sum(violations)
```

### Optimizer Parameters

#### CoordsNSGA2 Parameter Description

- `problem`: Problem instance
- `pop_size`: Population size (must be even)
- `prob_crs`: Crossover probability (between 0-1)
- `prob_mut`: Mutation probability (between 0-1)
- `random_seed`: Random seed (for reproducibility)

#### Parameter Tuning Suggestions

- **Population size**: Usually set to 20-100, use larger populations for complex problems
- **Crossover probability**: Usually set to 0.5-0.9
- **Mutation probability**: Usually set to 0.01-0.1
- **Generations**: Set based on problem complexity, usually 100-1000 generations

### Result Analysis

After optimization is complete, you can access the following attributes:

```python
# Final population
final_population = optimizer.P

# Objective function values (shape: n_objectives × pop_size)
values = optimizer.values_P
values1 = values[0]
values2 = values[1]

# Optimization history
population_history = optimizer.P_history
values_history = optimizer.values_history  # list of (n_objectives, pop_size) per generation

# Find Pareto optimal solutions (based on last generation objective values)
from coords_nsga2.utils import fast_non_dominated_sort
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_front = optimizer.P[fronts[0]]
```

### Save and Load

```python
# Save optimization state
optimizer.save("optimization_result.npz")

# Load optimization state
optimizer.load("optimization_result.npz")
```