# 使用

Below is a minimal example demonstrating how to run a coordinate-based NSGA-II optimization using this library:

```python
import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# Define the optimization regions
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# Define your objective functions
def objective_1(coords):
    # coords is a array of (x, y) points
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# Define constraints if needed
spacing = 0.05  # spacing constraint
def constraint_1(coords):
    # Return total penalty
    dist_list = distance.pdist(coords)
    penalty_list = spacing-dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

# Setup the problem
problem = Problem(func1=objective_1,
                  func2=objective_2,
                  n_points=10,
                  region=region,
                  constraints=[constraint_1])

# Initialize the optimizer
optimizer = CoordsNSGA2(problem=problem,
                        pop_size=20,
                        prob_crs=0.5,
                        prob_mut=0.1,
                        verbose=True)  # Set to False to disable progress bar

# Run optimization
result = optimizer.run(1000)

# Optimization results
print(result)
```