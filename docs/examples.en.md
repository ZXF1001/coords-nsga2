# Example Code

> **⚠️ Important Notice**: This document is AI-generated based on source-code analysis. Although we strive for accuracy, inconsistencies or issues may still exist. We are actively improving and validating all content. If you encounter any problems, please report them promptly.

## Basic Examples

### 1. Simple Rectangular Region Optimization

```python
import numpy as np
import matplotlib.pyplot as plt
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_range

# Define a rectangular region
region = region_from_range(0, 10, 0, 5)

# Define objective functions
def objective_1(coords):
    """Maximize the sum of x-coordinates"""
    return np.sum(coords[:, 0])

def objective_2(coords):
    """Maximize the sum of y-coordinates"""
    return np.sum(coords[:, 1])

# Create the problem (supports multiple objectives; two objectives here)
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=8,
    region=region
)

# Create the optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# Run the optimization
result = optimizer.run(500)

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot the final population
plt.subplot(1, 2, 1)
for i in range(len(result)):
    plt.scatter(result[i, :, 0], result[i, :, 1], alpha=0.6)
plt.title('Final Population')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Plot objective-function values
plt.subplot(1, 2, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1])
plt.title('Objective Function Values')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. Polygonal Region Optimization with Constraints

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# Define a polygonal region
region = region_from_points([
    [0, 0],
    [2, 0],
    [3, 1],
    [2, 2],
    [0, 2],
    [-1, 1]
])

# Define objective functions
def objective_1(coords):
    """Maximize the distance to the origin"""
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    return np.mean(distances)

def objective_2(coords):
    """Maximize the dispersion among points"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# Define constraints
def constraint_min_spacing(coords):
    """Minimum spacing constraint"""
    dist_list = distance.pdist(coords)
    min_spacing = 0.5
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

def constraint_max_spacing(coords):
    """Maximum spacing constraint"""
    dist_list = distance.pdist(coords)
    max_spacing = 3.0
    violations = dist_list[dist_list > max_spacing] - max_spacing
    return np.sum(violations)

# Create the problem
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=6,
    region=region,
    constraints=[constraint_min_spacing, constraint_max_spacing]
)

# Create the optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=30,
    prob_crs=0.7,
    prob_mut=0.05
)

# Run the optimization
result = optimizer.run(800)

# Visualize the results
plt.figure(figsize=(15, 5))

# Plot the region and final population
plt.subplot(1, 3, 1)
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.2, fc='gray', ec='black', label='Region')

for i in range(len(result)):
    plt.scatter(result[i, :, 0], result[i, :, 1], alpha=0.6)
plt.title('Final Population in Region')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# Plot objective-function values
plt.subplot(1, 3, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1])
plt.title('Objective Function Values')
plt.xlabel('Objective 1 (Mean Distance)')
plt.ylabel('Objective 2 (Spread)')
plt.grid(True)

# Plot optimization history
plt.subplot(1, 3, 3)
best_obj1 = [np.max(vals[0]) for vals in optimizer.values_history]
best_obj2 = [np.max(vals[1]) for vals in optimizer.values_history]
plt.plot(best_obj1, label='Best Objective 1')
plt.plot(best_obj2, label='Best Objective 2')
plt.title('Optimization History')
plt.xlabel('Generation')
plt.ylabel('Best Objective Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Advanced Examples

### 3. Wind-Turbine Layout Optimization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# Define the wind-farm area (irregular polygon)
region = region_from_points([
    [0, 0],
    [5, 0],
    [8, 2],
    [7, 5],
    [4, 6],
    [1, 4],
    [-1, 2]
])

# Define objective functions
def objective_power_production(coords):
    """Maximize total power production (simplified model)"""
    center = np.array([3.5, 3])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    power = np.sum(1 / (1 + distances))
    return power

def objective_cost(coords):
    """Minimize total cost (simplified model)"""
    total_distance = np.sum(np.sqrt(np.sum(coords**2, axis=1)))
    return -total_distance  # Negative sign because we maximize

# Define constraints
def constraint_turbine_spacing(coords):
    """Minimum spacing between turbines"""
    dist_list = distance.pdist(coords)
    min_spacing = 2.0
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

def constraint_boundary_distance(coords):
    """Minimum distance from the boundary"""
    boundary_distance = 0.5
    violations = 0
    for point in coords:
        x, y = point
        if x < boundary_distance or y < boundary_distance:
            violations += boundary_distance - min(x, y)
        if x > 8 - boundary_distance or y > 6 - boundary_distance:
            violations += max(0, x - (8 - boundary_distance)) + max(0, y - (6 - boundary_distance))
    return violations

# Create the problem
problem = Problem(
    objectives=[objective_power_production, objective_cost],
    n_points=12,  # 12 turbines
    region=region,
    constraints=[constraint_turbine_spacing, constraint_boundary_distance]
)

# Create the optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=50,
    prob_crs=0.8,
    prob_mut=0.02
)

# Run the optimization
result = optimizer.run(1000)

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot the wind-farm area and Pareto-optimal layouts
plt.subplot(2, 2, 1)
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.2, fc='lightblue', ec='blue', label='Wind Farm Area')

from coords_nsga2.utils import fast_non_dominated_sort
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_solutions = result[fronts[0]]

for i, solution in enumerate(pareto_solutions):
    plt.scatter(solution[:, 0], solution[:, 1],
                c=f'C{i}', marker='o', s=100, alpha=0.7,
                label=f'Solution {i+1}')

plt.title('Wind Turbine Layout – Pareto Optimal Solutions')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

# Plot objective-function space
plt.subplot(2, 2, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1], alpha=0.6, label='All Solutions')
plt.scatter(optimizer.values_P[0][fronts[0]], optimizer.values_P[1][fronts[0]],
            c='red', s=100, label='Pareto Front')
plt.title('Objective Function Space')
plt.xlabel('Power Production')
plt.ylabel('Cost (negative)')
plt.legend()
plt.grid(True)

# Plot optimization history
plt.subplot(2, 2, 3)
best_power = [np.max(vals[0]) for vals in optimizer.values_history]
best_cost = [np.max(vals[1]) for vals in optimizer.values_history]
plt.plot(best_power, label='Best Power Production')
plt.plot(best_cost, label='Best Cost')
plt.title('Optimization History')
plt.xlabel('Generation')
plt.ylabel('Best Objective Value')
plt.legend()
plt.grid(True)

# Plot convergence analysis
plt.subplot(2, 2, 4)
avg_power = [np.mean(vals[0]) for vals in optimizer.values_history]
avg_cost = [np.mean(vals[1]) for vals in optimizer.values_history]
plt.plot(avg_power, label='Average Power Production')
plt.plot(avg_cost, label='Average Cost')
plt.title('Population Average History')
plt.xlabel('Generation')
plt.ylabel('Average Objective Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Output summary of best solutions
print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
print(f"Best power production: {np.max(optimizer.values_P[0]):.4f}")
print(f"Best cost: {np.max(optimizer.values_P[1]):.4f}")
```

### 4. Sensor-Network Deployment Optimization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_range

# Define the monitoring area
region = region_from_range(0, 20, 0, 15)

# Define objective functions
def objective_coverage(coords):
    """Maximize coverage area"""
    coverage_radius = 3.0
    x_grid, y_grid = np.meshgrid(np.linspace(0, 20, 50), np.linspace(0, 15, 40))
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    covered_points = 0
    for grid_point in grid_points:
        distances = np.sqrt(np.sum((coords - grid_point)**2, axis=1))
        if np.any(distances <= coverage_radius):
            covered_points += 1
    return covered_points / len(grid_points)  # Coverage ratio

def objective_energy_efficiency(coords):
    """Maximize energy efficiency (minimize total transmission distance)"""
    center = np.array([10, 7.5])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    total_distance = np.sum(distances)
    return -total_distance  # Negative sign because we maximize

# Define constraints
def constraint_sensor_spacing(coords):
    """Minimum spacing between sensors"""
    dist_list = distance.pdist(coords)
    min_spacing = 2.0
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

def constraint_battery_life(coords):
    """Battery-life constraint (based on distance to the center node)"""
    center = np.array([10, 7.5])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    max_distance = 12.0
    violations = distances[distances > max_distance] - max_distance
    return np.sum(violations)

# Create the problem
problem = Problem(
    objectives=[objective_coverage, objective_energy_efficiency],
    n_points=8,  # 8 sensors
    region=region,
    constraints=[constraint_sensor_spacing, constraint_battery_life]
)

# Create the optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=40,
    prob_crs=0.6,
    prob_mut=0.03
)

# Run the optimization
result = optimizer.run(600)

# Visualize the results
plt.figure(figsize=(16, 12))

# Plot monitoring area and sensor deployment
plt.subplot(2, 3, 1)
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.1, fc='lightgreen', ec='green', label='Monitoring Area')

from coords_nsga2.utils import fast_non_dominated_sort
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_solutions = result[fronts[0]]

# Plot the first Pareto-optimal solution
best_solution = pareto_solutions[0]
plt.scatter(best_solution[:, 0], best_solution[:, 1],
            c='red', marker='s', s=200, label='Sensors')

coverage_radius = 3.0
for sensor in best_solution:
    circle = plt.Circle(sensor, coverage_radius, alpha=0.2, fc='blue')
    plt.gca().add_patch(circle)

plt.scatter(10, 7.5, c='black', marker='*', s=300, label='Center Node')
plt.title('Sensor Network Deployment')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

# Plot objective-function space
plt.subplot(2, 3, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1], alpha=0.6, label='All Solutions')
plt.scatter(optimizer.values_P[0][fronts[0]], optimizer.values_P[1][fronts[0]],
            c='red', s=100, label='Pareto Front')
plt.title('Objective Function Space')
plt.xlabel('Coverage Rate')
plt.ylabel('Energy Efficiency')
plt.legend()
plt.grid(True)

# Plot optimization history
plt.subplot(2, 3, 3)
best_coverage = [np.max(vals[0]) for vals in optimizer.values_history]
best_energy = [np.max(vals[1]) for vals in optimizer.values_history]
plt.plot(best_coverage, label='Best Coverage')
plt.plot(best_energy, label='Best Energy Efficiency')
plt.title('Optimization History')
plt.xlabel('Generation')
plt.ylabel('Best Objective Value')
plt.legend()
plt.grid(True)

# Plot population diversity
plt.subplot(2, 3, 4)
diversity_coverage = [np.std(vals[0]) for vals in optimizer.values_history]
diversity_energy = [np.std(vals[1]) for vals in optimizer.values_history]
plt.plot(diversity_coverage, label='Coverage Diversity')
plt.plot(diversity_energy, label='Energy Diversity')
plt.title('Population Diversity')
plt.xlabel('Generation')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)

# Plot population averages
plt.subplot(2, 3, 5)
avg_coverage = [np.mean(vals[0]) for vals in optimizer.values_history]
avg_energy = [np.mean(vals[1]) for vals in optimizer.values_history]
plt.plot(avg_coverage, label='Average Coverage')
plt.plot(avg_energy, label='Average Energy Efficiency')
plt.title('Population Average')
plt.xlabel('Generation')
plt.ylabel('Average Objective Value')
plt.legend()
plt.grid(True)

# Plot Pareto front
plt.subplot(2, 3, 6)
pareto_coverage = optimizer.values_P[0][fronts[0]]
pareto_energy = optimizer.values_P[1][fronts[0]]
plt.scatter(pareto_coverage, pareto_energy, c='red', s=100)
plt.title('Pareto Front')
plt.xlabel('Coverage Rate')
plt.ylabel('Energy Efficiency')
plt.grid(True)

plt.tight_layout()
plt.show()

# Output summary
print("Sensor-network deployment optimization completed")
print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
print(f"Best coverage: {np.max(optimizer.values_P[0]):.4f}")
print(f"Best energy efficiency: {np.max(optimizer.values_P[1]):.4f}")
```

## Custom Operator Example

### 5. Custom Crossover and Mutation Operators

```python
import numpy as np
import matplotlib.pyplot as plt
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_range

# Define a custom crossover operator
def custom_crossover(population, prob_crs):
    """Distance-based crossover"""
    n_points = population.shape[1]
    for i in range(0, len(population), 2):
        if np.random.rand() < prob_crs:
            parent1 = population[i]
            parent2 = population[i+1]
            distances = np.sqrt(np.sum((parent1 - parent2)**2, axis=1))
            cross_points = distances > np.median(distances)
            population[i, cross_points] = parent2[cross_points]
            population[i+1, cross_points] = parent1[cross_points]
    return population

# Define a custom mutation operator
def custom_mutation(population, prob_mut, region):
    """Gaussian mutation with region check"""
    from coords_nsga2.spatial import create_points_in_polygon
    mutation_mask = np.random.random(population.shape[:-1]) < prob_mut
    for i in range(len(population)):
        for j in range(population.shape[1]):
            if mutation_mask[i, j]:
                current_point = population[i, j]
                new_point = current_point + np.random.normal(0, 0.5, 2)
                if region.contains(plt.matplotlib.patches.Circle(new_point, 0)):
                    population[i, j] = new_point
                else:
                    population[i, j] = create_points_in_polygon(region, 1)[0]
    return population

# Define region and objective functions
region = region_from_range(0, 10, 0, 10)

def objective_1(coords):
    return np.sum(coords[:, 0])

def objective_2(coords):
    return np.sum(coords[:, 1])

# Create the problem
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=5,
    region=region
)

# Create the optimizer and replace operators
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

optimizer.crossover = custom_crossover
optimizer.mutation = custom_mutation

# Run the optimization
result = optimizer.run(300)

# Visualize the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.2, fc='gray', ec='black')
for i in range(len(result)):
    plt.scatter(result[i, :, 0], result[i, :, 1], alpha=0.6)
plt.title('Custom Operators – Final Population')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1])
plt.title('Custom Operators – Objective Values')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.grid(True)

plt.tight_layout()
plt.show()
```

These examples showcase various ways to use the Coords-NSGA2 library, ranging from basic multi-objective optimization to complex real-world applications. Feel free to modify them to suit your needs.