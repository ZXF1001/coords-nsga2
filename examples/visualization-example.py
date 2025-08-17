"""
Comprehensive visualization example for coords-nsga2
This example demonstrates all available visualization functions
"""

import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# Create optimization region
region = region_from_points([
    [0, 0],
    [2, 0],
    [2, 1.5],
    [1, 2],
    [0, 1.5],
])

# Define multiple objective functions


def objective_1(coords):
    """Maximize sum of x and y coordinates (prefer upper-right)"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])


def objective_2(coords):
    """Maximize layout dispersion"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])


def objective_3(coords):
    """Minimize distance to center"""
    center = np.array([1.0, 1.0])  # Region center
    distances = np.linalg.norm(coords - center, axis=1)
    return -np.mean(distances)  # Negative for maximization


def objective_4(coords):
    """Maximize minimum distance between points"""
    if len(coords) < 2:
        return 0
    dist_matrix = distance.pdist(coords)
    return np.min(dist_matrix)


# Define constraints
min_spacing = 0.1


def constraint_spacing(coords):
    """Minimum spacing constraint between points"""
    if len(coords) < 2:
        return 0
    dist_list = distance.pdist(coords)
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)


# Create multi-objective optimization problem
problem = Problem(
    objectives=[objective_1, objective_2, objective_3, objective_4],
    n_points=10,
    region=region,
    constraints=[constraint_spacing]
)

# Create optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=50,
    prob_crs=0.7,
    prob_mut=0.2,
    random_seed=42
)

# Run optimization
result = optimizer.run(200, verbose=True)


# 1. Pareto Front Visualizations
print("\n1. Plotting Pareto Front (2D and 3D)...")
optimizer.plot.pareto_front(obj_indices=[0, 1])  # 2D
optimizer.plot.pareto_front(obj_indices=[0, 1, 2])  # 3D

# # 2. Parallel Coordinates Plot
# print("\n2. Plotting Parallel Coordinates...")
# optimizer.plot.parallel_coordinates()

# # 3. Hypervolume Trend
# print("\n3. Plotting Hypervolume Trend...")
# optimizer.plot.hypervolume_trend()

# # 4. Objective Optimal Layouts
# print("\n4. Plotting Optimal Layouts for Each Objective...")
# optimizer.plot.objective_optimal_layouts()

# # 5. Solution Comparison
# print("\n5. Plotting Solution Comparison...")
# # Select some diverse solutions for comparison
# selected_solutions = [0, 5, 10, 15, 20, 25]  # Indices of solutions to compare
# optimizer.plot.solution_comparison(solution_indices=selected_solutions)

# # 6. Convergence Analysis
# print("\n6. Plotting Convergence Analysis...")
# optimizer.plot.convergence_analysis()

# # 7. Constraint Violations
# print("\n7. Plotting Constraint Violations...")
# optimizer.plot.constraint_violations()

# # 8. Objective Distributions
# print("\n8. Plotting Objective Distributions...")
# optimizer.plot.objective_distributions()

# # 9. Objective Correlations
# print("\n9. Plotting Objective Correlations...")
# optimizer.plot.objective_correlations()
