import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2

# 下面的三个函数还是得手动重新定义一下

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


def constraint_spacing(coords):
    min_spacing = 0.1  # 间距限制
    """Minimum spacing constraint between points"""
    if len(coords) < 2:
        return 0
    dist_list = distance.pdist(coords)
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")
loaded_optimizer.run(200, verbose=True)
loaded_optimizer.plot.pareto_front([0,1])