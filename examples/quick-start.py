import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 创建边界
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
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


def constraint_spacing(coords):
    min_spacing = 0.1  # 间距限制
    """Minimum spacing constraint between points"""
    if len(coords) < 2:
        return 0
    dist_list = distance.pdist(coords)
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

problem = Problem(
    objectives=[objective_1, objective_2, objective_3, objective_4],
    n_points=10,
    region=region,
    constraints=[constraint_spacing]
)

optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)
result = optimizer.run(500, verbose=True) # 设置为True显示进度条，False则不显示

# 保存当前状态
optimizer.save("examples/data/test_optimizer.pkl")

# 可视化各目标函数的最优布局
optimizer.plot.optimal_coords(obj_indices=[0, 1])