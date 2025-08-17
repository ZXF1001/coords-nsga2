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

# 定义目标函数1：更靠近右上方
def objective_1(coords):
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

# 定义目标函数2：布局更分散
def objective_2(coords):
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

spacing = 0.05  # 间距限制

def constraint_1(coords):
    dist_list = distance.pdist(coords)
    penalty_list = spacing-dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)
result = optimizer.run(1000, verbose=True) # 设置为True显示进度条，False则不显示

# 可视化各目标函数的最优布局
optimizer.plot.example(arg1=1, arg2=2)