import matplotlib.pyplot as plt
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

# 获取最终的目标函数值
final_values = optimizer.values_P
v1_max_index = final_values[0].argmax()
v1_min_index = final_values[0].argmin()

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(result[v1_max_index, :, 0], result[v1_max_index, :, 1], 
           color='red', label='Best Solution')
plt.scatter(result[v1_min_index, :, 0], result[v1_min_index, :, 1], 
           color='blue', label='Worst Solution')

# 绘制多边形边界
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.2, fc='gray', ec='black')
plt.show()
