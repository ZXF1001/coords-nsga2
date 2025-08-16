import time

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

# 定义一个计算密集型的目标函数
def objective_1(coords):
    # 模拟计算密集型操作
    time.sleep(0.01)  # 每个个体评估增加一点延迟，模拟复杂计算
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

# 定义目标函数2：布局更分散
def objective_2(coords):
    # 模拟计算密集型操作
    time.sleep(0.01)  # 每个个体评估增加一点延迟，模拟复杂计算
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

spacing = 0.05  # 间距限制

def constraint_1(coords):
    dist_list = distance.pdist(coords)
    penalty_list = spacing-dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

# 创建问题实例
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

# 比较串行和并行计算的性能差异
def run_optimization(n_jobs, generations=10):
    start_time = time.time()
    
    optimizer = CoordsNSGA2(
        problem=problem,
        pop_size=20,
        prob_crs=0.5,
        prob_mut=0.1,
        n_jobs=n_jobs  # 设置并行作业数
    )
    
    result = optimizer.run(generations, verbose=True)
    
    end_time = time.time()
    return result, end_time - start_time

# 串行计算 (n_jobs=1)
print("运行串行计算...")
result_serial, time_serial = run_optimization(n_jobs=1)
print(f"串行计算耗时: {time_serial:.2f} 秒")

# 并行计算 (n_jobs=-1，使用所有可用CPU核心)
print("\n运行并行计算...")
result_parallel, time_parallel = run_optimization(n_jobs=-1)
print(f"并行计算耗时: {time_parallel:.2f} 秒")
print(f"加速比: {time_serial / time_parallel:.2f}x")

# 绘制结果比较
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制串行结果
ax1.set_title(f"串行计算结果 (耗时: {time_serial:.2f}s)")
ax1.scatter(result_serial[:, :, 0].flatten(), result_serial[:, :, 1].flatten(), color='blue')
x, y = region.exterior.xy
ax1.fill(x, y, alpha=0.2, fc='gray', ec='black')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# 绘制并行结果
ax2.set_title(f"并行计算结果 (耗时: {time_parallel:.2f}s)")
ax2.scatter(result_parallel[:, :, 0].flatten(), result_parallel[:, :, 1].flatten(), color='red')
ax2.fill(x, y, alpha=0.2, fc='gray', ec='black')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.show()