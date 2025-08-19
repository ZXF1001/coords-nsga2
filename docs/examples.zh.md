# 示例代码

> **⚠️ 重要提示**: 本文档是基于源码分析由AI生成的。虽然我们努力确保准确性，但仍可能存在不一致或问题。我们正在积极改进和验证所有内容。如遇到任何问题，请及时报告。

## 基础示例

### 1. 简单矩形区域优化

```python
import numpy as np
import matplotlib.pyplot as plt
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_range

# 定义矩形区域
region = region_from_range(0, 10, 0, 5)

# 定义目标函数
def objective_1(coords):
    """最大化x坐标和"""
    return np.sum(coords[:, 0])

def objective_2(coords):
    """最大化y坐标和"""
    return np.sum(coords[:, 1])

# 创建问题（支持多个目标函数，以下为2目标示例）
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=8,
    region=region
)

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 运行优化
result = optimizer.run(500)

# 可视化结果
plt.figure(figsize=(12, 5))

# 绘制最终种群
plt.subplot(1, 2, 1)
for i in range(len(result)):
    plt.scatter(result[i, :, 0], result[i, :, 1], alpha=0.6)
plt.title('Final Population')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# 绘制目标函数值
plt.subplot(1, 2, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1])
plt.title('Objective Function Values')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. 带约束的多边形区域优化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 定义多边形区域
region = region_from_points([
    [0, 0],
    [2, 0],
    [3, 1],
    [2, 2],
    [0, 2],
    [-1, 1]
])

# 定义目标函数
def objective_1(coords):
    """最大化到原点的距离"""
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    return np.mean(distances)

def objective_2(coords):
    """最大化点之间的分散度"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# 定义约束条件
def constraint_min_spacing(coords):
    """最小间距约束"""
    dist_list = distance.pdist(coords)
    min_spacing = 0.5
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

def constraint_max_spacing(coords):
    """最大间距约束"""
    dist_list = distance.pdist(coords)
    max_spacing = 3.0
    violations = dist_list[dist_list > max_spacing] - max_spacing
    return np.sum(violations)

# 创建问题
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=6,
    region=region,
    constraints=[constraint_min_spacing, constraint_max_spacing]
)

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=30,
    prob_crs=0.7,
    prob_mut=0.05
)

# 运行优化
result = optimizer.run(800)

# 可视化结果
plt.figure(figsize=(15, 5))

# 绘制区域和最终种群
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

# 绘制目标函数值
plt.subplot(1, 3, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1])
plt.title('Objective Function Values')
plt.xlabel('Objective 1 (Mean Distance)')
plt.ylabel('Objective 2 (Spread)')
plt.grid(True)

# 绘制优化历史
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

### 3. 可变点数优化

此示例演示了可变点数的功能，其中点的数量可以在优化过程中在指定范围内变化。

```python
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

# 定义多个目标函数
def objective_1(coords):
    """最大化点数和右上角位置"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """最小化布局分散度"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

def objective_3(coords):
    """最小化到中心的距离"""
    center = np.array([1.0, 1.0])  # 区域中心
    distances = np.linalg.norm(coords - center, axis=1)
    return -np.mean(distances)  # 负号表示最大化

def objective_4(coords):
    """最大化点之间的最小距离"""
    if len(coords) < 2:
        return 0
    dist_matrix = distance.pdist(coords)
    return np.min(dist_matrix)

def constraint_spacing(coords):
    """点之间的最小间距约束"""
    min_spacing = 0.1  # 间距限制
    if len(coords) < 2:
        return 0
    dist_list = distance.pdist(coords)
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

# 创建具有可变点数的问题
problem = Problem(
    objectives=[objective_1, objective_2, objective_3, objective_4],
    n_points=[10, 30],  # 可变点数：10到30个点之间
    region=region,
    constraints=[constraint_spacing]
)

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=40,
    prob_crs=0.5,
    prob_mut=0.1
)

# 运行优化
result = optimizer.run(1000, verbose=True)  # 设置为True显示进度条

# 可视化结果
optimizer.plot.optimal_coords([0, 1, 2, 3])  # 显示所有目标的最优布局
optimizer.plot.pareto_front([0, 1, 2])       # 显示前3个目标的帕累托前沿
```

此示例展示了如何：
- 通过指定 `n_points=[10, 30]` 而不是固定数字来使用可变点数
- 处理多个目标（本例中为4个）
- 应用适用于可变点数的约束
- 可视化可变点数问题的结果

可变点数功能在以下情况下特别有用：
- 最佳点数未知
- 您希望探索解决方案复杂性和性能之间的权衡
- 不同的目标可能偏好不同数量的点

## 高级示例

### 4. 风力发电机布局优化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 定义风场区域（不规则多边形）
region = region_from_points([
    [0, 0],
    [5, 0],
    [8, 2],
    [7, 5],
    [4, 6],
    [1, 4],
    [-1, 2]
])

# 定义目标函数
def objective_power_production(coords):
    """最大化总发电量（简化模型）"""
    # 假设发电量与到中心点的距离成反比
    center = np.array([3.5, 3])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    power = np.sum(1 / (1 + distances))
    return power

def objective_cost(coords):
    """最小化总成本（简化模型）"""
    # 假设成本与总距离成正比
    total_distance = np.sum(np.sqrt(np.sum(coords**2, axis=1)))
    return -total_distance  # 负号因为我们要最大化

# 定义约束条件
def constraint_turbine_spacing(coords):
    """风力发电机最小间距约束"""
    dist_list = distance.pdist(coords)
    min_spacing = 2.0  # 最小间距2个单位
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

def constraint_boundary_distance(coords):
    """边界距离约束"""
    # 确保所有点距离边界至少0.5个单位
    boundary_distance = 0.5
    violations = 0
    
    for point in coords:
        # 计算点到边界的距离（简化计算）
        x, y = point
        if x < boundary_distance or y < boundary_distance:
            violations += boundary_distance - min(x, y)
        if x > 8 - boundary_distance or y > 6 - boundary_distance:
            violations += max(0, x - (8 - boundary_distance)) + max(0, y - (6 - boundary_distance))
    
    return violations

# 创建问题
problem = Problem(
    objectives=[objective_power_production, objective_cost],
    n_points=12,  # 12台风力发电机
    region=region,
    constraints=[constraint_turbine_spacing, constraint_boundary_distance]
)

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=50,
    prob_crs=0.8,
    prob_mut=0.02
)

# 运行优化
result = optimizer.run(1000)

# 可视化结果
plt.figure(figsize=(15, 10))

# 绘制风场区域和最优解
plt.subplot(2, 2, 1)
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.2, fc='lightblue', ec='blue', label='Wind Farm Area')

# 找到帕累托最优解
from coords_nsga2.utils import fast_non_dominated_sort
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_solutions = result[fronts[0]]

# 绘制帕累托最优解
for i, solution in enumerate(pareto_solutions):
    plt.scatter(solution[:, 0], solution[:, 1], 
               c=f'C{i}', marker='o', s=100, alpha=0.7, 
               label=f'Solution {i+1}')

plt.title('Wind Turbine Layout - Pareto Optimal Solutions')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

# 绘制目标函数值
plt.subplot(2, 2, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1], alpha=0.6, label='All Solutions')
plt.scatter(optimizer.values_P[0][fronts[0]], optimizer.values_P[1][fronts[0]], 
           c='red', s=100, label='Pareto Front')
plt.title('Objective Function Space')
plt.xlabel('Power Production')
plt.ylabel('Cost (negative)')
plt.legend()
plt.grid(True)

# 绘制优化历史
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

# 绘制收敛性分析
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

# 输出最优解信息
print(f"找到 {len(pareto_solutions)} 个帕累托最优解")
print(f"最佳发电量: {np.max(optimizer.values_P[0]):.4f}")
print(f"最佳成本: {np.max(optimizer.values_P[1]):.4f}")
```

### 5. 传感器网络部署优化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_range

# 定义监控区域
region = region_from_range(0, 20, 0, 15)

# 定义目标函数
def objective_coverage(coords):
    """最大化覆盖面积"""
    # 简化的覆盖模型：每个传感器覆盖半径为3的圆形区域
    coverage_radius = 3.0
    
    # 生成网格点来评估覆盖
    x_grid, y_grid = np.meshgrid(np.linspace(0, 20, 50), np.linspace(0, 15, 40))
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    
    covered_points = 0
    for grid_point in grid_points:
        distances = np.sqrt(np.sum((coords - grid_point)**2, axis=1))
        if np.any(distances <= coverage_radius):
            covered_points += 1
    
    return covered_points / len(grid_points)  # 覆盖率

def objective_energy_efficiency(coords):
    """最大化能量效率（最小化总传输距离）"""
    # 假设有一个中心节点在(10, 7.5)
    center = np.array([10, 7.5])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    total_distance = np.sum(distances)
    return -total_distance  # 负号因为我们要最大化

# 定义约束条件
def constraint_sensor_spacing(coords):
    """传感器最小间距约束"""
    dist_list = distance.pdist(coords)
    min_spacing = 2.0
    violations = min_spacing - dist_list[dist_list < min_spacing]
    return np.sum(violations)

def constraint_battery_life(coords):
    """电池寿命约束（基于距离中心节点的距离）"""
    center = np.array([10, 7.5])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    max_distance = 12.0  # 最大传输距离
    violations = distances[distances > max_distance] - max_distance
    return np.sum(violations)

# 创建问题
problem = Problem(
    objectives=[objective_coverage, objective_energy_efficiency],
    n_points=8,  # 8个传感器
    region=region,
    constraints=[constraint_sensor_spacing, constraint_battery_life]
)

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=40,
    prob_crs=0.6,
    prob_mut=0.03
)

# 运行优化
result = optimizer.run(600)

# 可视化结果
plt.figure(figsize=(16, 12))

# 绘制监控区域和传感器部署
plt.subplot(2, 3, 1)
x, y = region.exterior.xy
plt.fill(x, y, alpha=0.1, fc='lightgreen', ec='green', label='Monitoring Area')

# 找到帕累托最优解
from coords_nsga2.utils import fast_non_dominated_sort
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_solutions = result[fronts[0]]

# 绘制最优解
best_solution = pareto_solutions[0]  # 选择第一个帕累托解
plt.scatter(best_solution[:, 0], best_solution[:, 1], 
           c='red', marker='s', s=200, label='Sensors')

# 绘制覆盖范围
coverage_radius = 3.0
for sensor in best_solution:
    circle = plt.Circle(sensor, coverage_radius, alpha=0.2, fc='blue')
    plt.gca().add_patch(circle)

# 绘制中心节点
plt.scatter(10, 7.5, c='black', marker='*', s=300, label='Center Node')

plt.title('Sensor Network Deployment')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

# 绘制目标函数值
plt.subplot(2, 3, 2)
plt.scatter(optimizer.values_P[0], optimizer.values_P[1], alpha=0.6, label='All Solutions')
plt.scatter(optimizer.values_P[0][fronts[0]], optimizer.values_P[1][fronts[0]], 
           c='red', s=100, label='Pareto Front')
plt.title('Objective Function Space')
plt.xlabel('Coverage Rate')
plt.ylabel('Energy Efficiency')
plt.legend()
plt.grid(True)

# 绘制优化历史
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

# 绘制种群多样性
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

# 绘制收敛性分析
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

# 绘制帕累托前沿
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

# 输出结果统计
print(f"传感器网络部署优化完成")
print(f"找到 {len(pareto_solutions)} 个帕累托最优解")
print(f"最佳覆盖率: {np.max(optimizer.values_P[0]):.4f}")
print(f"最佳能量效率: {np.max(optimizer.values_P[1]):.4f}")
```
