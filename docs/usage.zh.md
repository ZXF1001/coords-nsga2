# 使用指南 / Usage Guide

> **⚠️ 重要提示**: 本文档是基于源码分析由AI生成的。虽然我们努力确保准确性，但仍可能存在不一致或问题。我们正在积极改进和验证所有内容。如遇到任何问题，请及时报告。

## 中文使用指南

### 基本概念

Coords-NSGA2 库的核心概念包括：

1. **Problem（问题）**：定义优化问题的目标函数、约束条件和搜索区域
2. **CoordsNSGA2（优化器）**：执行NSGA-II算法的优化器
3. **Region（区域）**：定义坐标点的有效搜索空间
4. **Constraints（约束）**：限制解的可行性的条件

### 快速开始示例

以下是一个完整的使用示例，演示如何优化10个坐标点的布局：

```python
import numpy as np
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 1. 定义优化区域（多边形）
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# 2. 定义目标函数
def objective_1(coords):
    """第一个目标：最大化坐标和"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """第二个目标：最大化点的分布"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# 3. 定义约束条件
spacing = 0.05  # 最小间距
def constraint_1(coords):
    """约束：点之间的最小间距"""
    dist_list = distance.pdist(coords)
    penalty_list = spacing - dist_list[dist_list < spacing]
    return np.sum(penalty_list)

# 4. 创建问题实例（支持任意多个目标）
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

# 5. 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 6. 运行优化
result = optimizer.run(1000)

# 7. 查看结果
print(f"优化完成！结果形状: {result.shape}")
print(f"种群大小: {len(result)}")
print(f"每个解的坐标点数: {result.shape[1]}")
```

### 区域定义

#### 从点列表创建多边形区域

```python
from coords_nsga2.spatial import region_from_points

# 定义多边形的顶点
points = [
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
]
region = region_from_points(points)
```

#### 从坐标范围创建矩形区域

```python
from coords_nsga2.spatial import region_from_range

# 定义矩形的边界
region = region_from_range(x_min=0, x_max=10, y_min=0, y_max=5)
```

### 目标函数定义

目标函数应该接受一个形状为 `(n_points, 2)` 的numpy数组作为输入，返回一个标量值：

```python
def my_objective(coords):
    """
    参数:
        coords: numpy数组，形状为(n_points, 2)
                每行是一个坐标点 [x, y]
    
    返回:
        float: 目标函数值
    """
    # 示例：计算所有点到原点的平均距离
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    return np.mean(distances)
```

### 约束条件定义

约束函数应该返回违反约束的惩罚值。返回0表示没有违反约束：

```python
def my_constraint(coords):
    """
    参数:
        coords: numpy数组，形状为(n_points, 2)
    
    返回:
        float: 约束违反的惩罚值（0表示无违反）
    """
    # 示例：确保所有点都在单位圆内
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    violations = distances[distances > 1] - 1
    return np.sum(violations)
```

### 优化器参数

#### CoordsNSGA2 参数说明

- `problem`: Problem实例
- `pop_size`: 种群大小（必须为偶数）
- `prob_crs`: 交叉概率（0-1之间）
- `prob_mut`: 变异概率（0-1之间）
- `random_seed`: 随机种子（用于可重现性）

#### 参数调优建议

- **种群大小**: 通常设置为20-100，问题复杂时使用更大的种群
- **交叉概率**: 通常设置为0.5-0.9
- **变异概率**: 通常设置为0.01-0.1
- **代数**: 根据问题复杂度设置，通常100-1000代

### 结果分析

优化完成后，您可以访问以下属性：

```python
# 最终种群
final_population = optimizer.P

# 目标函数值（形状: n_objectives × pop_size）
values = optimizer.values_P
values1 = values[0]
values2 = values[1]

# 优化历史
population_history = optimizer.P_history
values_history = optimizer.values_history  # 列表，每代一个 (n_objectives, pop_size) 数组

# 找到帕累托前沿（基于最后一代目标值）
from coords_nsga2.utils import fast_non_dominated_sort
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_front = optimizer.P[fronts[0]]
```

### 保存和加载

```python
# 保存优化状态
optimizer.save("optimization_result.npz")

# 加载优化状态
optimizer.load("optimization_result.npz")
```