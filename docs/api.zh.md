# API 参考文档

> **⚠️ 重要提示**: 本文档是基于源码分析由AI生成的。虽然我们努力确保准确性，但仍可能存在不一致或问题。我们正在积极改进和验证所有内容。如遇到任何问题，请及时报告。

## 核心类

### Problem

多目标优化问题定义类。

**构造函数：**

```python
Problem(objectives, n_points, region, constraints=[], penalty_weight=1e6)
```

**参数：**

- `objectives` (list[callable]): 目标函数列表，每个函数接受 `coords (n_points, 2)` 并返回标量
- `n_points` (int): 要优化的坐标点数量
- `region` (shapely.geometry.Polygon): 定义有效区域的Shapely多边形
- `constraints` (list, optional): 约束函数列表，默认为空列表
- `penalty_weight` (float, optional): 约束违反的权重，默认为1e6

**方法：**

#### sample_population(pop_size)
生成初始种群。

**参数：**

- `pop_size` (int): 种群大小

**返回：**

- `numpy.ndarray`: 形状为(pop_size, n_points, 2)的种群数组

#### evaluate(population)
评估种群中所有个体的目标函数值。

**参数：**

- `population` (numpy.ndarray): 形状为(pop_size, n_points, 2)的种群

**返回：**

- `numpy.ndarray`: 形状为 `(n_objectives, pop_size)` 的目标函数值数组

**示例：**

```python
from coords_nsga2 import Problem
from coords_nsga2.spatial import region_from_points

# 定义目标函数
def obj1(coords):
    return np.sum(coords[:, 0])

def obj2(coords):
    return np.sum(coords[:, 1])

# 创建问题
region = region_from_points([[0,0], [1,0], [1,1], [0,1]])
problem = Problem(objectives=[obj1, obj2], n_points=5, region=region)

# 生成初始种群
population = problem.sample_population(10)
print(f"种群形状: {population.shape}")  # (10, 5, 2)

# 评估种群
values = problem.evaluate(population)  # shape: (n_objectives, pop_size)
print(f"目标函数1值: {values[0]}")
print(f"目标函数2值: {values[1]}")
```

### CoordsNSGA2

NSGA-II坐标优化器类。

**构造函数：**

```python
CoordsNSGA2(problem, pop_size, prob_crs, prob_mut, random_seed=42)
```

**参数：**

- `problem` (Problem): 问题实例
- `pop_size` (int): 种群大小（必须为偶数）
- `prob_crs` (float): 交叉概率（0-1之间）
- `prob_mut` (float): 变异概率（0-1之间）
- `random_seed` (int, optional): 随机种子，默认为42

**属性：**

- `P`: 当前种群，形状 `(pop_size, n_points, 2)`
- `values_P`: 当前种群的目标函数值，形状 `(n_objectives, pop_size)`
- `P_history`: 种群历史记录
- `values_history`: 历史目标函数值列表，每代一个 `(n_objectives, pop_size)` 数组

**方法：**

#### run(generations, verbose=True)
运行优化算法。

**参数：**

- `generations` (int): 优化代数
- `verbose` (bool, optional): 是否显示进度条，默认为True

**返回：**

- `numpy.ndarray`: 最终种群

#### save(path)
保存优化状态到文件。

**参数：**

- `path` (str): 保存文件路径

#### load(path)
从文件加载优化状态。

**参数：**

- `path` (str): 加载文件路径

**示例：**

```python
from coords_nsga2 import CoordsNSGA2, Problem

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 运行优化
result = optimizer.run(1000)

# 保存结果
optimizer.save("optimization_result.npz")

# 加载结果
optimizer.load("optimization_result.npz")
```

## 空间工具

### region_from_points(points)

从坐标点列表创建多边形区域。

**参数：**

- `points` (list): 坐标点列表，格式为[[x1,y1], [x2,y2], ...]

**返回：**

- `shapely.geometry.Polygon`: Shapely多边形对象

**示例：**

```python
from coords_nsga2.spatial import region_from_points

# 创建三角形区域
points = [[0, 0], [1, 0], [0.5, 1]]
region = region_from_points(points)
print(f"区域面积: {region.area}")
```

### region_from_range(x_min, x_max, y_min, y_max)

从坐标边界创建矩形区域。

**参数：**

- `x_min` (float): x坐标最小值
- `x_max` (float): x坐标最大值
- `y_min` (float): y坐标最小值
- `y_max` (float): y坐标最大值

**返回：**

- `shapely.geometry.Polygon`: Shapely矩形对象

**示例：**

```python
from coords_nsga2.spatial import region_from_range

# 创建矩形区域
region = region_from_range(0, 10, 0, 5)
print(f"区域边界: {region.bounds}")
```

### create_points_in_polygon(polygon, n)

在多边形内生成n个随机点。

**参数：**

- `polygon` (shapely.geometry.Polygon): 目标多边形
- `n` (int): 要生成的点数量

**返回：**

- `numpy.ndarray`: 形状为(n, 2)的坐标点数组

**示例：**

```python
from coords_nsga2.spatial import create_points_in_polygon, region_from_points

# 创建区域
region = region_from_points([[0,0], [1,0], [1,1], [0,1]])

# 生成随机点
points = create_points_in_polygon(region, 10)
print(f"生成的点: {points}")
```

## 遗传算子

### coords_crossover(population, prob_crs)

坐标特定的交叉算子，在父代之间交换点子集。

**参数：**

- `population` (numpy.ndarray): 形状为(pop_size, n_points, 2)的种群
- `prob_crs` (float): 交叉概率

**返回：**

- `numpy.ndarray`: 交叉后的种群

**算法说明：**

- 对每对父代个体，以概率prob_crs进行交叉
- 随机选择1到n_points-1个点进行交换
- 保持种群大小不变

**示例：**

```python
from coords_nsga2.operators.crossover import coords_crossover

# 执行交叉操作
new_population = coords_crossover(population, prob_crs=0.5)
```

### coords_mutation(population, prob_mut, region)

坐标特定的变异算子，在区域内随机重新定位点。

**参数：**

- `population` (numpy.ndarray): 形状为(pop_size, n_points, 2)的种群
- `prob_mut` (float): 变异概率
- `region` (shapely.geometry.Polygon): 有效区域

**返回：**

- `numpy.ndarray`: 变异后的种群

**算法说明：**

- 对每个坐标点，以概率prob_mut进行变异
- 变异时在区域内重新生成随机位置
- 确保变异后的点仍在有效区域内

**示例：**

```python
from coords_nsga2.operators.mutation import coords_mutation

# 执行变异操作
new_population = coords_mutation(population, prob_mut=0.1, region=region)
```

### coords_selection(population, values_P, tourn_size=3)

基于非支配排序和拥挤距离的锦标赛选择。

**参数：**

- `population` (numpy.ndarray): 形状为 `(pop_size, n_points, 2)` 的种群
- `values_P` (numpy.ndarray): 形状为 `(n_objectives, pop_size)` 的目标函数值数组
- `tourn_size` (int, optional): 锦标赛大小，默认为3

**返回：**

- `numpy.ndarray`: 选择后的种群

**算法说明：**

- 使用快速非支配排序确定前沿等级
- 计算每个前沿内的拥挤距离
- 使用锦标赛选择，优先选择前沿等级低的个体
- 前沿等级相同时，选择拥挤距离大的个体

**示例：**

```python
from coords_nsga2.operators.selection import coords_selection

# 执行选择操作
selected_population = coords_selection(population, values_P, tourn_size=3)
```

## 工具函数

### fast_non_dominated_sort(objectives)

快速非支配排序算法。

**参数：**

- `objectives` (numpy.ndarray): 形状为 `(n_objectives, pop_size)` 的目标函数值数组

**返回：**

- `list`: 前沿列表，每个前沿包含该前沿中个体的索引

**算法说明：**

- 计算每个个体被支配的次数
- 记录每个个体支配的其他个体
- 按前沿等级对个体进行排序
- 返回按前沿分组的个体索引

**示例：**

```python
from coords_nsga2.utils import fast_non_dominated_sort

# 执行非支配排序
fronts = fast_non_dominated_sort(values)
print(f"前沿数量: {len(fronts)}")
for i, front in enumerate(fronts):
    print(f"前沿{i}: {front}")
```

### crowding_distance(objectives)

计算拥挤距离。

**参数：**

- `objectives` (numpy.ndarray): 形状为 `(n_objectives, pop_size)` 的目标函数值数组

**返回：**

- `numpy.ndarray`: 拥挤距离数组

**算法说明：**

- 对每个目标函数值进行排序
- 边界点的拥挤距离设为无穷大
- 中间点的拥挤距离为相邻点目标函数值差的归一化和
- 返回按原始顺序排列的拥挤距离

**示例：**

```python
from coords_nsga2.utils import crowding_distance

# 计算拥挤距离（对同一前沿子集计算时传入子集）
distances = crowding_distance(values)
print(f"拥挤距离: {distances}")
```

## 完整示例

```python
import numpy as np
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points
from coords_nsga2.utils import fast_non_dominated_sort, crowding_distance

# 1. 定义问题
def objective_1(coords):
    """最大化x坐标和"""
    return np.sum(coords[:, 0])

def objective_2(coords):
    """最大化y坐标和"""
    return np.sum(coords[:, 1])

def constraint(coords):
    """最小间距约束"""
    dist_list = distance.pdist(coords)
    penalty = np.sum(0.1 - dist_list[dist_list < 0.1])
    return penalty

# 2. 创建区域和问题
region = region_from_points([[0,0], [1,0], [1,1], [0,1]])
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=5,
    region=region,
    constraints=[constraint]
)

# 3. 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 4. 运行优化
result = optimizer.run(100)

# 5. 分析结果
print(f"最终种群形状: {result.shape}")
print(f"目标函数1值: {optimizer.values_P[0]}")
print(f"目标函数2值: {optimizer.values_P[1]}")

# 6. 找到帕累托前沿
fronts = fast_non_dominated_sort(optimizer.values_P)
pareto_front = result[fronts[0]]  # 第一个前沿就是帕累托前沿
print(f"帕累托前沿解数量: {len(pareto_front)}")
```
