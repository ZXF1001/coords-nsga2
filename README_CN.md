![coords-nsga2](logo.drawio.svg)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-coords--nsga2-blue.svg)](https://pypi.org/project/coords-nsga2/)
[![GitHub Tag](https://img.shields.io/github/v/tag/ZXF1001/coords-nsga2)](https://github.com/ZXF1001/coords-nsga2/tags)

[English](README.md) | [中文](README_CN.md)

> **⚠️ 重要提示**: 本文档和README文件是基于源码分析由AI生成的。虽然我们努力确保准确性，但仍可能存在不一致或问题。我们正在积极改进和验证所有内容。如遇到任何问题，请及时报告。

一个基于Python实现的坐标点布局多目标优化算法库，基于NSGA-II（非支配排序遗传算法II）改进。该库专门为优化坐标点布局而设计，具有专门的约束条件、交叉和变异算子，可直接作用于坐标点。

## 特性

- **坐标优化专用**：专门为优化坐标点布局而设计
- **专业约束条件**：内置支持点间距、边界限制和自定义约束
- **定制遗传算子**：专门作用于坐标点的交叉和变异算子
- **多目标优化**：基于成熟的NSGA-II算法
- **灵活区域定义**：支持多边形和矩形区域
- **轻量级可扩展**：易于自定义算子和约束条件
- **进度跟踪**：内置进度条和优化历史记录
- **保存/加载功能**：保存和恢复优化状态

## 安装

### 从PyPI安装
```bash
pip install coords-nsga2
```

### 从源码安装
```bash
git clone https://github.com/ZXF1001/coords-nsga2.git
cd coords-nsga2
pip install -e .
```

## 快速开始

以下是一个演示如何运行基于坐标的NSGA-II优化的最小示例（支持任意多个目标）：

```python
import numpy as np
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 定义优化区域
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# 定义目标函数
def objective_1(coords):
    """最大化x和y坐标的和"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """最大化点的分布"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# 定义约束条件
spacing = 0.05
def constraint_1(coords):
    """点之间的最小间距"""
    dist_list = distance.pdist(coords)
    penalty_list = spacing - dist_list[dist_list < spacing]
    return np.sum(penalty_list)

# 设置问题（传入目标函数列表）
problem = Problem(
    objectives=[objective_1, objective_2],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

# 初始化优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# 运行优化
result = optimizer.run(1000)

# 访问结果
print(f"最优解形状: {result.shape}")
print(f"优化历史长度: {len(optimizer.P_history)}")
```

## API参考

### 核心类

#### Problem
多目标优化的主要问题定义类。

```python
Problem(objectives, n_points, region, constraints=[], penalty_weight=1e6)
```

**参数：**
- `objectives`：目标函数列表（可调用对象），每个函数接受 `coords` 并返回标量
- `n_points`：要优化的坐标点数量
- `region`：定义有效区域的Shapely多边形
- `constraints`：约束函数列表（可选）
- `penalty_weight`：约束违反的权重（默认：1e6）

#### CoordsNSGA2
实现NSGA-II坐标优化的主要优化器类。

```python
CoordsNSGA2(problem, pop_size, prob_crs, prob_mut, random_seed=42)
```

**参数：**
- `problem`：问题实例
- `pop_size`：种群大小（必须为偶数）
- `prob_crs`：交叉概率
- `prob_mut`：变异概率
- `random_seed`：随机种子，用于可重现性

**方法：**
- `run(generations, verbose=True)`：运行指定代数和进度显示方式的优化
- `save(path)`：保存优化状态到文件
- `load(path)`：从文件加载优化状态

### 空间工具

#### region_from_points(points)
从坐标点列表创建多边形区域。

#### region_from_range(x_min, x_max, y_min, y_max)
从坐标边界创建矩形区域。

#### create_points_in_polygon(polygon, n)
在多边形内生成n个随机点。

### 遗传算子

#### coords_crossover(population, prob_crs)
坐标特定的交叉算子，在父代之间交换点子集。

#### coords_mutation(population, prob_mut, region)
坐标特定的变异算子，在区域内随机重新定位点。

#### coords_selection(population, values_P, tourn_size=3)
基于非支配排序和拥挤距离的锦标赛选择。

## 示例

查看 `examples/` 目录获取更详细的使用示例：

- [快速开始示例](examples/quick-start.py) - 2目标基本用法
- [多目标示例](examples/multi-objective-example.py) - 4目标及可视化

## 文档

完整文档可在 [docs/](docs) 文件夹中找到。

要在本地启动文档服务器：
```bash
mkdocs serve
```

要构建文档：
```bash
mkdocs build
```

## 贡献

欢迎贡献！请随时提交拉取请求。对于重大更改，请先打开一个问题来讨论您想要更改的内容。

1. Fork 该仓库
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开拉取请求

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 引用

如果您在研究中使用了这个库，请引用：

```bibtex
@software{coords_nsga2,
  title={coords-nsga2: 基于坐标的多目标优化Python库},
  author={Zhang, Xiaofeng},
  year={2024},
  url={https://github.com/ZXF1001/coords-nsga2}
}
```
