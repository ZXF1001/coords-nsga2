# coords-nsga2

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-coords--nsga2-blue.svg)](https://pypi.org/project/coords-nsga2/)
[![GitHub Tag](https://img.shields.io/github/v/tag/ZXF1001/coords-nsga2)](https://github.com/ZXF1001/coords-nsga2/tags)

[English](README.md) | [中文](README_CN.md)

> **⚠️ Important Notice**: This documentation and README files are AI-generated based on the source code analysis. While we strive for accuracy, there may be inconsistencies or issues. We are actively working to improve and verify all content. Please report any problems you encounter.

A Python library implementing a coordinate-based NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization. This library is specifically designed for optimizing coordinate point layouts, featuring specialized constraints, crossover, and mutation operators that work directly on coordinate points.

## Features

- **Coordinate-focused optimization**: Designed specifically for optimizing layouts of coordinate points
- **Multi-objective optimization**: Supports 2 or more objective functions using NSGA-II algorithm
- **Specialized constraints**: Built-in support for point spacing, boundary limits, and custom constraints
- **Tailored genetic operators**: Custom crossover and mutation operators that directly act on coordinate points
- **Flexible region definition**: Support for both polygon and rectangular regions
- **Lightweight and extensible**: Easy to customize operators and constraints
- **Progress tracking**: Built-in progress bars and optimization history
- **Save/Load functionality**: Save and restore optimization states

## Installation

### From PyPI
```bash
pip install coords-nsga2
```

### From Source
```bash
git clone https://github.com/ZXF1001/coords-nsga2.git
cd coords-nsga2
pip install -e .
```

## Quick Start

Here's a minimal example demonstrating how to run a coordinate-based NSGA-II optimization with multiple objectives:

```python
import numpy as np
from scipy.spatial import distance
from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# Define the optimization region
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# Define objective functions
def objective_1(coords):
    """Maximize sum of x and y coordinates"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """Maximize spread of points"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# Define constraints
spacing = 0.05
def constraint_1(coords):
    """Minimum spacing between points"""
    dist_list = distance.pdist(coords)
    penalty_list = spacing - dist_list[dist_list < spacing]
    return np.sum(penalty_list)

# Setup the problem
problem = Problem(
    objectives=[objective_1, objective_2],  # List of objective functions
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

# Initialize the optimizer
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=20,
    prob_crs=0.5,
    prob_mut=0.1
)

# Run optimization
result = optimizer.run(1000)

# Access results
print(f"Best solution shape: {result.shape}")
print(f"Number of objectives: {len(optimizer.values_P)}")
print(f"Optimization history length: {len(optimizer.P_history)}")
```

## Multi-Objective Example

Here's an example with 4 objective functions:

```python
# Define 4 objective functions
def objective_1(coords):
    """Maximize sum of coordinates"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """Maximize spread"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

def objective_3(coords):
    """Minimize distance to center"""
    center = np.array([1.0, 0.5])
    distances = np.linalg.norm(coords - center, axis=1)
    return -np.mean(distances)

def objective_4(coords):
    """Minimize distance to boundary"""
    boundary_distances = []
    for point in coords:
        min_dist = min(point[0], point[1], 2-point[0], 1-point[1])
        boundary_distances.append(min_dist)
    return -np.mean(boundary_distances)

# Create problem with 4 objectives
problem = Problem(
    objectives=[objective_1, objective_2, objective_3, objective_4],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)
```

## API Reference

### Core Classes

#### Problem
The main problem definition class for multi-objective optimization.

```python
Problem(objectives, n_points, region, constraints=[], penalty_weight=1e6)
```

**Parameters:**
- `objectives`: List of objective functions (callable), each taking coords as input and returning a scalar value
- `n_points`: Number of coordinate points to optimize
- `region`: Shapely Polygon defining the valid region
- `constraints`: List of constraint functions (optional)
- `penalty_weight`: Weight for constraint violations (default: 1e6)

#### CoordsNSGA2
The main optimizer class implementing NSGA-II for coordinate optimization.

```python
CoordsNSGA2(problem, pop_size, prob_crs, prob_mut, random_seed=42)
```

**Parameters:**
- `problem`: Problem instance
- `pop_size`: Population size (must be even)
- `prob_crs`: Crossover probability
- `prob_mut`: Mutation probability
- `random_seed`: Random seed for reproducibility

**Methods:**
- `run(generations, verbose=True)`: Run optimization for specified number and verbose type of generations
- `save(path)`: Save optimization state to file
- `load(path)`: Load optimization state from file

**Attributes:**
- `values_P`: Tuple of objective function values for current population
- `values_history`: List of objective function values for all generations

### Spatial Utilities

#### region_from_points(points)
Create a polygon region from a list of coordinate points.

#### region_from_range(x_min, x_max, y_min, y_max)
Create a rectangular region from coordinate bounds.

#### create_points_in_polygon(polygon, n)
Generate n random points within a polygon.

### Genetic Operators

#### coords_crossover(population, prob_crs)
Coordinate-specific crossover operator that exchanges point subsets between parents.

#### coords_mutation(population, prob_mut, region)
Coordinate-specific mutation operator that randomly repositions points within the region.

#### coords_selection(population, values_P, tourn_size=3)
Tournament selection based on non-dominated sorting and crowding distance.

## Examples

See the `examples/` directory for more detailed usage examples:

- [Quick Start Example](examples/quick-start.py) - Basic usage with 2 objectives
- [Multi-Objective Example](examples/multi-objective-example.py) - Advanced usage with 4 objectives and visualization

## Documentation

Complete documentation is available in the [docs/](docs) folder.

To start the documentation server locally:
```bash
mkdocs serve
```

To build the documentation:
```bash
mkdocs build
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{coords_nsga2,
  title={coords-nsga2: A Python library for coordinate-based multi-objective optimization},
  author={Zhang, Xiaofeng},
  year={2024},
  url={https://github.com/ZXF1001/coords-nsga2}
}
```
