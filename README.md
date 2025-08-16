![Coords-NSGA2](./docs/logo.drawio.svg)

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
@software{Coords-NSGA2,
  title={Coords-NSGA2: A Python library for coordinate-based multi-objective optimization},
  author={Zhang, Xiaofeng},
  year={2025},
  url={https://github.com/ZXF1001/coords-nsga2}
}
```
