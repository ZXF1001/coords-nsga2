# coords-nsga2

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/ZXF1001/coord-nsga2.svg)](https://github.com/ZXF1001/coord-nsga2/issues)
[![Forks](https://img.shields.io/github/forks/ZXF1001/coord-nsga2.svg)](https://github.com/ZXF1001/coord-nsga2/network)
[![Stars](https://img.shields.io/github/stars/ZXF1001/coord-nsga2.svg)](https://github.com/ZXF1001/coord-nsga2/stargazers)


A Python library implementing a coordinate-based NSGA-II for multi-objective optimization. It features specialized constraints, crossover, and mutation operators that work directly on coordinate points. (developing...)

--------------------------------------------------------------------------------

## Table of Contents
- [coords-nsga2](#coords-nsga2)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Examples](#examples)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

--------------------------------------------------------------------------------

## Features
- Coordinate-focused constraints (e.g., point spacing, boundary limits)
- Tailored crossover and mutation operators that directly act on coordinate points
- Lightweight, extensible design for customizing operators

--------------------------------------------------------------------------------
## Installation

To install from PyPI:
```bash
pip install coord-nsga2
```

Or install the latest development version from GitHub:
```bash
git clone https://github.com/YourUsername/coord-nsga2.git
cd coord-nsga2
pip install -e .
```

--------------------------------------------------------------------------------

## Quick Start
Below is a minimal example demonstrating how to run a coordinate-based NSGA-II optimization using this library:

```python
import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# Define the optimization regions
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# Define your objective functions
def objective_1(coords):
    # coords is a array of (x, y) points
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

# Define constraints if needed
spacing = 0.05  # spacing constraint
def constraint_1(coords):
    # Return total penalty
    dist_list = distance.pdist(coords)
    penalty_list = spacing-dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

# Setup the problem
problem = Problem(func1=objective_1,
                  func2=objective_2,
                  n_points=10,
                  region=region,
                  constraints=[constraint_1])

# Initialize the optimizer
optimizer = CoordsNSGA2(problem=problem,
                        pop_size=20,
                        prob_crs=0.5,
                        prob_mut=0.1,
                        verbose=True)  # Set to False to disable progress bar

# Run optimization
result = optimizer.run(1000)

# Optimization results
print(result)
```

Check the [Examples](#examples) and [Documentation](#documentation) sections below for more detailed usage scenarios.

--------------------------------------------------------------------------------

## Examples
building
<!-- - [Basic Example](examples/basic_example.py)  
- [Multiple Constraints Example](examples/advanced_constraints.py)  
- [Integration with Other Libraries](examples/integration_example.py)   -->

--------------------------------------------------------------------------------

## Documentation
building
Complete documentation is available in the [docs/](docs) folder.

To start the documentation on local server:
```bash
mkdocs serve
```

To build the documentation locally:
```bash
mkdocs build
```

--------------------------------------------------------------------------------

## Contributing
Contributions of all kinds are welcome! To get started:  
1. Fork the repository and clone it locally.  
2. Create a new git branch for your feature or bugfix.  
3. Make changes with clear and concise commit messages.  
4. Submit a pull request describing your changes in detail.  

--------------------------------------------------------------------------------

## License
This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software in accordance with the license terms.

--------------------------------------------------------------------------------

Feel free to modify or extend this template to better suit your project’s structure and requirements.
