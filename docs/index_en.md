# Welcome to coords-nsga2

> **⚠️ Important Notice**: This documentation is AI-generated based on source code analysis. While we strive for accuracy, there may be inconsistencies or issues. We are actively working to improve and verify all content. Please report any problems you encounter.

A Python library implementing coordinate-based multi-objective optimization using NSGA-II algorithm.

## Overview

coords-nsga2 is a Python library specifically designed for optimizing coordinate point layouts. It is based on the classic NSGA-II (Non-dominated Sorting Genetic Algorithm II) algorithm but has been specially improved for coordinate optimization problems, including:

- Specialized coordinate crossover and mutation operators
- Built-in constraint handling mechanisms
- Flexible region definition
- Efficient optimization algorithms

## Key Features

- **Coordinate-focused optimization**: Specifically designed for optimizing coordinate point layouts
- **Specialized constraints**: Built-in support for point spacing, boundary limits, and custom constraints
- **Tailored genetic operators**: Custom crossover and mutation operators that directly act on coordinate points
- **Multi-objective optimization**: Based on the proven NSGA-II algorithm
- **Flexible region definition**: Support for both polygon and rectangular regions
- **Lightweight and extensible**: Easy to customize operators and constraints
- **Progress tracking**: Built-in progress bars and optimization history
- **Save/Load functionality**: Save and restore optimization states

## Quick Start

- **Installation**: See [Installation Guide](install.md)
- **Usage**: Detailed tutorial see [Usage Guide](usage.md)
- **API Reference**: See [API Documentation](api.md)
- **Examples**: See [Example Code](examples.md)

## Application Scenarios

- Wind turbine layout optimization
- Sensor network deployment
- Facility location problems
- Robot path planning
- Other scenarios requiring coordinate point layout optimization

## System Requirements

- Python 3.9+
- NumPy >= 1.23
- tqdm >= 4
- Shapely >= 2
- SciPy (optional, for distance calculations)
