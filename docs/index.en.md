![Coords-NSGA2](logo.drawio.svg)
# Welcome to Coords-NSGA2

> **⚠️ Important Notice**: This document is AI-generated based on source-code analysis. Although we strive for accuracy, inconsistencies or issues may still exist. We are actively improving and validating all content. If you encounter any problems, please report them promptly.

Coords-NSGA2 is a Python library for multi-objective optimization of coordinate layouts, built upon an enhanced version of the NSGA-II algorithm.

## Overview

Coords-NSGA2 is a Python library specifically designed for optimizing the layout of coordinate points, based on an improved implementation of the classic NSGA-II (Non-Dominated Sorting Genetic Algorithm II).

## Key Features

- **Coordinate-focused optimization**: Purpose-built for optimizing point layouts  
- **Specialized constraints**: Built-in support for inter-point spacing, boundary limits, and custom constraints  
- **Customized genetic operators**: Crossover and mutation operators tailored to coordinate data  
- **Multi-objective optimization**: Leveraging the proven NSGA-II algorithm  
- **Flexible region definitions**: Supports both polygonal and rectangular regions  
- **Lightweight and extensible**: Easy to add custom operators and constraints  
- **Progress tracking**: Built-in progress bar and optimization history  
- **Save/Load capability**: Persist and restore optimization state  

## Quick Start

- **Installation**: See the [Installation Guide](install.md)  
- **Usage**: Detailed tutorial in the [User Guide](usage.md)  
- **API Reference**: Refer to the [API Documentation](api.md)  
- **Examples**: Browse the [Example Code](examples.md)  

## Use Cases

- Wind turbine layout optimization  
- Sensor network deployment  
- Facility location problems  
- Robot path planning  
- Any scenario requiring optimized coordinate layouts  

## System Requirements

- Python 3.9+
- NumPy >= 1.23
- tqdm >= 4
- Shapely >= 2
- SciPy (optional, for distance computations)
- matplotlib (optional, for result visualization)