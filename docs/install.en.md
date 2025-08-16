# Installation Guide

> **⚠️ Important Notice**: This document is AI-generated based on source-code analysis. Although we strive for accuracy, inconsistencies or issues may still exist. We are actively improving and validating all content. If you encounter any problems, please report them promptly.

## Installation (Chinese)

### System Requirements

- Python 3.9 or later
- pip package manager

### Install from PyPI (Recommended)

```bash
pip install coords-nsga2
```

### Install from Source

If you want the latest development version or plan to modify the code:

```bash
git clone https://github.com/ZXF1001/coords-nsga2.git
cd coords-nsga2
pip install -e .
```

### Set Up a Development Environment

If you intend to contribute:

```bash
git clone https://github.com/ZXF1001/coords-nsga2.git
cd coords-nsga2
pip install -e ".[test]"
```
### Verify the Installation

After installation, verify with:

```python
import coords_nsga2
print(coords_nsga2.__version__)
```

## Dependencies
### Required
- **numpy >= 1.23**: Numerical computation
- **tqdm >= 4**: Progress bars
- **shapely >= 2**: Geometry operations

### Optional

- **scipy**: Distance calculations and other scientific computing
- **matplotlib**: Result visualization

### Development

- **pytest >= 8.2**: Testing framework
- **pytest-cov >= 5**: Test coverage
- **coverage[toml] >= 7.5**: Code coverage
- **hypothesis >= 6.100**: Property-based testing
- **ruff >= 0.11**: Code formatting and linting
- **pre-commit >= 3.7**: Git hooks