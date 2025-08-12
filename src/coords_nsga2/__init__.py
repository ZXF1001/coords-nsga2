from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

# 公共 API
from .algorithm import CoordsNSGA2, Problem

__all__ = ["CoordsNSGA2", "Problem", "__version__"]

if __name__ == "__main__":
    print(__version__)