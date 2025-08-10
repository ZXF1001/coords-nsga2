from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

# 公共 API
from .algorithm import coords_nsga2
__all__ = ["coords_nsga2", "__version__"]

if __name__ == "__main__":
    print(__version__)