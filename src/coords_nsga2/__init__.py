from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

# 公共 API
from .algorithm import CoordsNSGA2
__all__ = ["CoordsNSGA2", "__version__"]

if __name__ == "__main__":
    print(__version__)