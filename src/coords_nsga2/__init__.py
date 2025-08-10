from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # 未安装（在源码目录里跑）
    __version__ = "0.0.0"

if __name__ == "__main__":  # pragma: no cover
    print(__version__)