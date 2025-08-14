import numpy as np
from shapely.geometry import Point, Polygon


def region_from_points(points):
    """根据一组顶点坐标的数组生成一个多边形区域"""
    return Polygon(points)


def region_from_range(x_min, x_max, y_min, y_max):
    """根据一组范围生成一个矩形区域"""
    return Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])


def create_points_in_polygon(polygons, n=1):
    minx, miny, maxx, maxy = polygons.bounds
    n_valid = 0
    points = []
    # 选择合适的随机数生成函数
    while n_valid < n:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if polygons.contains(Point(x, y)):
            points.append([x, y])
            n_valid += 1
    return np.array(points)