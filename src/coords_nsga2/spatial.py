import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon


def region_from_points(points):
    """根据一组顶点坐标的数组生成一个多边形区域"""
    return Polygon(points)


def region_from_range(x_min, x_max, y_min, y_max):
    """根据一组范围生成一个矩形区域"""
    return Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])


def create_point_in_polygon(polygons, is_int=False):
    minx, miny, maxx, maxy = polygons.bounds
    # 选择合适的随机数生成函数
    rand_func = np.random.randint if is_int else np.random.uniform
    while True:
        x = rand_func(minx, maxx)
        y = rand_func(miny, maxy)
        if polygons.contains(Point(x, y)):
            return x, y