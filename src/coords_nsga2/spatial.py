import numpy as np
from shapely.geometry import Point


def create_point_in_polygon(polygons, is_int=False):
    minx, miny, maxx, maxy = polygons.bounds
    while True:
        if is_int:
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)
        else:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
        if polygons.contains(Point(x, y)):
            return x, y
