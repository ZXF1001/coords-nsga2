from typing import List, Union

import numpy as np
import shapely
from shapely.geometry import Point, Polygon


def region_from_points(points: Union[np.ndarray, List[List[float]]]) -> Polygon:
    """
    Generates a polygonal region from an array of vertex coordinates.
    
    Args:
        points: An array or list of lists of vertex coordinates, e.g., [[x1, y1], [x2, y2], ...].
        
    Returns:
        A Shapely Polygon object representing the defined region.
    """
    return Polygon(points)


def region_from_range(x_min: float, x_max: float, y_min: float, y_max: float) -> Polygon:
    """
    Generates a rectangular region from a given range of x and y coordinates.
    
    Args:
        x_min: Minimum x-coordinate.
        x_max: Maximum x-coordinate.
        y_min: Minimum y-coordinate.
        y_max: Maximum y-coordinate.
        
    Returns:
        A Shapely Polygon object representing the rectangular region.
    """
    return Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])


def create_points_in_polygon(polygons: Polygon, n: int = 1) -> np.ndarray:
    """
    Generates random points within a given polygonal region.
    
    Args:
        polygons: A Shapely Polygon object defining the boundary.
        n: The number of points to generate. Defaults to 1.
        
    Returns:
        A NumPy array of shape (n, 2) containing the generated points.
    """
    minx, miny, maxx, maxy = polygons.bounds
    points = np.empty((0, 2))
    
    # Adaptive batch size
    batch_size = max(n * 2, 100)
    
    while len(points) < n:
        # Generate batch
        xs = np.random.uniform(minx, maxx, batch_size)
        ys = np.random.uniform(miny, maxy, batch_size)
        
        # Vectorized check
        candidates = shapely.points(xs, ys)
        mask = shapely.contains(polygons, candidates)
        
        if np.any(mask):
            valid_points = np.column_stack((xs[mask], ys[mask]))
            points = np.vstack((points, valid_points))
            
    return points[:n]
