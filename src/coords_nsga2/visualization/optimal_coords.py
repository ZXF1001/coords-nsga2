import matplotlib.pyplot as plt
import numpy as np


def _plot_region_boundary(ax, region):
    """Plot the optimization region boundary"""
    if hasattr(region, 'exterior'):
        # Shapely polygon
        x, y = region.exterior.xy
        ax.plot(x, y, 'k', alpha=0.7)
        ax.fill(x, y, alpha=0.1, color='gray')


def plot_optimal_coords(optimizer, obj_indices, figsize, is_show):
    if isinstance(obj_indices, int):
        obj_indices = [obj_indices]

    # Find best solution for each objective
    for obj_index in obj_indices:
        best_idx = np.argmax(optimizer.values_P[obj_index])
        fig, ax = plt.subplots(figsize=figsize)
        _plot_region_boundary(ax, optimizer.problem.region)
        best_solution = optimizer.P[best_idx]
        ax.scatter(best_solution[:, 0], best_solution[:,
                   1],  alpha=0.8, edgecolors='black')

        ax.set_title(f'Optimal Layout for Objective {obj_index}\n'
                     f'Value: {optimizer.values_P[obj_index][best_idx]:.4f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if is_show:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    from coords_nsga2 import CoordsNSGA2
    # 这些是pickle读取时必要的，但是内容不重要
    objective_1 = objective_2 = objective_3 = objective_4 = constraint_spacing = None

    loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")

    plot_optimal_coords(loaded_optimizer, 1, (8, 6), True)
    plot_optimal_coords(loaded_optimizer, [0, 1], (8, 6), True)
