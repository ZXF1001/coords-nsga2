import matplotlib.pyplot as plt
import numpy as np

from .utils import _plot_region_boundary


def plot_optimal_coords(optimizer, obj_indices, generation=-1, figsize=None, is_show=True):
    # 根据generation参数选择数据源
    # 允许负数索引，例如-1表示最新一代
    if abs(generation) >= len(optimizer.values_history):
        raise ValueError(
            f"Generation {generation} is out of bounds. Must be between {-len(optimizer.values_history)} and {len(optimizer.values_history) - 1}.")

    values_to_plot = optimizer.values_history[generation]
    coords_to_plot = optimizer.P_history[generation]

    if isinstance(obj_indices, int):
        obj_indices = [obj_indices]

    # Find best solution for each objective
    for obj_index in obj_indices:
        best_idx = np.argmax(values_to_plot[obj_index])
        fig, ax = plt.subplots(figsize=figsize)
        _plot_region_boundary(ax, optimizer.problem.region)
        best_solution = coords_to_plot[best_idx]
        ax.scatter(best_solution[:, 0], best_solution[:,
                   1],  alpha=0.8, edgecolors='black')

        is_max_or_min = ""
        if hasattr(optimizer, "n_points_max"):
            if len(best_solution) == optimizer.n_points_max:
                is_max_or_min = " (Max)"
            elif len(best_solution) == optimizer.n_points_min:
                is_max_or_min = " (Min)"
        generation_label = generation if generation >= 0 \
            else len(optimizer.P_history) + generation
        ax.set_title(f'Optimal Layout for Objective {obj_index}\n'
                     f'Generation: {generation_label}, '
                     f'Value: {values_to_plot[obj_index][best_idx]:.4f}, Points Number: {len(best_solution)}{is_max_or_min}')
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

    plot_optimal_coords(loaded_optimizer, 1)
    plot_optimal_coords(loaded_optimizer, [0, 1])
