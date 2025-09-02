import matplotlib.pyplot as plt

from .utils import _plot_region_boundary


def plot_solution_comparison(optimizer, solution_indices, generation=-1, figsize=None, is_show=True):
    """
    Compare multiple solutions side by side

    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    solution_indices : list, optional
        Indices of solutions to compare. If None, selects diverse solutions
    figsize : tuple
        Figure size
    """
    # 根据generation参数选择数据源
    # 允许负数索引，例如-1表示最新一代
    if abs(generation) >= len(optimizer.values_history):
        raise ValueError(
            f"Generation {generation} is out of bounds. Must be between {-len(optimizer.values_history)} and {len(optimizer.values_history) - 1}.")

    values_to_plot = optimizer.values_history[generation]
    coords_to_plot = optimizer.P_history[generation]

    n_solutions = len(solution_indices)

    fig, axes = plt.subplots(1, n_solutions, figsize=figsize)

    for i, sol_idx in enumerate(solution_indices):
        ax = axes[i]
        # Plot region boundary
        if hasattr(optimizer, 'problem') and hasattr(optimizer.problem, 'region'):
            _plot_region_boundary(ax, optimizer.problem.region)

        # Plot solution
        solution = coords_to_plot[sol_idx]

        ax.scatter(
            solution[:, 0],
            solution[:, 1],
            color=f'C{i}',
            alpha=0.8,
            edgecolors='k')
        is_max_or_min = ""
        if hasattr(optimizer, "n_points_max"):
            if len(solution) == optimizer.n_points_max:
                is_max_or_min = " (Max)"
            elif len(solution) == optimizer.n_points_min:
                is_max_or_min = " (Min)"

        obj_values = [f'{values_to_plot[j][sol_idx]:.2f}' for j in range(
            len(values_to_plot))]
        generation_label = generation if generation >= 0 \
            else len(optimizer.P_history) + generation
        ax.set_title(
            f'Generation: {generation_label}\n'
            f'Solution: {sol_idx}\n'
            f'Points Number: {len(solution)}{is_max_or_min}\n'
            f'Objectives: [{", ".join(obj_values)}]')
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

    plot_solution_comparison(loaded_optimizer, [1, 5])
