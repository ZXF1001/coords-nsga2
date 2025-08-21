import matplotlib.pyplot as plt

from .utils import _plot_region_boundary


def plot_solution_comparison(optimizer, solution_indices, figsize=None, is_show=True):
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
    n_solutions = len(solution_indices)

    fig, axes = plt.subplots(n_solutions, 1, figsize=figsize)

    for i, sol_idx in enumerate(solution_indices):
        ax = axes[i]
        # Plot region boundary
        if hasattr(optimizer, 'problem') and hasattr(optimizer.problem, 'region'):
            _plot_region_boundary(ax, optimizer.problem.region)

        # Plot solution
        solution = optimizer.P[sol_idx]

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

        obj_values = [f'{optimizer.values_P[j][sol_idx]:.2f}' for j in range(
            len(optimizer.values_P))]
        ax.set_title(
            f'Solution {sol_idx}\n'
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
