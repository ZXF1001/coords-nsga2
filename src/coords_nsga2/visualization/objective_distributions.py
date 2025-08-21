import matplotlib.pyplot as plt
import numpy as np


def plot_objective_distributions(optimizer, figsize=None, is_show=True):
    """
    Plot distribution of objective function values

    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    """
    n_objectives = len(optimizer.values_P)

    # Create subplots
    cols = min(2, n_objectives)
    rows = (n_objectives + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1:
        axes = axes.reshape(1, -1)

    for obj in range(n_objectives):
        row, col = obj // cols, obj % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        values = optimizer.values_P[obj]

        # Create histogram
        ax.hist(values, bins=20, alpha=0.7, color=f'C{obj}', edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='orange', linestyle='--',
                   label=f'Median: {np.median(values):.2f}')

        ax.set_xlabel(f'Objective {obj} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Objective {obj}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for obj in range(n_objectives, rows * cols):
        row, col = obj // cols, obj % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    if is_show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from coords_nsga2 import CoordsNSGA2

    # 这些是pickle读取时必要的，但是内容不重要
    objective_1 = objective_2 = objective_3 = objective_4 = constraint_spacing = None

    loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")

    plot_objective_distributions(loaded_optimizer)
