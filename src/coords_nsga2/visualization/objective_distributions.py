import matplotlib.pyplot as plt
import numpy as np


def plot_objective_distributions(optimizer, generation = -1, figsize=None, is_show=True):
    """
    Plot distribution of objective function values

    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    generation : int
        Generation to plot (-1 for latest generation)
    figsize : tuple
        Figure size
    """
    # 根据generation参数选择数据源
    # 允许负数索引，例如-1表示最新一代
    if abs(generation) >= len(optimizer.values_history):
        raise ValueError(f"Generation {generation} is out of bounds. Must be between {-len(optimizer.values_history)} and {len(optimizer.values_history) - 1}.")
    
    values_to_plot = optimizer.values_history[generation]
    
    n_objectives = len(values_to_plot)

    # Create subplots
    cols = min(2, n_objectives)
    rows = (n_objectives + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1:
        axes = axes.reshape(1, -1)

    for obj in range(n_objectives):
        row, col = obj // cols, obj % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        values = values_to_plot[obj]

        # Create histogram
        ax.hist(values, bins=20, alpha=0.7, color=f'C{obj}', edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='orange', linestyle='--',
                   label=f'Median: {np.median(values):.2f}')

        ax.set_xlabel(f'Objective {obj} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Objective {obj}\n(Generation: {generation})')
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
