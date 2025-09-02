import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_objective_correlations(optimizer, generation=-1, figsize=None, is_show=True):
    """
    Plot correlation heatmap between objectives

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
        raise ValueError(
            f"Generation {generation} is out of bounds. Must be between {-len(optimizer.values_history)} and {len(optimizer.values_history) - 1}.")

    values_to_plot = optimizer.values_history[generation]

    n_objectives = len(values_to_plot)

    # Calculate correlation matrix
    correlation_matrix = np.zeros((n_objectives, n_objectives))
    p_values = np.zeros((n_objectives, n_objectives))

    for i in range(n_objectives):
        for j in range(n_objectives):
            if i == j:
                correlation_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                correlation, pvalue = pearsonr(
                    values_to_plot[i], values_to_plot[j])
                correlation_matrix[i, j] = correlation
                p_values[i, j] = pvalue

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')

    # Set ticks and labels
    ax.set_xticks(range(n_objectives))
    ax.set_yticks(range(n_objectives))
    ax.set_xticklabels([f'Obj {i}' for i in range(n_objectives)])
    ax.set_yticklabels([f'Obj {i}' for i in range(n_objectives)])

    # Add correlation values as text
    for i in range(n_objectives):
        for j in range(n_objectives):
            text = f'{correlation_matrix[i, j]:.3f}'
            if p_values[i, j] < 0.05:
                text += '*'  # Mark significant correlations
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    generation_label = generation if generation >= 0 \
        else len(optimizer.P_history) + generation
    ax.set_title(
        f'Objective Correlations (Generation: {generation_label})\n(* indicates p < 0.05)')

    if is_show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from coords_nsga2 import CoordsNSGA2

    # 这些是pickle读取时必要的，但是内容不重要
    objective_1 = objective_2 = objective_3 = objective_4 = constraint_spacing = None

    loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")

    plot_objective_correlations(loaded_optimizer)
