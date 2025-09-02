import matplotlib.pyplot as plt
import numpy as np


def plot_parallel_coordinates(optimizer, generation = -1, figsize=None, is_show=True):
    """
    Plot parallel coordinates for multi-objective solutions

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
    if n_objectives < 3:
        print("Parallel coordinates plot is most useful for 3+ objectives")

    # Normalize objectives to [0, 1] for better visualization
    normalized_values = np.zeros_like(values_to_plot)
    for i in range(n_objectives):
        obj_values = values_to_plot[i]
        min_val, max_val = obj_values.min(), obj_values.max()
        if max_val > min_val:
            normalized_values[i] = (obj_values - min_val) / (max_val - min_val)
        else:
            normalized_values[i] = 0.5  # All values are the same

    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines for each solution
    for i in range(normalized_values.shape[1]):
        ax.plot(range(n_objectives), normalized_values[:, i],
                alpha=0.6, linewidth=1, color='blue')

    ax.set_xticks(range(n_objectives))
    ax.set_xticklabels([f'Obj {i+1}' for i in range(n_objectives)])
    ax.set_ylabel('Normalized Objective Value')
    ax.set_title(f'Parallel Coordinates Plot of Pareto Solutions (Generation: {generation})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    if is_show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from coords_nsga2 import CoordsNSGA2

    # 这些是pickle读取时必要的，但是内容不重要
    objective_1 = objective_2 = objective_3 = objective_4 = constraint_spacing = None

    loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")

    plot_parallel_coordinates(loaded_optimizer)
