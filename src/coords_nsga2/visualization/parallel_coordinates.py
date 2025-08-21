import matplotlib.pyplot as plt
import numpy as np


def plot_parallel_coordinates(optimizer, figsize=None, is_show=True):
    """
    Plot parallel coordinates for multi-objective solutions

    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    """

    n_objectives = len(optimizer.values_P)
    if n_objectives < 3:
        print("Parallel coordinates plot is most useful for 3+ objectives")

    # Normalize objectives to [0, 1] for better visualization
    normalized_values = np.zeros_like(optimizer.values_P)
    for i in range(n_objectives):
        obj_values = optimizer.values_P[i]
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
    ax.set_title('Parallel Coordinates Plot of Pareto Solutions')
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
