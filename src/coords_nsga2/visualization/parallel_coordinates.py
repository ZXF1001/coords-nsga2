"""
Parallel coordinates visualization tool
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_parallel_coordinates(optimizer, figsize=(12, 6), save_path=None):
    """
    Plot parallel coordinates for multi-objective solutions
    
    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(optimizer, 'values_P'):
        raise ValueError("No optimization results found. Run optimization first.")
    
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
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
