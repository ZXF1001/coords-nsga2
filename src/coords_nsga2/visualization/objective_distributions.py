"""
Objective distributions visualization tool
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_objective_distributions(optimizer, figsize=(15, 10), save_path=None):
    """
    Plot distribution of objective function values
    
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
    
    # Create subplots
    cols = min(3, n_objectives)
    rows = (n_objectives + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_objectives == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for obj in range(n_objectives):
        row, col = obj // cols, obj % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        values = optimizer.values_P[obj]
        
        # Create histogram
        ax.hist(values, bins=20, alpha=0.7, color=f'C{obj}', edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
        
        ax.set_xlabel(f'Objective {obj+1} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Objective {obj+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for obj in range(n_objectives, rows * cols):
        row, col = obj // cols, obj % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
