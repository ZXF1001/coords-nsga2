"""
Convergence analysis visualization tool
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_analysis(optimizer, figsize=(15, 10), save_path=None):
    """
    Plot convergence analysis for each objective
    
    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(optimizer, 'values_history'):
        raise ValueError("No optimization history found. Run optimization first.")
    
    n_objectives = len(optimizer.values_history[0])
    
    # Calculate statistics for each generation
    generations = range(len(optimizer.values_history))
    best_values = np.zeros((n_objectives, len(generations)))
    mean_values = np.zeros((n_objectives, len(generations)))
    std_values = np.zeros((n_objectives, len(generations)))
    
    for gen, values in enumerate(optimizer.values_history):
        for obj in range(n_objectives):
            best_values[obj, gen] = np.max(values[obj])
            mean_values[obj, gen] = np.mean(values[obj])
            std_values[obj, gen] = np.std(values[obj])
    
    # Create subplots
    cols = min(2, n_objectives)
    rows = (n_objectives + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_objectives == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for obj in range(n_objectives):
        row, col = obj // cols, obj % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Plot best, mean, and confidence interval
        ax.plot(generations, best_values[obj], 'r-', linewidth=2, label='Best')
        ax.plot(generations, mean_values[obj], 'b-', linewidth=2, label='Mean')
        ax.fill_between(generations, 
                       mean_values[obj] - std_values[obj],
                       mean_values[obj] + std_values[obj],
                       alpha=0.3, color='blue', label='±1 Std')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel(f'Objective {obj+1} Value')
        ax.set_title(f'Convergence Analysis - Objective {obj+1}')
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
