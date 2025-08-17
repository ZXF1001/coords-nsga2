"""
Optimal layouts visualization tool
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def _plot_region_boundary(ax, region):
    """Plot the optimization region boundary"""
    if hasattr(region, 'exterior'):
        # Shapely polygon
        x, y = region.exterior.xy
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.7)
        ax.fill(x, y, alpha=0.1, color='gray')
    elif isinstance(region, np.ndarray):
        # Numpy array of points
        polygon = Polygon(region, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(polygon)
        ax.fill(region[:, 0], region[:, 1], alpha=0.1, color='gray')


def plot_objective_optimal_layouts(optimizer, figsize=(15, 10), save_path=None):
    """
    Plot layouts for solutions that are optimal in each objective
    
    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(optimizer, 'values_P') or not hasattr(optimizer, 'P'):
        raise ValueError("No optimization results found. Run optimization first.")
    
    n_objectives = len(optimizer.values_P)
    
    # Find best solution for each objective
    best_indices = []
    for i in range(n_objectives):
        best_idx = np.argmax(optimizer.values_P[i])
        best_indices.append(best_idx)
    
    # Create subplots
    cols = min(3, n_objectives)
    rows = (n_objectives + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_objectives == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_objectives):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Plot region boundary
        if hasattr(optimizer, 'problem') and hasattr(optimizer.problem, 'region'):
            _plot_region_boundary(ax, optimizer.problem.region)
        
        # Plot optimal solution for this objective
        best_solution = optimizer.P[best_indices[i]]
        ax.scatter(best_solution[:, 0], best_solution[:, 1], 
                  c='red', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        ax.set_title(f'Optimal Layout for Objective {i+1}\n'
                    f'Value: {optimizer.values_P[i][best_indices[i]]:.4f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(n_objectives, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
