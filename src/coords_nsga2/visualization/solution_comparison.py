"""
Solution comparison visualization tool
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


def plot_solution_comparison(optimizer, solution_indices=None, figsize=(15, 10), save_path=None):
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
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(optimizer, 'P'):
        raise ValueError("No optimization results found. Run optimization first.")
    
    if solution_indices is None:
        # Select diverse solutions
        n_solutions = min(6, len(optimizer.P))
        solution_indices = np.linspace(0, len(optimizer.P)-1, n_solutions, dtype=int)
    
    n_solutions = len(solution_indices)
    cols = min(3, n_solutions)
    rows = (n_solutions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_solutions == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sol_idx in enumerate(solution_indices):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Plot region boundary
        if hasattr(optimizer, 'problem') and hasattr(optimizer.problem, 'region'):
            _plot_region_boundary(ax, optimizer.problem.region)
        
        # Plot solution
        solution = optimizer.P[sol_idx]
        ax.scatter(solution[:, 0], solution[:, 1], 
                  c=f'C{i}', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add objective values in title
        if hasattr(optimizer, 'values_P'):
            obj_values = [f'{optimizer.values_P[j][sol_idx]:.3f}' for j in range(len(optimizer.values_P))]
            ax.set_title(f'Solution {sol_idx}\nObjectives: [{", ".join(obj_values)}]')
        else:
            ax.set_title(f'Solution {sol_idx}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(n_solutions, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if figsize:
        plt.show()