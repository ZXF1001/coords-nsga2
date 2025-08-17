"""
Objective correlations visualization tool
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_objective_correlations(optimizer, figsize=(10, 8), save_path=None):
    """
    Plot correlation heatmap between objectives
    
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
    
    if n_objectives < 2:
        print("Need at least 2 objectives for correlation analysis")
        return
    
    # Calculate correlation matrix
    correlation_matrix = np.zeros((n_objectives, n_objectives))
    p_values = np.zeros((n_objectives, n_objectives))
    
    for i in range(n_objectives):
        for j in range(n_objectives):
            if i == j:
                correlation_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                corr, p_val = pearsonr(optimizer.values_P[i], optimizer.values_P[j])
                correlation_matrix[i, j] = corr
                p_values[i, j] = p_val
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks and labels
    ax.set_xticks(range(n_objectives))
    ax.set_yticks(range(n_objectives))
    ax.set_xticklabels([f'Obj {i+1}' for i in range(n_objectives)])
    ax.set_yticklabels([f'Obj {i+1}' for i in range(n_objectives)])
    
    # Add correlation values as text
    for i in range(n_objectives):
        for j in range(n_objectives):
            text = f'{correlation_matrix[i, j]:.3f}'
            if p_values[i, j] < 0.05:
                text += '*'  # Mark significant correlations
            ax.text(j, i, text, ha='center', va='center', 
                   color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    ax.set_title('Objective Function Correlations\n(* indicates p < 0.05)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print correlation summary
    print("Correlation Summary:")
    for i in range(n_objectives):
        for j in range(i+1, n_objectives):
            corr = correlation_matrix[i, j]
            p_val = p_values[i, j]
            significance = " (significant)" if p_val < 0.05 else " (not significant)"
            print(f"Objective {i+1} vs Objective {j+1}: r = {corr:.3f}, p = {p_val:.3f}{significance}")
