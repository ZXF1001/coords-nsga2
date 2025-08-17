"""
Pareto front visualization tool
"""

import matplotlib.pyplot as plt


def plot_pareto_front(optimizer, obj_indices=None, figsize=(12, 8), save_path=None):
    """
    Plot 2D or 3D Pareto front
    
    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    obj_indices : list, optional
        Indices of objectives to plot. If None, plots first 2 or 3 objectives
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(optimizer, 'values_P'):
        raise ValueError("No optimization results found. Run optimization first.")
    
    n_objectives = len(optimizer.values_P)
    
    if obj_indices is None:
        if n_objectives >= 3:
            obj_indices = [0, 1, 2]  # 3D plot
        else:
            obj_indices = [0, 1] if n_objectives >= 2 else [0]
    
    if len(obj_indices) == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(optimizer.values_P[obj_indices[0]], optimizer.values_P[obj_indices[1]], 
                  alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'Objective {obj_indices[0]+1}')
        ax.set_ylabel(f'Objective {obj_indices[1]+1}')
        ax.set_title('2D Pareto Front')
        ax.grid(True, alpha=0.3)
        
    elif len(obj_indices) == 3:
        # 3D Pareto front
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(optimizer.values_P[obj_indices[0]], optimizer.values_P[obj_indices[1]], 
                  optimizer.values_P[obj_indices[2]], alpha=0.7, s=50, c='blue', 
                  edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'Objective {obj_indices[0]+1}')
        ax.set_ylabel(f'Objective {obj_indices[1]+1}')
        ax.set_zlabel(f'Objective {obj_indices[2]+1}')
        ax.set_title('3D Pareto Front')
        
    else:
        raise ValueError("Can only plot 2D or 3D Pareto fronts")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
