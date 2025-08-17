"""
Hypervolume trend visualization tool with automatic range adjustment
"""

import matplotlib.pyplot as plt
import numpy as np


def _calculate_hypervolume_2d(objectives, reference_point):
    """Calculate 2D hypervolume"""
    points = objectives.T  # Shape: (n_points, 2)
    
    # Sort points by first objective
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    hypervolume = 0
    prev_x = reference_point[0]
    
    for point in sorted_points:
        if point[1] < reference_point[1]:
            width = prev_x - point[0]
            height = reference_point[1] - point[1]
            hypervolume += width * height
            prev_x = point[0]
    
    return max(0, hypervolume)


def _calculate_hypervolume_approx(objectives, reference_point):
    """Approximate hypervolume for higher dimensions"""
    # Simple approximation: sum of dominated volumes
    points = objectives.T
    total_volume = 1.0
    
    for i in range(len(reference_point)):
        obj_range = reference_point[i] - np.min(points[:, i])
        total_volume *= max(0, obj_range)
    
    return total_volume


def plot_hypervolume_trend(optimizer, reference_point=None, figsize=(10, 6), save_path=None, 
                          auto_range=True, convergence_threshold=0.1):
    """
    Plot hypervolume trend over generations with automatic range adjustment
    
    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    reference_point : array-like, optional
        Reference point for hypervolume calculation. If None, uses worst values + 10%
    auto_range : bool, default True
        Whether to automatically adjust the y-axis range to focus on meaningful changes
    convergence_threshold : float, default 0.1
        Threshold for detecting convergence phase (relative change)
    """
    if not hasattr(optimizer, 'values_history'):
        raise ValueError("No optimization history found. Run optimization first.")
    
    hypervolumes = []
    
    for generation_values in optimizer.values_history:
        if reference_point is None:
            # Use worst values as reference point
            ref_point = np.max(generation_values, axis=1) * 1.1
        else:
            ref_point = np.array(reference_point)
        
        # Simple hypervolume calculation (for 2D case)
        if len(generation_values) == 2:
            hv = _calculate_hypervolume_2d(generation_values, ref_point)
        else:
            # For higher dimensions, use a simplified approximation
            hv = _calculate_hypervolume_approx(generation_values, ref_point)
        
        hypervolumes.append(hv)
    
    hypervolumes = np.array(hypervolumes)
    
    # Auto-range detection to focus on meaningful changes
    if auto_range and len(hypervolumes) > 10:
        # Find the convergence point where changes become small
        relative_changes = np.abs(np.diff(hypervolumes)) / (np.abs(hypervolumes[:-1]) + 1e-10)
        
        # Find where relative changes consistently drop below threshold
        convergence_start = None
        window_size = min(10, len(relative_changes) // 4)
        
        for i in range(window_size, len(relative_changes)):
            window_changes = relative_changes[i-window_size:i]
            if np.mean(window_changes) < convergence_threshold:
                convergence_start = i - window_size
                break
        
        if convergence_start is not None and convergence_start > 5:
            # Focus on the convergence phase
            start_idx = max(0, convergence_start - 5)
            focused_hypervolumes = hypervolumes[start_idx:]
            focused_generations = range(start_idx, len(hypervolumes))
            
            # Create subplot with both full and focused views
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.5))
            
            # Full view
            ax1.plot(range(len(hypervolumes)), hypervolumes, 'b-', linewidth=2, marker='o', markersize=3)
            ax1.axvline(x=start_idx, color='red', linestyle='--', alpha=0.7, label=f'Focus start (gen {start_idx})')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Hypervolume')
            ax1.set_title('Hypervolume Trend - Full View')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Focused view
            ax2.plot(focused_generations, focused_hypervolumes, 'g-', linewidth=2, marker='o', markersize=4)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Hypervolume')
            ax2.set_title(f'Hypervolume Trend - Focused View (from generation {start_idx})')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            final_hv = hypervolumes[-1]
            start_hv = hypervolumes[start_idx]
            improvement = final_hv - start_hv
            rel_improvement = improvement / abs(start_hv) * 100 if start_hv != 0 else 0
            
            stats_text = f'Improvement in focused range:\nAbsolute: {improvement:.6f}\nRelative: {rel_improvement:.3f}%'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        else:
            # No clear convergence detected, show full range
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(range(len(hypervolumes)), hypervolumes, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Hypervolume')
            ax.set_title('Hypervolume Trend Over Generations')
            ax.grid(True, alpha=0.3)
    else:
        # Standard single plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(hypervolumes)), hypervolumes, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Trend Over Generations')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("Hypervolume Statistics:")
    print(f"  Initial: {hypervolumes[0]:.6f}")
    print(f"  Final: {hypervolumes[-1]:.6f}")
    print(f"  Total improvement: {hypervolumes[-1] - hypervolumes[0]:.6f}")
    if len(hypervolumes) > 1:
        rel_improvement = (hypervolumes[-1] - hypervolumes[0]) / abs(hypervolumes[0]) * 100
        print(f"  Relative improvement: {rel_improvement:.3f}%")
