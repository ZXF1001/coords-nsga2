"""
Constraint violations visualization tool
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_constraint_violations(optimizer, figsize=(12, 8), save_path=None):
    """
    Plot constraint violation statistics
    
    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(optimizer, 'P_history') or not hasattr(optimizer, 'problem'):
        print("No constraint information available")
        return
    
    if not hasattr(optimizer.problem, 'constraints') or not optimizer.problem.constraints:
        print("No constraints defined in the problem")
        return
    
    generations = range(len(optimizer.P_history))
    violation_stats = []
    
    for population in optimizer.P_history:
        violations = []
        for individual in population:
            total_violation = sum([constraint(individual) for constraint in optimizer.problem.constraints])
            violations.append(max(0, total_violation))  # Only positive violations
        
        violation_stats.append({
            'mean': np.mean(violations),
            'max': np.max(violations),
            'feasible_ratio': np.sum(np.array(violations) == 0) / len(violations)
        })
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Mean violation over generations
    mean_violations = [stats['mean'] for stats in violation_stats]
    axes[0, 0].plot(generations, mean_violations, 'r-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Mean Constraint Violation')
    axes[0, 0].set_title('Mean Constraint Violation Trend')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Max violation over generations
    max_violations = [stats['max'] for stats in violation_stats]
    axes[0, 1].plot(generations, max_violations, 'orange', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Max Constraint Violation')
    axes[0, 1].set_title('Maximum Constraint Violation Trend')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feasible solutions ratio
    feasible_ratios = [stats['feasible_ratio'] for stats in violation_stats]
    axes[1, 0].plot(generations, feasible_ratios, 'g-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Feasible Solutions Ratio')
    axes[1, 0].set_title('Feasible Solutions Ratio Trend')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final generation violation distribution
    final_violations = []
    for individual in optimizer.P_history[-1]:
        total_violation = sum([constraint(individual) for constraint in optimizer.problem.constraints])
        final_violations.append(max(0, total_violation))
    
    axes[1, 1].hist(final_violations, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Constraint Violation')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Final Generation Violation Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
