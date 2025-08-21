import matplotlib.pyplot as plt
import numpy as np


def plot_constraint_violations(optimizer, figsize=None, is_show=True):
    """
    Plot constraint violation statistics

    Parameters:
    -----------
    optimizer : CoordsNSGA2
        The optimizer instance with optimization results
    figsize : tuple
        Figure size
    """

    if not hasattr(optimizer.problem, 'constraints') or not optimizer.problem.constraints:
        print("No constraints defined in the problem")
        return
    
    generations = range(len(optimizer.P_history))
    violation_stats = []

    for population in optimizer.P_history:
        violations = []
        for individual in population:
            total_violation = sum([constraint(individual)
                                  for constraint in optimizer.problem.constraints])
            # Only positive violations
            violations.append(max(0, total_violation))

        violation_stats.append({
            'mean': np.mean(violations),
            'max': np.max(violations),
            'feasible_ratio': np.sum(np.array(violations) == 0) / len(violations)
        })

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Mean violation over generations
    mean_violations = [stats['mean'] for stats in violation_stats]
    axes[0, 0].plot(generations, mean_violations, 'r-',
                    marker='o', markersize=3)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Mean Constraint Violation')
    axes[0, 0].set_title('Mean Constraint Violation Trend')
    axes[0, 0].grid(True, alpha=0.3)

    # Max violation over generations
    max_violations = [stats['max'] for stats in violation_stats]
    axes[0, 1].plot(generations, max_violations, 'orange',
                    marker='s', markersize=3)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Max Constraint Violation')
    axes[0, 1].set_title('Max Constraint Violation Trend')
    axes[0, 1].grid(True, alpha=0.3)

    # Feasible solutions ratio
    feasible_ratios = [stats['feasible_ratio'] for stats in violation_stats]
    axes[1, 0].plot(generations, feasible_ratios, 'g-',
                    marker='^', markersize=4)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Feasible Solutions Ratio')
    axes[1, 0].set_title('Feasible Solutions Ratio Trend')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)

    # Final generation violation distribution
    final_violations = []
    for individual in optimizer.P_history[-1]:
        total_violation = sum([constraint(individual)
                              for constraint in optimizer.problem.constraints])
        final_violations.append(max(0, total_violation))

    axes[1, 1].hist(final_violations, bins=20, alpha=0.7,
                    color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Constraint Violation')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Final Generation Violation Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    if is_show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from scipy.spatial import distance

    from coords_nsga2 import CoordsNSGA2

    # 这些是pickle读取时必要的，但是内容不重要
    objective_1 = objective_2 = objective_3 = objective_4 = None

    def constraint_spacing(coords):
        min_spacing = 0.1  # 间距限制
        """Minimum spacing constraint between points"""
        if len(coords) < 2:
            return 0
        dist_list = distance.pdist(coords)
        violations = min_spacing - dist_list[dist_list < min_spacing]
        return np.sum(violations)

    loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")

    plot_constraint_violations(loaded_optimizer)
