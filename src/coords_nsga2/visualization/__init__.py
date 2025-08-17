"""
Visualization tools for coords-nsga2 optimization results
Each tool is implemented as an independent function that takes optimizer data as input
"""

from .constraint_violations import plot_constraint_violations
from .convergence_analysis import plot_convergence_analysis
from .hypervolume_trend import plot_hypervolume_trend
from .objective_correlations import plot_objective_correlations
from .objective_distributions import plot_objective_distributions
from .objective_optimal_layouts import plot_objective_optimal_layouts
from .parallel_coordinates import plot_parallel_coordinates
from .pareto_front import plot_pareto_front
from .solution_comparison import plot_solution_comparison

__all__ = [
    'plot_pareto_front',
    'plot_parallel_coordinates', 
    'plot_hypervolume_trend',
    'plot_objective_optimal_layouts',
    'plot_solution_comparison',
    'plot_convergence_analysis',
    'plot_constraint_violations',
    'plot_objective_distributions',
    'plot_objective_correlations'
]
