class VisualizationPlotter:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def hypervolume_trend(self, **kwargs):
        from .hypervolume_trend import plot_hypervolume_trend
        return plot_hypervolume_trend(self.optimizer, **kwargs)
    
    def pareto_front(self, **kwargs):
        from .pareto_front import plot_pareto_front
        return plot_pareto_front(self.optimizer, **kwargs)
    
    def parallel_coordinates(self, **kwargs):
        from .parallel_coordinates import plot_parallel_coordinates
        return plot_parallel_coordinates(self.optimizer, **kwargs)
    
    def constraint_violations(self, **kwargs):
        from .constraint_violations import plot_constraint_violations
        return plot_constraint_violations(self.optimizer, **kwargs)
    
    def convergence_analysis(self, **kwargs):
        from .convergence_analysis import plot_convergence_analysis
        return plot_convergence_analysis(self.optimizer, **kwargs)
    
    def objective_correlations(self, **kwargs):
        from .objective_correlations import plot_objective_correlations
        return plot_objective_correlations(self.optimizer, **kwargs)
    
    def objective_distributions(self, **kwargs):
        from .objective_distributions import plot_objective_distributions
        return plot_objective_distributions(self.optimizer, **kwargs)
        
    def objective_optimal_layouts(self, **kwargs):
        from .objective_optimal_layouts import plot_objective_optimal_layouts
        return plot_objective_optimal_layouts(self.optimizer, **kwargs)
        
    def solution_comparison(self, **kwargs):
        from .solution_comparison import plot_solution_comparison
        return plot_solution_comparison(self.optimizer, **kwargs)
