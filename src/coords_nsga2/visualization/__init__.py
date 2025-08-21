class Plotting:
    def __init__(self, optimizer_instance):
        self.optimizer = optimizer_instance

    def pareto_front(self, obj_indices, figsize=None, is_show=True):
        from .pareto_front import plot_pareto_front
        return plot_pareto_front(self.optimizer, obj_indices, figsize, is_show)

    def optimal_coords(self, obj_indices, figsize=None, is_show=True):
        from .optimal_coords import plot_optimal_coords
        return plot_optimal_coords(self.optimizer, obj_indices, figsize, is_show)

    def solution_comparison(self, solution_indices, figsize=None, is_show=True):
        from .solution_comparison import plot_solution_comparison
        return plot_solution_comparison(self.optimizer, solution_indices, figsize, is_show)

    def objective_correlations(self, figsize=None, is_show=True):
        from .objective_correlations import plot_objective_correlations
        return plot_objective_correlations(self.optimizer, figsize, is_show)

    def objective_distributions(self, figsize=None, is_show=True):
        from .objective_distributions import plot_objective_distributions
        return plot_objective_distributions(self.optimizer, figsize, is_show)

    def constraint_violations(self, figsize=None, is_show=True):
        from .constraint_violations import plot_constraint_violations
        return plot_constraint_violations(self.optimizer, figsize, is_show)

    def parallel_coordinates(self, figsize=None, is_show=True):
        from .parallel_coordinates import plot_parallel_coordinates
        return plot_parallel_coordinates(self.optimizer, figsize, is_show)
