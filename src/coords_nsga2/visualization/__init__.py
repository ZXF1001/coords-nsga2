class Plotting:
    def __init__(self, optimizer_instance):
        self.optimizer = optimizer_instance

    def pareto_front(self, obj_indices, figsize=(8, 6), is_show=True):
        from .pareto_front import plot_pareto_front
        return plot_pareto_front(self.optimizer, obj_indices, figsize, is_show)
    
    def optimal_coords(self, obj_indices, figsize=(8, 6), is_show=True):
        from .optimal_coords import plot_optimal_coords
        return plot_optimal_coords(self.optimizer, obj_indices, figsize, is_show)
