class Plotting:
    def __init__(self, optimizer_instance):
        self.optimizer = optimizer_instance
        
    def example(self, arg1, arg2):
        from .example import plot_example
        return plot_example(self.optimizer, arg1, arg2)
