import matplotlib.pyplot as plt


def plot_pareto_front(optimizer, obj_indices, figsize, is_show):
    if len(obj_indices) == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(optimizer.values_P[obj_indices[0]], optimizer.values_P[obj_indices[1]],
                   alpha=0.7, edgecolors='black')
        ax.set_xlabel(f'Objective {obj_indices[0]+1}')
        ax.set_ylabel(f'Objective {obj_indices[1]+1}')
        ax.set_title('2D Pareto Front')
        ax.grid(True, alpha=0.3)

    elif len(obj_indices) == 3:
        # 3D Pareto front
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(optimizer.values_P[obj_indices[0]], optimizer.values_P[obj_indices[1]],
                   optimizer.values_P[obj_indices[2]], alpha=0.7,
                   edgecolors='black')
        ax.set_xlabel(f'Objective {obj_indices[0]}')
        ax.set_ylabel(f'Objective {obj_indices[1]}')
        ax.set_zlabel(f'Objective {obj_indices[2]}')
        ax.set_title('3D Pareto Front')

    else:
        raise ValueError("Can only plot 2D or 3D Pareto fronts")

    if is_show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import numpy as np
    test_data = np.load("examples/data/test.npz")

    class Optimizer:
        def __init__(self, data):
            self.values_P = data["values_P"]

    optimizer = Optimizer(test_data)
    plot_pareto_front(optimizer, [0, 1], (8, 6), True)
    plot_pareto_front(optimizer, [0, 1, 2], (8, 6), True)