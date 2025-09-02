import matplotlib.pyplot as plt


def plot_pareto_front(optimizer, obj_indices, generation=-1, figsize=None, is_show=True):
    # 根据generation参数选择数据源
    # 允许负数索引，例如-1表示最新一代
    if abs(generation) >= len(optimizer.values_history):
        raise ValueError(
            f"Generation {generation} is out of bounds. Must be between {-len(optimizer.values_history)} and {len(optimizer.values_history) - 1}.")

    values_to_plot = optimizer.values_history[generation]

    if len(obj_indices) == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(values_to_plot[obj_indices[0]], values_to_plot[obj_indices[1]],
                   alpha=0.7, edgecolors='black')
        ax.set_xlabel(f'Objective {obj_indices[0]+1}')
        ax.set_ylabel(f'Objective {obj_indices[1]+1}')
        generation_label = generation if generation >= 0 \
            else len(optimizer.P_history) + generation
        ax.set_title(f'2D Pareto Front (Generation: {generation_label})')
        ax.grid(True, alpha=0.3)

    elif len(obj_indices) == 3:
        # 3D Pareto front
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(values_to_plot[obj_indices[0]], values_to_plot[obj_indices[1]],
                   values_to_plot[obj_indices[2]], alpha=0.7,
                   edgecolors='black')
        ax.set_xlabel(f'Objective {obj_indices[0]}')
        ax.set_ylabel(f'Objective {obj_indices[1]}')
        ax.set_zlabel(f'Objective {obj_indices[2]}')
        generation_label = generation if generation >= 0 \
            else len(optimizer.P_history) + generation
        ax.set_title(f'3D Pareto Front (Generation: {generation_label})')

    else:
        raise ValueError("Can only plot 2D or 3D Pareto fronts")

    if is_show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from coords_nsga2 import CoordsNSGA2

    # 这些是pickle读取时必要的，但是内容不重要
    objective_1 = objective_2 = objective_3 = objective_4 = constraint_spacing = None

    loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")

    plot_pareto_front(loaded_optimizer, [0, 1])
    plot_pareto_front(loaded_optimizer, [0, 1, 2])
