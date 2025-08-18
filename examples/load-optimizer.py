import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2


# 下面的三个函数还是得手动重新定义一下
# 定义目标函数1：更靠近右上方
def objective_1(coords):
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

# 定义目标函数2：布局更分散
def objective_2(coords):
    return np.std(coords[:, 0]) + np.std(coords[:, 1])


spacing = 0.05  # 间距限制

def constraint_1(coords):
    dist_list = distance.pdist(coords)
    penalty_list = spacing-dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

loaded_optimizer = CoordsNSGA2.load("examples/data/test_optimizer.pkl")
loaded_optimizer.run(200, verbose=True)
loaded_optimizer.plot.pareto_front([0,1])