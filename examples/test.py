import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
from scipy.spatial import distance
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from coords_nsga2.spatial import region_from_points
from coords_nsga2 import coords_nsga2
# 创建边界
polygon = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])
multi_polygon = MultiPolygon([polygon])

# 定义目标函数1
def objective_1(coords):
    return np.mean(coords[:, 0])

# 定义目标函数2
def objective_2(coords):
    return np.mean(coords[:, 1])

spacing = 0.1  # 间距限制
def constraint_1(coords):
    dist_list = distance.pdist(coords)
    penalty_list = spacing-dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

def constraint_2(coords): # x平均值不超过1
    return np.mean(coords[:, 0]) - 1

optimizer = coords_nsga2(func1=objective_1,
                         func2=objective_2,
                         pop_size=20,
                         n_points=5,
                         prob_crs=0.9,
                         prob_mut=0.1,
                         polygons=multi_polygon,
                         constraints=[constraint_1, constraint_2],
                         random_seed=10,
                         is_int=False)
result = optimizer.run(1000)

# 绘制结果
plt.figure(figsize=(10, 6))
for i in range(result.shape[0]):
    plt.scatter(result[i, :, 0], result[i, :, 1], label=f'Generation {i+1}')

# 绘制多边形边界
x, y = polygon.exterior.xy
plt.fill(x, y, alpha=0.2, fc='gray', ec='black')
plt.show()
