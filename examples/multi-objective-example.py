import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from coords_nsga2 import CoordsNSGA2, Problem
from coords_nsga2.spatial import region_from_points

# 创建边界
region = region_from_points([
    [0, 0],
    [1, 0],
    [2, 1],
    [1, 1],
])

# 定义多个目标函数

def objective_1(coords):
    """目标函数1：更靠近右上方"""
    return np.sum(coords[:, 0]) + np.sum(coords[:, 1])

def objective_2(coords):
    """目标函数2：布局更分散"""
    return np.std(coords[:, 0]) + np.std(coords[:, 1])

def objective_3(coords):
    """目标函数3：更靠近中心"""
    center = np.array([1.0, 0.5])  # 区域中心
    distances = np.linalg.norm(coords - center, axis=1)
    return -np.mean(distances)  # 负号表示最小化距离

def objective_4(coords):
    """目标函数4：更靠近边界"""
    # 计算到边界的平均距离
    boundary_distances = []
    for point in coords:
        # 简化的边界距离计算
        min_dist = min(point[0], point[1], 2-point[0], 1-point[1])
        boundary_distances.append(min_dist)
    return -np.mean(boundary_distances)  # 负号表示最小化距离

# 约束函数
spacing = 0.05  # 间距限制

def constraint_1(coords):
    """约束：点之间保持最小距离"""
    if len(coords) < 2:
        return 0
    dist_list = distance.pdist(coords)
    penalty_list = spacing - dist_list[dist_list < spacing]
    penalty_sum = np.sum(penalty_list)
    return penalty_sum

# 创建多目标优化问题
problem = Problem(
    objectives=[objective_1, objective_2, objective_3, objective_4],
    n_points=10,
    region=region,
    constraints=[constraint_1]
)

# 创建优化器
optimizer = CoordsNSGA2(
    problem=problem,
    pop_size=50,  # 增加种群大小以更好地处理多目标
    prob_crs=0.5,
    prob_mut=0.1
)

# 运行优化
result = optimizer.run(500)

# 获取最终种群的目标函数值
final_values = optimizer.values_P

# 绘制结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('多目标优化结果 (4个目标函数)', fontsize=16)

# 绘制Pareto前沿（选择前3个目标函数进行3D可视化）
ax1 = axes[0, 0]
ax1.scatter(final_values[0], final_values[1], alpha=0.6)
ax1.set_xlabel('目标函数1: 靠近右上方')
ax1.set_ylabel('目标函数2: 布局分散')
ax1.set_title('目标1 vs 目标2')

ax2 = axes[0, 1]
ax2.scatter(final_values[0], final_values[2], alpha=0.6)
ax2.set_xlabel('目标函数1: 靠近右上方')
ax2.set_ylabel('目标函数3: 靠近中心')
ax2.set_title('目标1 vs 目标3')

ax3 = axes[0, 2]
ax3.scatter(final_values[1], final_values[2], alpha=0.6)
ax3.set_xlabel('目标函数2: 布局分散')
ax3.set_ylabel('目标函数3: 靠近中心')
ax3.set_title('目标2 vs 目标3')

# 绘制坐标布局
ax4 = axes[1, 0]
# 选择Pareto前沿上的几个解进行可视化
front_0 = optimizer.values_history[-1]  # 最后一代
# 简单选择几个解进行展示
for i in range(min(5, len(result))):
    ax4.scatter(result[i, :, 0], result[i, :, 1], alpha=0.7, label=f'解 {i+1}')
ax4.set_xlabel('X坐标')
ax4.set_ylabel('Y坐标')
ax4.set_title('坐标布局')
ax4.legend()

# 绘制目标函数4的分布
ax5 = axes[1, 1]
ax5.scatter(final_values[0], final_values[3], alpha=0.6)
ax5.set_xlabel('目标函数1: 靠近右上方')
ax5.set_ylabel('目标函数4: 靠近边界')
ax5.set_title('目标1 vs 目标4')

# 绘制所有目标函数的分布
ax6 = axes[1, 2]
ax6.boxplot([final_values[i] for i in range(4)], labels=['目标1', '目标2', '目标3', '目标4'])
ax6.set_ylabel('目标函数值')
ax6.set_title('所有目标函数值分布')

plt.tight_layout()
plt.show()

# 打印一些统计信息
print("优化完成！")
print(f"最终种群大小: {len(result)}")
print(f"目标函数数量: {len(final_values)}")
print("每个目标函数的统计信息:")
for i, values in enumerate(final_values):
    print(f"  目标{i+1}: 最小值={values.min():.4f}, 最大值={values.max():.4f}, 平均值={values.mean():.4f}")
