# 自己开发的针对风力机坐标点位布局用的NSGA-II算法
import numpy as np
from tqdm import trange

from .operators.crossover import coords_crossover
from .operators.mutation import coords_mutation
from .operators.selection import coords_selection
from .spatial import create_points_in_polygon
from .utils import crowding_distance, fast_non_dominated_sort


class Problem:
    def __init__(self, objectives, n_points, region, constraints=[], penalty_weight=1e6):
        self.objectives = objectives
        self.n_points = n_points
        self.region = region
        self.constraints = constraints
        self.penalty_weight = penalty_weight  # 可改为自适应

    def sample_population(self, pop_size):
        coords = create_points_in_polygon(
            self.region, pop_size * self.n_points)
        return coords.reshape(pop_size, self.n_points, 2)

    def evaluate(self, population):
        values = []
        for obj_func in self.objectives:
            values.append([obj_func(x) for x in population])
        values = np.array(values)
        if self.constraints:
            penalty = self.penalty_weight * \
                np.array([np.sum([c(x) for c in self.constraints])
                         for x in population])
            values -= penalty
        return values


class CoordsNSGA2:
    def __init__(self, problem, pop_size, prob_crs, prob_mut, random_seed=42):
        self.problem = problem
        self.pop_size = pop_size
        self.prob_crs = prob_crs
        self.prob_mut = prob_mut

        np.random.seed(random_seed)
        assert pop_size % 2 == 0, "pop_size must be even number"
        self.P = self.problem.sample_population(pop_size)
        self.values_P = self.problem.evaluate(self.P)  # 评估
        self.P_history = [self.P]  # 记录每一代的解
        self.values_history = [self.values_P]  # 记录每一代的所有目标函数值

        # todo: 这部分未来要放在optimizer的定义的参数中
        self.crossover = coords_crossover  # 使用外部定义的crossover函数
        self.mutation = coords_mutation  # 使用外部定义的mutation函数
        self.selection = coords_selection  # 使用外部定义的selection函数

    def get_next_population(self, R,
                            population_sorted_in_fronts,
                            crowding_distances):
        """
        通过前沿等级、拥挤度，选取前pop_size个解，作为下一代种群
        输入：
        population_sorted_in_fronts 为所有解快速非支配排序后按照前沿等级分组的解索引
        crowding_distances 为所有解快速非支配排序后按照前沿等级分组的拥挤距离数组
        输出：
        new_idx 为下一代种群的解的索引（也就是R的索引）
        """
        new_idx = []
        for i, front in enumerate(population_sorted_in_fronts):
            remaining_size = self.pop_size - len(new_idx)
            # 先尽可能吧每个靠前的前沿加进来
            if len(front) < remaining_size:
                new_idx.extend(front)
            elif len(front) == remaining_size:
                new_idx.extend(front)
                break
            else:
                # 如果加上这个前沿后超过pop_size，则按照拥挤度排序，选择拥挤度大的解
                # 先按照拥挤度从大到小，对索引进行排序
                crowding_dist = np.array(crowding_distances[i])
                sorted_front_idx = np.argsort(crowding_dist)[::-1]  # 从大到小排序
                sorted_front = np.array(front)[sorted_front_idx]
                new_idx.extend(sorted_front[:remaining_size])
                break
        return R[new_idx]

    def run(self, gen=1000, verbose=True):
        if verbose:
            iterator = trange(gen)
        else:
            iterator = range(gen)

        for _ in iterator:
            Q = self.selection(self.P, self.values_P)  # 选择
            Q = self.crossover(Q, self.prob_crs)  # 交叉
            Q = self.mutation(Q, self.prob_mut, self.problem.region)  # 变异

            values_Q = self.problem.evaluate(Q)  # 评估

            # 合并为R=(P,Q)
            R = np.concatenate([self.P, Q])
            values_R = np.concatenate([self.values_P, values_Q], axis=1)

            # 快速非支配排序
            population_sorted_in_fronts = fast_non_dominated_sort(values_R)
            crowding_distances = [crowding_distance(
                values_R[:, front]) for front in population_sorted_in_fronts]

            # 选择下一代种群
            self.P = self.get_next_population(R,
                                              population_sorted_in_fronts, crowding_distances)

            self.values_P = self.problem.evaluate(
                self.P)  # 评估

            self.P_history.append(self.P)  # 这里后面改成全流程使用np数组
            self.values_history.append(self.values_P)
            # todo: 排序后再输出
        return self.P

    def save(self, path):
        # 将self.P, self.values_P, self.P_history, self.values_history保存到path
        np.savez(path, P=self.P, values_P=self.values_P, P_history=self.P_history,
                 values_history=self.values_history)

    def load(self, path):
        # 从path中加载self.P, self.values_P, self.P_history, self.values_history
        data = np.load(path)
        self.P = data['P']
        self.values_P = data['values_P']
        self.P_history = data['P_history'].tolist()
        self.values_history = data['values_history'].tolist()
        print(f'Loaded generation {len(self.P_history)} successfully!')
