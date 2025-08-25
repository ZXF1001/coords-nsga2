# 自己开发的针对风力机坐标点位布局用的NSGA-II算法
import pickle

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from .operators.crossover import coords_crossover, region_crossover
from .operators.mutation import coords_mutation, variable_mutation
from .operators.selection import coords_selection
from .spatial import create_points_in_polygon
from .utils import crowding_distance, fast_non_dominated_sort
from .visualization import Plotting


class Problem:
    def __init__(self, objectives, n_points, region, constraints=[], penalty_weight=1e6):
        self.objectives = objectives
        self.n_points = n_points
        self.region = region
        self.constraints = constraints
        self.penalty_weight = penalty_weight  # 可改为自适应

        if isinstance(self.n_points, int):
            self.variable_n_points = False
        else:
            assert len(self.n_points) == 2
            self.variable_n_points = True
            self.n_points_min = self.n_points[0]
            self.n_points_max = self.n_points[1]

    def sample_population(self, pop_size):
        if self.variable_n_points:
            # 可变坐标点数的代码
            n_points_list = np.random.randint(
                self.n_points_min, self.n_points_max+1, pop_size)
            coords = [create_points_in_polygon(
                self.region, n_p) for n_p in n_points_list]
            return coords
        else:
            coords = create_points_in_polygon(
                self.region, pop_size * self.n_points)
            return coords.reshape(pop_size, self.n_points, 2)

    def _evaluate_individual(self, individual):
        """
        计算单个个体的所有目标函数值，并根据约束进行惩罚调整。
        如果是列表形式的目标函数，则分别进行计算
        如果是单个多输出目标函数，则一次性计算
        惩罚函数同理
        """
        if isinstance(self.objectives, list):
            obj_values = np.array([obj(individual) for obj in self.objectives])
        else:
            obj_values = np.array(self.objectives(individual))
        # 如果设有约束，则计算约束惩罚
        if self.constraints:
            if isinstance(self.constraints, list):
                penalty = self.penalty_weight * \
                    np.sum([c(individual) for c in self.constraints])
            else:
                penalty = self.penalty_weight * \
                    np.sum(self.constraints(individual))
            obj_values -= penalty
        return obj_values

    def evaluate(self, population, n_jobs=1):
        """
        评估种群中每个个体的目标函数值

        参数:
            population: 种群，形状为 (pop_size, n_points, 2)
            n_jobs: 并行计算的作业数，默认为 1（串行计算），否则使用 joblib 并行

        返回:
            values: 目标函数值，形状为 (n_objectives, pop_size)
        """
        if n_jobs == 1:
            results = [self._evaluate_individual(ind) for ind in population]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_individual)(ind) for ind in population
            )

        # 组织结果为所需的形状 (n_objectives, pop_size)
        return np.array(results).T


class CoordsNSGA2:
    def __init__(self, problem, pop_size, prob_crs, prob_mut, random_seed=42, n_jobs=1):
        self.problem = problem
        self.variable_n_points = self.problem.variable_n_points
        self.pop_size = pop_size
        self.prob_crs = prob_crs
        self.prob_mut = prob_mut
        self.n_jobs = n_jobs  # 并行计算的作业数，默认为1（串行计算）

        np.random.seed(random_seed)
        assert pop_size % 2 == 0, "pop_size must be even number"
        self.P = self.problem.sample_population(pop_size)
        self.values_P = self.problem.evaluate(
            self.P, n_jobs=self.n_jobs)  # 并行评估
        self.P_history = [self.P]  # 记录每一代的解
        self.values_history = [self.values_P]  # 记录每一代的所有目标函数值

        # 初始化可视化模块
        self.plot = Plotting(self)

        self.selection = coords_selection  # 使用外部定义的selection函数
        if self.variable_n_points:
            self.n_points_min = problem.n_points_min
            self.n_points_max = problem.n_points_max
            self.crossover = region_crossover  # 使用外部定义的crossover函数
            self.mutation = variable_mutation  # 使用外部定义的mutation函数
        else:
            self.crossover = coords_crossover  # 使用外部定义的crossover函数
            self.mutation = coords_mutation  # 使用外部定义的mutation函数

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
        return [R[i] for i in new_idx] if self.variable_n_points else R[new_idx]

    def run(self, gen=1000, verbose=True):
        if verbose:
            iterator = trange(gen)
        else:
            iterator = range(gen)

        for _ in iterator:
            Q = self.selection(self.P, self.values_P)  # 选择
            if self.variable_n_points:
                Q = self.crossover(
                    Q, self.prob_crs, self.n_points_min, self.n_points_max)  # 交叉
                Q = self.mutation(
                    Q, self.prob_mut, self.problem.region, self.n_points_min, self.n_points_max)
                assert np.min([len(q) for q in Q]) >= self.n_points_min \
                    and np.max([len(q) for q in Q]) <= self.n_points_max
            else:
                Q = self.crossover(Q, self.prob_crs)  # 交叉
                Q = self.mutation(Q, self.prob_mut, self.problem.region)  # 变异

            values_Q = self.problem.evaluate(Q, n_jobs=self.n_jobs)  # 并行评估

            # 合并为R=(P,Q)
            R = self.P + Q if self.variable_n_points \
                else np.concatenate([self.P, Q])
            values_R = np.concatenate([self.values_P, values_Q], axis=1)

            # 快速非支配排序
            population_sorted_in_fronts = fast_non_dominated_sort(values_R)
            crowding_distances = [crowding_distance(
                values_R[:, front]) for front in population_sorted_in_fronts]

            # 选择下一代种群
            self.P = self.get_next_population(R,
                                              population_sorted_in_fronts, crowding_distances)

            self.values_P = self.problem.evaluate(
                self.P, n_jobs=self.n_jobs)  # 并行评估

            self.P_history.append(self.P)  # 这里后面改成全流程使用np数组
            self.values_history.append(self.values_P)
            # todo: 排序后再输出
        return self.P

    def save(self, filepath):
        print("请确保优化器所用到的外部函数（objective和constraint）都是自我封闭的，如果用到了外部的全局变量，可能会导致后续load之后无法读取！")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"CoordsNSGA2 instance successfully saved to {filepath}")
        except Exception as e:
            print("Warning: Failed to save CoordsNSGA2 instance. This may be due to unpickleable objects (e.g., lambda functions or nested functions).")
            print(f"Error details: {e}")
            raise

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        print(f"CoordsNSGA2 instance successfully loaded from {filepath}")
        return instance
