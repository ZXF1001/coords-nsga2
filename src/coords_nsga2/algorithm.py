# 自己开发的针对风力机坐标点位布局用的NSGA-II算法
import numpy as np
from tqdm import trange

from .operators.crossover import coords_crossover
from .operators.mutation import coords_mutation
from .operators.selection import coords_selection
from .spatial import create_point_in_polygon
from .utils import crowding_distance, fast_non_dominated_sort


class CoordsNSGA2:
    def __init__(self, func1, func2, pop_size, n_points, prob_crs, prob_mut, polygons, constraints=[], random_seed=0, is_int=False):
        self.func1 = func1
        self.func2 = func2
        self.pop_size = pop_size
        self.n_points = n_points  # 风力机的台数
        self.prob_crs = prob_crs
        self.prob_mut = prob_mut
        self.polygons = polygons  # 布点区域list，每个元素为一个多边形的坐标list
        self.constraints = constraints  # 约束条件list，每个元素为一个约束函数，返回值为0表示满足约束，否则返回惩罚值
        self.is_int = is_int

        np.random.seed(random_seed)
        assert pop_size % 2 == 0, "种群数量必须为偶数"
        self.P = self.init_population()  # 解种群
        self.values1_P, self.values2_P = self.evaluation(self.P)  # 评估
        self.P_history = [self.P]  # 记录每一代的解
        self.values1_history = [self.values1_P]  # 记录每一代的最前沿解的第一个目标函数值
        self.values2_history = [self.values2_P]  # 记录每一代的最前沿解的第一个目标函数值

        self.crossover = coords_crossover  # 使用外部定义的crossover函数
        self.mutation = coords_mutation  # 使用外部定义的mutation函数
        self.selection = coords_selection  # 使用外部定义的selection函数

    def init_population(self):
        # 批量生成坐标
        coords = np.array([create_point_in_polygon(self.polygons, self.is_int) for _ in range(self.pop_size * self.n_points)])
        # 重塑为目标形状
        init_pop = coords.reshape(self.pop_size, self.n_points, 2)
        # 返回整型或浮点型结果
        return init_pop.astype(int) if self.is_int else init_pop

    def evaluation(self, population):
        # todo: 这里可以采用并行+缓存的方式加速（已经实现，但是是在具体应用时重写这个函数通过joblib库来实现的，后续可以改写成装饰器的形式）

        func1_values = np.array([self.func1(x) for x in population])
        func2_values = np.array([self.func2(x) for x in population])
        # 评估约束惩罚函数
        if len(self.constraints) > 0:
            #todo: 这个1e6要自适应
            constraints_penalty = 1e6 * \
                np.array([np.sum([con(x) for con in self.constraints])
                         for x in population])
            func1_values -= constraints_penalty
            func2_values -= constraints_penalty
        return func1_values, func2_values

    def get_next_population(self,
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
        return new_idx

    def run(self, gen=1000):
        for _ in trange(gen):
            Q = self.selection(self.P, self.values1_P, self.values2_P)  # 选择
            Q = self.crossover(Q, self.prob_crs)  # 交叉
            Q = self.mutation(Q, self.prob_mut, self.polygons, self.is_int)  # 变异

            values1_Q, values2_Q = self.evaluation(Q)  # 评估
            R = np.concatenate([self.P, Q])  # 合并为R=(P,Q)
            values1_R = np.concatenate([self.values1_P, values1_Q])
            values2_R = np.concatenate([self.values2_P, values2_Q])

            # 快速非支配排序
            population_sorted_in_fronts = fast_non_dominated_sort(
                values1_R, values2_R)
            crowding_distances = [crowding_distance(
                values1_R, values2_R, front) for front in population_sorted_in_fronts]

            # 选择下一代种群
            R_idx = self.get_next_population(
                population_sorted_in_fronts, crowding_distances)
            self.P = R[R_idx]

            self.values1_P, self.values2_P = self.evaluation(self.P)  # 评估

            self.P_history.append(self.P) # 这里后面改成全流程使用np数组
            self.values1_history.append(self.values1_P)
            self.values2_history.append(self.values2_P)
            #todo: 排序后再输出
        return self.P

    def save(self, path):
        # 将self.P, self.values1_P, self.values2_P, self.P_history, self.values1_history, self.values2_history保存到path
        np.savez(path, P=self.P, values1_P=self.values1_P, values2_P=self.values2_P, P_history=self.P_history,
                 values1_history=self.values1_history, values2_history=self.values2_history)

    def load(self, path):
        # 从path中加载self.P, self.values1_P, self.values2_P, self.P_history, self.values1_history, self.values2_history
        data = np.load(path)
        self.P = data['P']
        self.values1_P = data['values1_P']
        self.values2_P = data['values2_P']
        self.P_history = data['P_history'].tolist()
        self.values1_history = data['values1_history'].tolist()
        self.values2_history = data['values2_history'].tolist()
        print(f'Loaded generation {len(self.P_history)} successfully!')
