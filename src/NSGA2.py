# 自己开发的针对风力机坐标点位布局用的NSGA-II算法
import numpy as np
from tqdm import trange
from shapely.geometry import Point

class NSGA2():
    # 这个是一开始用来学习NSGA-II的代码时候写的，后来开始开发风电场布局优化的时候就不用这个了，因此有些功能没有实现
    def __init__(self, func1, func2, pop_size, max_gen, prob_cross, prob_mut, min_x, max_x, random_seed=0):
        self.func1 = func1
        self.func2 = func2
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.min_x = min_x
        self.max_x = max_x
        np.random.seed(random_seed)
        assert pop_size % 2 == 0, "种群数量必须为偶数"

        self.P = self.init_population()  # 解种群
        self.values1_P, self.values2_P = self.evaluation(self.P)  # 评估

    def init_population(self):
        return np.random.uniform(self.min_x, self.max_x, self.pop_size)

    def evaluation(self, population):
        # todo: 这里可以采用并行+缓存的方式加速
        func1_values = np.array([self.func1(x) for x in population])
        func2_values = np.array([self.func2(x) for x in population])
        return func1_values, func2_values

    def selection(self, tourn_size=3):
        # 锦标赛选择，选择的依据是快速非支配排序的结果和拥挤度
        # 1. 先把所有的解进行快速非支配排序和拥挤度计算
        population_sorted_in_fronts = self.fast_non_dominated_sort(
            self.values1_P, self.values2_P)
        crowding_distances = [self.crowding_distance(
            self.values1_P, self.values2_P, front) for front in population_sorted_in_fronts]
        # 将这两个结果组成一个列表，方便后续比较：第一列为index，第二列为前沿等级，第三列为拥挤度
        compare_table = []
        for i, front in enumerate(population_sorted_in_fronts):
            for j, idx in enumerate(front):
                compare_table.append([idx, i, crowding_distances[i][j]])
        # 按照index排序
        compare_table = np.array(sorted(compare_table, key=lambda x: x[0]))

        # 2. 生成[0,self.pop_size)的随机数，形状为(self.pop_size, tourn_size)
        aspirants_idx = np.random.randint(
            self.pop_size, size=(self.pop_size, tourn_size))

        # 3. 选择每组中最前沿（前沿等级相同时选择拥挤度最高）的解
        array = compare_table[aspirants_idx]
        sorted_indices = np.lexsort((-array[..., 2], array[..., 1]))
        Q_idx = aspirants_idx[np.arange(self.pop_size), sorted_indices[:, 0]]
        return self.P[Q_idx]

    def crossover(self, population):
        # 先写一个简单的交叉算子，定义为一个是(a+b)/2，另一个是(a-b)/2
        for i in range(0, len(population), 2):
            if np.random.rand() < self.prob_cross:
                temp1 = (population[i] + population[i+1]) / 2
                temp2 = (population[i] - population[i+1]) / 2
                population[i] = temp1
                population[i+1] = temp2
        return population

    def mutation(self, population):
        for i in range(len(population)):
            rand = np.random.rand()
            if rand < self.prob_mut:
                population[i] = np.random.uniform(self.min_x, self.max_x)
        return population

    def fast_non_dominated_sort(self, values1, values2):
        """
        输入：values1, values2 为两个目标函数的值列表；
        输出：返回一个列表，列表中的每个元素是一个列表，表示一个前沿
        """
        # 初始化数据结构
        num_population = len(values1)
        dominated_solutions = [[] for _ in range(num_population)]
        domination_count = [0] * num_population
        ranks = [0] * num_population
        fronts = [[]]

        # 确定支配关系
        for p in range(num_population):
            for q in range(num_population):
                # p 支配 q：p 在所有目标上都不差于 q，且至少在一个目标上优于 q
                if (values1[p] > values1[q] and values2[p] >= values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]):
                    dominated_solutions[p].append(q)
                elif (values1[q] > values1[p] and values2[q] >= values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]):
                    domination_count[p] += 1

            # 如果没有解支配 p，则 p 属于第一个前沿
            if domination_count[p] == 0:
                fronts[0].append(p)

        # 按前沿层次进行排序
        current_rank = 0
        while fronts[current_rank]:
            next_front = []
            for p in fronts[current_rank]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = current_rank + 1
                        next_front.append(q)
            current_rank += 1
            fronts.append(next_front)

        # 去掉最后一个空层
        fronts.pop()
        return fronts

    def sort_by_values(self, idx_lst, values_lst):
        # 根据values对list进行排序
        needed_values = [values_lst[i] for i in idx_lst]
        sorted_list = [x for _, x in sorted(zip(needed_values, idx_lst))]
        return sorted_list

    def crowding_distance(self, values1, values2, front):
        """
        输入：values1, values2 为两个目标函数的值列表；front 为一个前沿中的解的索引列表
        输出：返回一个列表，列表中的每个元素是一个解的拥挤距离
        """
        # 初始化拥挤距离
        distance = [0] * len(front)

        # 对每个目标进行排序（根据后面的值给前面的值排序）
        sorted1 = self.sort_by_values(front, values1)
        sorted2 = self.sort_by_values(front, values2)
        # 计算每个解的拥挤距离
        min1, max1 = min(values1), max(values1)
        min2, max2 = min(values2), max(values2)

        for i, ind in enumerate(front):
            idx_in_sorted1 = sorted1.index(ind)
            if idx_in_sorted1 == 0 or idx_in_sorted1 == len(front) - 1:
                distance[i] = float('inf')
            else:
                distance[i] += (values1[sorted1[idx_in_sorted1 + 1]] -
                                values1[sorted1[idx_in_sorted1 - 1]]) / (max1 - min1)

            idx_in_sorted2 = sorted2.index(ind)
            if idx_in_sorted2 == 0 or idx_in_sorted2 == len(front) - 1:
                distance[i] = float('inf')
            else:
                distance[i] += (values2[sorted2[idx_in_sorted2 + 1]] -
                                values2[sorted2[idx_in_sorted2 - 1]]) / (max2 - min2)

        return distance

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
            # 先尽可能吧每个靠前的前沿加进来
            if len(new_idx) + len(front) < self.pop_size:
                new_idx.extend(front)
            elif len(new_idx) + len(front) == self.pop_size:
                new_idx.extend(front)
                break
            else:
                # 如果加上这个前沿后超过pop_size，则按照拥挤度排序，选择拥挤度大的解
                # 先按照拥挤度从大到小，对索引进行排序
                sorted_combined = sorted(
                    zip(crowding_distances[i], front), reverse=True)
                sorted_front = [item for _, item in sorted_combined]
                # 选择前pop_size-len(new_idx)个解
                new_idx.extend(sorted_front[:self.pop_size - len(new_idx)])
                break
        return new_idx

    def run(self, gen=None):
        self.max_gen = gen or self.max_gen
        for gen in trange(self.max_gen):
            Q = self.selection()  # 选择
            Q = self.crossover(Q)  # 交叉
            Q = self.mutation(Q)  # 变异

            values1_Q, values2_Q = self.evaluation(Q)  # 评估
            R = np.concatenate([self.P, Q])  # 合并为R=(P,Q)
            values1_R = np.concatenate([self.values1_P, values1_Q])
            values2_R = np.concatenate([self.values2_P, values2_Q])

            # 快速非支配排序
            population_sorted_in_fronts = self.fast_non_dominated_sort(
                values1_R, values2_R)
            crowding_distances = [self.crowding_distance(
                values1_R, values2_R, front) for front in population_sorted_in_fronts]

            # 选择下一代种群
            R_idx = self.get_next_population(
                population_sorted_in_fronts, crowding_distances)
            self.P = R[R_idx]

            self.values1_P, self.values2_P = self.evaluation(self.P)  # 评估

        return self.P, self.values1_P, self.values2_P


class NSGA2_XY():
    def __init__(self, func1, func2, pop_size, n_dim, prob_cross, prob_mut, polygons, constraints=[], random_seed=0, is_int=True):
        self.func1 = func1
        self.func2 = func2
        self.pop_size = pop_size
        self.n_dim = n_dim  # 风力机的台数
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.polygons = polygons # 布点区域list，每个元素为一个多边形的坐标list
        self.constraints = constraints # 约束条件list，每个元素为一个约束函数，返回值为0表示满足约束，否则返回惩罚值
        self.is_int = is_int

        np.random.seed(random_seed)
        assert pop_size % 2 == 0, "种群数量必须为偶数"
        self.P = self.init_population()  # 解种群
        self.values1_P, self.values2_P = self.evaluation(self.P)  # 评估
        self.P_history = [self.P]  # 记录每一代的解
        self.values1_history = [self.values1_P]  # 记录每一代的最前沿解的第一个目标函数值
        self.values2_history = [self.values2_P]  # 记录每一代的最前沿解的第一个目标函数值

    def create_point_in_polygon(self):
        minx, miny, maxx, maxy = self.polygons.bounds
        while True:
            if self.is_int:
                x = np.random.randint(minx, maxx)
                y = np.random.randint(miny, maxy)
            else:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
            if self.polygons.contains(Point(x, y)):
                return x, y
            
    def init_population(self):
        # 坐标的初始化
        xy = []
        for _ in range(self.pop_size*self.n_dim):
            x, y = self.create_point_in_polygon()
            xy.append([x, y])
        init_pop = np.array(xy).reshape(self.pop_size, self.n_dim, 2)
        if self.is_int:
            return init_pop.astype(int)
        else:
            return init_pop
            
    def evaluation(self, population):
        # todo: 这里可以采用并行+缓存的方式加速（已经实现，但是是在具体应用时重写这个函数通过joblib库来实现的）
        # 另外，这里为了减小计算量避免在做布局优化的过程中把风场在两个目标函数中计算两遍，在具体应用中重写了这个self.evalution()函数
        ## 评估目标函数值
        ## 并行测试
        # results = Parallel(n_jobs=4)(
        #     delayed(self.func1)(x) for x in population
        # )

        func1_values = np.array([self.func1(x) for x in population])
        func2_values = np.array([self.func2(x) for x in population])
        ## 评估约束惩罚函数
        if len(self.constraints)>0:
            constraints_penalty = 1e6 * np.array([np.sum([con(x) for con in self.constraints]) for x in population])
            func1_values -= constraints_penalty
            func2_values -= constraints_penalty
        return func1_values, func2_values

    def selection(self, tourn_size=3):
        # 锦标赛选择，选择的依据是快速非支配排序的结果和拥挤度
        # 1. 先把所有的解进行快速非支配排序和拥挤度计算
        population_sorted_in_fronts = self.fast_non_dominated_sort(
            self.values1_P, self.values2_P)
        crowding_distances = [self.crowding_distance(
            self.values1_P, self.values2_P, front) for front in population_sorted_in_fronts]
        # 将这两个结果组成一个列表，方便后续比较：第一列为index，第二列为前沿等级，第三列为拥挤度
        compare_table = []
        for i, front in enumerate(population_sorted_in_fronts):
            for j, idx in enumerate(front):
                compare_table.append([idx, i, crowding_distances[i][j]])
        # 按照index排序
        compare_table = np.array(sorted(compare_table, key=lambda x: x[0]))

        # 2. 生成[0,self.pop_size)的随机数，形状为(self.pop_size, tourn_size)
        aspirants_idx = np.random.randint(
            self.pop_size, size=(self.pop_size, tourn_size))

        # 3. 选择每组中最前沿（前沿等级相同时选择拥挤度最高）的解
        array = compare_table[aspirants_idx]
        sorted_indices = np.lexsort((-array[..., 2], array[..., 1]))
        Q_idx = aspirants_idx[np.arange(self.pop_size), sorted_indices[:, 0]]
        return self.P[Q_idx]

    def crossover(self, population):
        # 坐标的交叉算子
        for i in range(0, len(population), 2):
            if np.random.rand() < self.prob_cross:
                cross_num = np.random.randint(1, self.n_dim)
                cross_idx = np.random.choice(self.n_dim, cross_num, replace=False)
                temp = population[i][cross_idx]
                population[i][cross_idx] = population[i+1][cross_idx]
                population[i+1][cross_idx] = temp
        return population

    def mutation(self, population):
        # 坐标的变异算子，这里做小修改没有把整个个体变异，而是对每个坐标进行变异，因为这样才可以逐渐产生满足约束的解，如果对整个个体变异大概率会违反约束
        for i in range(len(population)):
            for j in range(self.n_dim):
                if np.random.rand() < self.prob_mut:
                    x, y = self.create_point_in_polygon()
                    population[i][j] = np.array([x, y])
        return population
        
    def fast_non_dominated_sort(self, values1, values2):
        """
        输入：values1, values2 为两个目标函数的值列表；
        输出：返回一个列表，列表中的每个元素是一个列表，表示一个前沿
        """
        # 初始化数据结构
        num_population = len(values1)
        dominated_solutions = [[] for _ in range(num_population)]
        domination_count = [0] * num_population
        ranks = [0] * num_population
        fronts = [[]]

        # 确定支配关系
        for p in range(num_population):
            for q in range(num_population):
                # p 支配 q：p 在所有目标上都不差于 q，且至少在一个目标上优于 q
                if (values1[p] > values1[q] and values2[p] >= values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]):
                    dominated_solutions[p].append(q)
                elif (values1[q] > values1[p] and values2[q] >= values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]):
                    domination_count[p] += 1

            # 如果没有解支配 p，则 p 属于第一个前沿
            if domination_count[p] == 0:
                fronts[0].append(p)

        # 按前沿层次进行排序
        current_rank = 0
        while fronts[current_rank]:
            next_front = []
            for p in fronts[current_rank]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = current_rank + 1
                        next_front.append(q)
            current_rank += 1
            fronts.append(next_front)

        # 去掉最后一个空层
        fronts.pop()
        return fronts

    def sort_by_values(self, idx_lst, values_lst):
        # 根据values对list进行排序
        needed_values = [values_lst[i] for i in idx_lst]
        sorted_list = [x for _, x in sorted(zip(needed_values, idx_lst))]
        return sorted_list

    def crowding_distance(self, values1, values2, front):
        """
        输入：values1, values2 为两个目标函数的值列表；front 为一个前沿中的解的索引列表
        输出：返回一个列表，列表中的每个元素是一个解的拥挤距离
        """
        # 初始化拥挤距离
        distance = [0] * len(front)

        # 对每个目标进行排序（根据values的值给front里的index排序）
        sorted1 = self.sort_by_values(front, values1)
        sorted2 = sorted1[::-1]
        # 计算每个解的拥挤距离
        min1, max1 = min(values1), max(values1) #! 注意：这里min的对象是整个种群，因此min可能会受到惩罚函数较大系数的影响。最好的解决方案是惩罚函数乘一个适中的系数，避免把拥挤度距离小到抹去。
        min2, max2 = min(values2), max(values2)
        if (min1 == max1) and (min2 == max2):
            return distance

        for i, ind in enumerate(front):
            idx_in_sorted1 = sorted1.index(ind)
            if idx_in_sorted1 == 0 or idx_in_sorted1 == len(front) - 1:
                distance[i] = float('inf')
            elif values1[sorted1[idx_in_sorted1]] == values1[sorted1[idx_in_sorted1 - 1]]: #* 如果和上一个点重合，就是0
                distance[i] += 0
            else:
                distance[i] += (values1[sorted1[idx_in_sorted1 + 1]] -
                                values1[sorted1[idx_in_sorted1 - 1]]) / (max1 - min1)

            idx_in_sorted2 = sorted2.index(ind)
            if idx_in_sorted2 == 0 or idx_in_sorted2 == len(front) - 1:
                distance[i] = float('inf')
            elif values2[sorted2[idx_in_sorted2]] == values2[sorted2[idx_in_sorted2 + 1]]: #* 如果和**下**一个点重合，就是0
                distance[i] += 0
            else:
                distance[i] += (values2[sorted2[idx_in_sorted2 + 1]] -
                                values2[sorted2[idx_in_sorted2 - 1]]) / (max2 - min2)

        return distance

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
            # 先尽可能吧每个靠前的前沿加进来
            if len(new_idx) + len(front) < self.pop_size:
                new_idx.extend(front)
            elif len(new_idx) + len(front) == self.pop_size:
                new_idx.extend(front)
                break
            else:
                # 如果加上这个前沿后超过pop_size，则按照拥挤度排序，选择拥挤度大的解
                # 先按照拥挤度从大到小，对索引进行排序
                sorted_combined = sorted(
                    zip(crowding_distances[i], front), reverse=True)
                sorted_front = [item for _, item in sorted_combined]
                # 选择前pop_size-len(new_idx)个解
                new_idx.extend(sorted_front[:self.pop_size - len(new_idx)])
                break
        return new_idx

    def run(self, gen=1000):
        for g in trange(gen):
            Q = self.selection()  # 选择
            Q = self.crossover(Q)  # 交叉
            Q = self.mutation(Q)  # 变异

            values1_Q, values2_Q = self.evaluation(Q)  # 评估
            R = np.concatenate([self.P, Q])  # 合并为R=(P,Q)
            values1_R = np.concatenate([self.values1_P, values1_Q])
            values2_R = np.concatenate([self.values2_P, values2_Q])

            # 快速非支配排序
            population_sorted_in_fronts = self.fast_non_dominated_sort(
                values1_R, values2_R)
            crowding_distances = [self.crowding_distance(
                values1_R, values2_R, front) for front in population_sorted_in_fronts]
            #* 这里的拥挤度距离计算是正常的（只有两个inf，其他重合点都是0），但是因为找到既能在第一前沿上又能和其他解不一样的点太难了，所以最后选择下一代的时候大多数还是和前面的重合了

            # 选择下一代种群
            R_idx = self.get_next_population(
                population_sorted_in_fronts, crowding_distances)
            self.P = R[R_idx]

            self.values1_P, self.values2_P = self.evaluation(self.P)  # 评估
            
            self.P_history.append(self.P)
            self.values1_history.append(self.values1_P)
            self.values2_history.append(self.values2_P)

        return self.P

    def save(self, path):
        # 将self.P, self.values1_P, self.values2_P, self.P_history, self.values1_history, self.values2_history保存到path
        np.savez(path, P=self.P, values1_P=self.values1_P, values2_P=self.values2_P, P_history=self.P_history, values1_history=self.values1_history, values2_history=self.values2_history)

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