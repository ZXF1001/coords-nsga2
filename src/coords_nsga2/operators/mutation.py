import numpy as np

from coords_nsga2.spatial import create_points_in_polygon


def coords_mutation(population, prob_mut, region):
    """Coordinate mutation operator that mutates individual coordinates within region.

    Args:
        population: numpy array of shape (n_individuals, n_points, 2)
        prob_mut: mutation probability for each coordinate (-1 is auto set as 1/N_points)
        region: region defining valid regions

    Returns:
        Mutated population array
    """
    if prob_mut == -1:
        prob_mut = 1/population.shape[1]
    # Generate mutation mask
    mutation_mask = np.random.random(population.shape[:-1]) < prob_mut

    # Count mutations needed
    n_mutations = np.sum(mutation_mask)

    if n_mutations > 0:
        # Generate all new points at once
        new_points = create_points_in_polygon(region, n_mutations)

        # Apply mutations using mask
        population[mutation_mask] = new_points

    return population


def variable_mutation(population, prob_mut, region, n_points_min, n_points_max):
    if prob_mut == -1:
        # If prob_mut is -1, auto-set it as 1/average_n_points
        total_points = sum(len(ind) for ind in population)
        avg_points = total_points / \
            len(population) if len(population) > 0 else 1
        prob_mut = 1 / avg_points

    new_population = []
    for ind in population:
        # ind为shape=(n_points, 2)的点集，不同ind的n_points不同。
        # 现在逻辑为：对ind中的每个point，判断是否突变
        # 如果突变，有1/3概率点消失，有1/3概率点数不变只是换个位置，有1/3概率换位置同时新增一个点。
        current_n_points = len(ind)
        temp_ind_points = []
        for point in ind:
            random_status = np.random.random()
            if random_status < 1/3*prob_mut:
                if current_n_points > n_points_min:
                    # 点数多余最小点数时才触发消失
                    current_n_points -= 1
                else:
                    # 如果点数已经最少，不能再消失，那就保留原来的
                    temp_ind_points.append(point)
            elif random_status < 2/3*prob_mut:
                # 触发不变
                new_point = create_points_in_polygon(region, 1)
                temp_ind_points.append(new_point[0])
            elif random_status < prob_mut:
                if current_n_points < n_points_max:
                    # 点数少于最大点数时触发新增
                    current_n_points += 1
                    new_points = create_points_in_polygon(region, 2).tolist()
                    temp_ind_points += new_points
                else:
                    # 如果点数已经要超过了，就不多突变新的那个了
                    new_point = create_points_in_polygon(region, 1)
                    temp_ind_points.append(new_point[0])
            else:
                # 不突变
                temp_ind_points.append(point)
        new_population.append(np.array(temp_ind_points))
    return new_population


if __name__ == "__main__":
    from coords_nsga2.spatial import region_from_range
    np.random.seed(42)
    population_list = [
        np.array([[0.37454012, 0.95071431],
                  [0.73199394, 0.59865848],
                  [0.15601864, 0.15599452],
                  [0.05808361, 0.86617615]]),
        np.array([[0.60111501, 0.70807258],
                  [0.02058449, 0.96990985]])
    ]
    res = variable_mutation(
        population_list, 1, region_from_range(0, 1, 0, 1), 1, 4)
    print(res)
