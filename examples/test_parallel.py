import multiprocessing
import numpy as np

def evaluation_individual(individual):
    func1 = individual**2
    func2 = individual**3
    return func1, func2

def parallel_evaluation(population):
    with multiprocessing.Pool() as pool:
        results = pool.map(evaluation_individual, population)

    func1_values, func2_values = zip(*results)
    func1_values = list(func1_values)
    func2_values = list(func2_values)
    return func1_values, func2_values

if __name__ == "__main__":
    population = np.random.rand(100, 1)
    func1_values, func2_values = parallel_evaluation(population)

    print(func1_values)
    print(func2_values)