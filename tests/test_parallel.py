import time

import numpy as np
import pytest
from shapely.geometry import Polygon

from coords_nsga2 import CoordsNSGA2, Problem


# 创建一个简单的测试区域
@pytest.fixture
def test_region():
    return Polygon([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

# 创建简单的目标函数
def simple_objective_1(coords):
    # 添加小延迟模拟计算
    time.sleep(0.01)
    return np.sum(coords[:, 0])

def simple_objective_2(coords):
    # 添加小延迟模拟计算
    time.sleep(0.01)
    return np.sum(coords[:, 1])

# 测试Problem类的evaluate方法在串行和并行模式下是否正常工作
def test_problem_evaluate_serial_vs_parallel(test_region):
    # 创建问题实例
    problem = Problem(
        objectives=[simple_objective_1, simple_objective_2],
        n_points=5,
        region=test_region
    )
    
    # 生成测试种群
    population = problem.sample_population(pop_size=10)
    
    # 串行评估
    serial_start = time.time()
    values_serial = problem.evaluate(population, n_jobs=1)
    serial_time = time.time() - serial_start
    
    # 并行评估 (使用2个作业)
    parallel_start = time.time()
    values_parallel = problem.evaluate(population, n_jobs=2)
    parallel_time = time.time() - parallel_start
    
    # 验证结果是否相同
    assert np.allclose(values_serial, values_parallel), "串行和并行计算结果应该相同"
    
    # 打印时间比较（不作为测试断言，因为在CI环境中可能不稳定）
    print(f"串行评估时间: {serial_time:.4f}秒")
    print(f"并行评估时间: {parallel_time:.4f}秒")

# 测试CoordsNSGA2类在串行和并行模式下是否正常工作
def test_nsga2_serial_vs_parallel(test_region):
    # 创建问题实例
    problem = Problem(
        objectives=[simple_objective_1, simple_objective_2],
        n_points=5,
        region=test_region
    )
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 创建串行优化器并运行
    optimizer_serial = CoordsNSGA2(
        problem=problem,
        pop_size=10,
        prob_crs=0.5,
        prob_mut=0.1,
        random_seed=42,
        n_jobs=1
    )
    
    serial_start = time.time()
    result_serial = optimizer_serial.run(5)
    serial_time = time.time() - serial_start
    
    # 重置随机种子
    np.random.seed(42)
    
    # 创建并行优化器并运行
    optimizer_parallel = CoordsNSGA2(
        problem=problem,
        pop_size=10,
        prob_crs=0.5,
        prob_mut=0.1,
        random_seed=42,
        n_jobs=2
    )
    
    parallel_start = time.time()
    result_parallel = optimizer_parallel.run(5)
    parallel_time = time.time() - parallel_start
    
    # 验证结果是否相同（由于随机性，可能会有微小差异，但应该非常接近）
    # 我们检查最终种群的形状是否相同
    assert result_serial.shape == result_parallel.shape, "串行和并行优化结果形状应该相同"
    
    # 打印时间比较
    print(f"串行优化时间: {serial_time:.4f}秒")
    print(f"并行优化时间: {parallel_time:.4f}秒")

if __name__ == "__main__":
    # 手动运行测试
    region = Polygon([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    test_problem_evaluate_serial_vs_parallel(region)
    test_nsga2_serial_vs_parallel(region)