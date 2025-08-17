def plot_example(optimizer, arg1, arg2):
    """
    示例可视化函数，展示如何使用可视化方法。
    
    参数:
        plotting_instance: Plotting实例，通过它访问CoordsNSGA2实例
        arg1: 第一个参数
        arg2: 第二个参数
    """
    
    # 假设我们从优化器中获取一些数据进行可视化
    # 注意：这里使用optimizer.P和optimizer.values_P作为示例数据
    # 实际应用中可能需要根据具体的可视化需求获取不同的数据
    population = optimizer.P_history[-1] # 获取最后一代的种群
    values = optimizer.values_history[-1] # 获取最后一代的目标函数值

    print(f"Plotting example with arg1={arg1}, arg2={arg2}")
    print(f"Population shape: {population.shape}")
    print(f"Values shape: {values.shape}")