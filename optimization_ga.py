import random


# 定义目标函数（优化目标 G）
def optimization_goal(I, H, E, S):
    k1, k2, k3, k4 = 0.3, 0.3, 0.35, 0.05  # 权重分配
    G = k1 * I + k2 * H + k3 * E + k4 * S  # 计算优化目标
    return G


# 定义评价函数，根据参数计算优化目标
def fitness_function(params):
    x, y, z, w = params  # 解的参数：税率、景点维护费用、社区支出、旅游宣传费用

    # 假设的函数：根据决策变量计算状态变量（这是一个简化的示例，实际模型可能更复杂）
    # 这里假设优化目标与一些简单的线性函数关系
    P = 1.5  # 假设游客数量
    I = (x + 0.5) * P - y - z - w  # 净旅游收入
    H = 0.7 * I + 0.3 * (y + z) - 0.5 * P  # 居民幸福感
    E = 0.8 * (y - P) + 0.4 * (z - P)  # 环境质量
    S = P / 2  # 游客满意度

    # 计算目标函数值
    G = optimization_goal(I, H, E, S)
    return G


# 初始化种群
def initialize_population(population_size, bounds):
    population = []
    for _ in range(population_size):
        # 随机生成解，范围在 bounds 指定的上下限之间
        individual = [
            random.uniform(bounds[i][0], bounds[i][1])
            for i in range(len(bounds))
        ]
        population.append(individual)
    return population


# 选择操作（轮盘赌选择）
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_indices = random.choices(
        range(len(population)), weights=selection_probabilities, k=2
    )
    return population[selected_indices[0]], population[selected_indices[1]]


# 交叉操作（单点交叉）
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# 变异操作（高斯变异）
def mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        mutation_index = random.randint(0, len(individual) - 1)
        mutation_value = random.uniform(-0.1, 0.1)
        individual[mutation_index] += mutation_value
    return individual


# 遗传算法主函数
def genetic_algorithm(
    population_size, max_generations, crossover_rate, mutation_rate, bounds
):
    population = initialize_population(population_size, bounds)
    best_solution = None
    best_fitness = -float("inf")

    for _ in range(max_generations):
        fitness_values = [
            fitness_function(individual) for individual in population
        ]  # 评估种群的适应度

        # 找到当前代的最优解
        max_fitness_idx, max_fitness_value = max(
            enumerate(fitness_values), key=lambda item: item[1]
        )
        if max_fitness_value > best_fitness:
            best_fitness = max_fitness_value
            best_solution = population[max_fitness_idx]

        # 创建新种群
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness_values)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population  # 更新种群

    return best_solution, best_fitness


def main():
    # 参数设置
    population_size = 50  # 种群大小
    max_generations = 100  # 最大迭代代数
    crossover_rate = 0.7  # 交叉率
    mutation_rate = 0.1  # 变异率
    bounds = [
        (0.1, 0.2),
        (10, 30),
        (5, 25),
        (0.1, 1),
    ]  # 参数上下限（税率、景点维护、社区支出、旅游宣传）

    # 运行遗传算法
    best_solution, best_fitness = genetic_algorithm(
        population_size, max_generations, crossover_rate, mutation_rate, bounds
    )

    # 输出最优解
    print("最优解：", best_solution)
    print("最优目标函数值：", best_fitness)


if __name__ == "__main__":
    main()
