import random
from deap import base, creator, tools, algorithms
from benchmark_functions import rastrigin
import matplotlib.pyplot as plt
import numpy as np

dimension = 10
population_size = 50
x_min, x_max = -5.12, 5.12

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, x_min, x_max)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=dimension)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament, tournsize=3)

class Island:
    def __init__(self):
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.pop = toolbox.population(n=population_size)
        self.gens = []
        self.avgs = []
        self.mins = []
        self.maxs = []
        self.current_generation = 0

    def restart_population(self):
        self.pop = toolbox.population(n=population_size)

    def evolution_step(self):
        self.pop, logbook = algorithms.eaSimple(self.pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=self.stats, halloffame=self.hof,
                                            verbose=False)
        avg, min_, max_ = logbook.select("avg", "min", "max")
        if self.current_generation != 0:
            avg = avg[1:]
            min_ = min_[1:]
            max_ = max_[1:]
        self.gens.append(self.current_generation)
        self.avgs = self.avgs + avg
        self.mins = self.mins + min_
        self.maxs = self.maxs +max_
        self.current_generation += 1

    def get_results(self):
        self.gens.append(self.current_generation)
        return self.gens, self.avgs, self.mins, self.maxs

    def get_hof(self):
        return self.hof

    def get_population(self):
        return self.pop

    def get_best_individual(self):
        return self.hof[0]

    def get_best_fitness(self):
        return self.hof[0].fitness


def are_similar(A, B, epsilon):
    corr_matrix = np.corrcoef(A, B)
    return abs(corr_matrix[1,0]) < epsilon

if __name__ == "__main__":
    island1 = Island()
    island2 = Island()
    island3 = Island()

    max_iter = 500
    epsilon = 0.001

    for i in range(max_iter):
        island1.evolution_step()
        island2.evolution_step()
        island3.evolution_step()

        if i%10 == 0 and are_similar(island1.get_best_individual(), island2.get_best_individual(), epsilon):
            island2.restart_population()

        if i%10 == 5 and are_similar(island2.get_best_individual(), island3.get_best_individual(),epsilon):
            island3.restart_population()




    gen, avg1, min_1, max_1 = island1.get_results()
    gen, avg2, min_2, max_2 = island2.get_results()
    gen, avg3, min_3, max_3 = island3.get_results()


    plt.plot(gen[15:], avg1[15:], label="average for island 1")
    plt.plot(gen[15:], avg2[15:], label="average for island 2")
    plt.plot(gen[15:], avg3[15:], label="average for island 3")
    # plt.plot(gen, min_, label="minimum")
    # plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right")
    plt.title(f"Population size: {population_size} epsilon: {epsilon}")
    plt.savefig(f"iterations_{max_iter}_epsilon_{epsilon}_population_{population_size}_2_zoom.jpg")

