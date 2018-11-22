import random
from deap import base, creator, tools, algorithms
from benchmark_functions import rastrigin
import matplotlib.pyplot as plt
import numpy as np

dimension = 100
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
        return self.hof[0].fitness.values[0]

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    if sum(v1) == 0 or sum(v2) == 0:
        return 0.
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def are_similar(v1, v2, epsilon):
    angle = angle_between(v1,v2)
    return angle/np.pi < epsilon

if __name__ == "__main__":
    max_iter = 200
    similarity = 0.1
    islands_count = 30
    islands = [Island() for i in range(islands_count)]
    restart_probability = 0.1
    for it in range(max_iter):
        for island in islands:
            island.evolution_step()

        for i in range(1, islands_count):
            if np.random.rand() < restart_probability:
                prev_best = islands[i-1].get_best_individual()
                current_best = islands[i].get_best_individual()
                if are_similar(prev_best, current_best, similarity):
                    islands[i].restart_population()


    results = [ island.get_results() for island in islands ]
    for i,r in enumerate(results):
        plt.plot(r[0], r[1], label="avarage for island {}".format(i))

    best_results = [ island.get_best_fitness() for island in islands]
    print(best_results)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right")
    plt.title(f"Population size: {population_size} similarity: {similarity}")
    plt.show()