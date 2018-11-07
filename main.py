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
                                            verbose=True)
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

if __name__ == "__main__":
    island = Island()
    for i in range(100):
        island.evolution_step()
    island.restart_population()
    for i in range(100):
        island.evolution_step()

    island.restart_population()
    for i in range(100):
        island.evolution_step()
    hof = island.get_hof()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    gen, avg, min_, max_ = island.get_results()
    print(gen)
    print(avg)

    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()

