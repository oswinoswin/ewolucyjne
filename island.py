import numpy as np
from deap import algorithms


class Message:
    def __init__(self, source, epoch, fitness, position, diversity):
        self.source = source
        self.epoch = epoch
        self.fitness = fitness
        self.position = position
        self.diversity = diversity


class Island:
    def __init__(self, toolbox, tools, population_size, id, logger):
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.pop = toolbox.population(n=population_size)
        self.toolbox = toolbox
        self.tools = tools
        self.population_size = population_size
        self.gens = []
        self.avgs = []
        self.mins = []
        self.maxs = []
        self.current_generation = 0
        self.last_restart_time = 0
        self.id = id
        self.logger = logger
        self.neighbours = []

    def restart_population(self):
        self.pop = self.toolbox.population(n=self.population_size)

    def evolution_step(self):
        self.pop, logbook = algorithms.eaSimple(self.pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=self.stats,
                                                halloffame=self.hof,
                                                verbose=False)
        avg, min_, max_ = logbook.select("avg", "min", "max")
        if self.current_generation != 0:
            avg = avg[1:]
            min_ = min_[1:]
            max_ = max_[1:]
        self.gens.append(self.current_generation)
        self.avgs = self.avgs + avg
        self.mins = self.mins + min_
        self.maxs = self.maxs + max_
        self.current_generation += 1
        self.logger.info("{},{},{}".format(self.current_generation, avg[0], self.id))

    def estimate_position(self):
        mean = np.mean(self.pop, axis=0)
        std = np.mean(np.std(self.pop, axis=0))
        return mean, std

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

    def get_population_age(self):
        return self.current_generation - self.last_restart_time

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)
