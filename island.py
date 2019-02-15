import numpy as np
import random
from deap import algorithms
import logging

x_min, x_max = -5.12, 5.12


class Message:
    def __init__(self, sender, epoch, fitness, position, diversity, ttl):
        self.sender = sender
        self.epoch = epoch
        self.fitness = fitness
        self.position = position
        self.diversity = diversity
        self.ttl = ttl

    def decrease_ttl(self):
        self.ttl = self.ttl - 1

    def __str__(self):
        return "MESSAGE sender: {}, epoch: {}, fitness: {}, div: {}, ttl: {}".format(self.sender, self.epoch,
                                                                                     self.fitness, self.diversity,
                                                                                     self.ttl)


class Island:
    def __init__(self, toolbox, tools, population_size, id, min_time_between_restarts, message_sending_probability,
                 default_ttl, min_angle, dimensions, x_min, x_max, mutation_probability, crossover_probability,
                 individual_soft_restart_probability, soft_restart_sigma):
        self.min_angle = min_angle
        self.default_ttl = default_ttl
        self.message_sending_probability = message_sending_probability
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
        self.logger = self.prepare_logger()
        self.logger.info("epoch,fitness,diversity,time_since_restart")
        self.neighbours = []
        self.avg_fitness = None
        self.dimensions = dimensions
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.individual_soft_restart_probability = individual_soft_restart_probability
        self.soft_restart_sigma = soft_restart_sigma

        self.message_buffer = []

        #  restart params
        self.min_time_between_restarts = min_time_between_restarts
        self.x_min = x_min
        self.x_max = x_max

    def move_population(self):
        for i in range(0, self.population_size):
            if np.random.rand() < self.individual_soft_restart_probability:
                self.pop[i] = self.toolbox.mutate(self.pop[i], sigma=self.soft_restart_sigma)[0]

    def prepare_logger(self):
        logger = logging.getLogger("islandsLogger{}".format(self.id))
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler("results/islands/island_{}.csv".format(self.id), mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        return logger

    def restart_population(self):
        self.pop = self.toolbox.population(n=self.population_size)
        self.last_restart_time = self.current_generation
        self.hof.clear()

    def soft_restart_population(self):
        self.move_population()
        self.last_restart_time = self.current_generation

    def cut_values_outside_range(self):
        for i in range(0, self.population_size):
            for j, _ in enumerate(self.pop[i]):
                if self.pop[i][j] > self.x_max:
                    self.pop[i][j] = self.x_max
                if self.pop[i][j] < self.x_min:
                    self.pop[i][j] = self.x_min

    def evolution_step(self):
        self.pop, logbook = algorithms.eaSimple(self.pop, 
                                                self.toolbox, 
                                                cxpb=self.crossover_probability, 
                                                mutpb=self.mutation_probability, 
                                                ngen=1, 
                                                stats=self.stats,
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
        diversity = self.estimate_population_diversity()
        time_since_restart = self.time_since_restart()
        fitness = self.toolbox.evaluate(self.hof[0])[0]
        self.avg_fitness = avg[0]
        self.logger.info("{},{},{},{}".format(self.current_generation, fitness, diversity, time_since_restart))

        # process buffer
        restarted = False
        for message in self.message_buffer:
            if message.ttl < 1:
                self.message_buffer.remove(message)
                continue
            if self.should_restart(message, time_since_restart, fitness):
                self.soft_restart_population()
                restarted = True
            else:
                message.decrease_ttl()
                self.send_to_all_neighbours(message)
            self.message_buffer.remove(message)

        if self.current_generation > self.min_time_between_restarts and not restarted and np.random.rand() < self.message_sending_probability:
            message = Message(self.id, self.current_generation, fitness, self.estimate_population_center(), diversity,
                              self.default_ttl)
            self.send_to_all_neighbours(message)
        self.cut_values_outside_range()

    def receive_a_message(self, message):
        self.message_buffer.append(message)

    def send_to_all_neighbours(self, message):
        for neighbour in self.neighbours:
            if message.sender != neighbour.id:
                neighbour.receive_a_message(message)

    def should_restart(self, message, time_since_restart, fitness):
        if time_since_restart < self.min_time_between_restarts:
            return False
        if fitness < message.fitness:
            return False
        return are_similar(self.estimate_population_center(), message.position, self.min_angle)

    def estimate_population_diversity(self):
        return np.mean(np.std(self.pop, axis=0))

    def estimate_population_center(self):
        return np.mean(self.pop, axis=0)

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

    def get_avg_fitness(self):
        # return self.hof[0].fitness.values[0]
        return self.avg_fitness

    def time_since_restart(self):
        return self.current_generation - self.last_restart_time

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)
        
    def get_neighbours(self):
    	return self.neighbours


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    if sum(v1) == 0 or sum(v2) == 0:
        return 0.
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def are_similar(v1, v2, epsilon):
    angle = angle_between(v1, v2)
    return angle / np.pi < epsilon
