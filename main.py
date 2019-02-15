import random
from deap import base, creator, tools
from benchmark_functions import rastrigin
import numpy as np
import argparse
import time
import logging
from collections import Sequence
from island import Island
from itertools import repeat
import math

def mutUniform(individual, low, up, indpb):
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, l, u in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] += random.uniform(l, u)

    return individual,

dimension = 50
population_size = 100
x_min, x_max = -5.12, 5.12


mutation_probability = 0.4
crossover_probability = 0.7
gene_mutation_probability = 1.0 / dimension
gaussian_mutation_sigma = 4.0
migration_probability = 0.2
min_time_between_restarts = 70
message_sending_probability = 0.0
message_ttl = 1
min_angle = np.pi / 8

individual_soft_restart_probability = 1.0
soft_restart_sigma = gaussian_mutation_sigma

max_iter = 1500

experiment_repetitions = 5


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, x_min, x_max)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=dimension)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=5, low=x_min, up=x_max)
toolbox.register("mutate", mutUniform, low=x_min, up=x_max, indpb=gene_mutation_probability)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("selectMig", tools.selRandom)
toolbox.register("selectRepl", random.sample)


def islands_too_close(a, b):
    curr_pos, curr_sd = a.estimate_position()
    next_pos, next_sd = b.estimate_position()
    dist = np.linalg.norm(curr_pos - next_pos)
    if next_sd > 0 and np.random.rand() > dist / next_sd:
        return True
    return False


def make_connection_between_islands(a, b):
    a.add_neighbour(b)
    b.add_neighbour(a)

def print_neighbours():
    for i in range(islands_count):
        print(str(islands[i].id) + ': ' + str([isl.id for isl in islands[i].get_neighbours()]))

def make_topology(type, islands, islands_count):
    if type == "ring":
        for i in range(islands_count-1):
            make_connection_between_islands(islands[i], islands[i+1])
        make_connection_between_islands(islands[0], islands[-1])
        return

    if type == "star":
        for i in range(1, islands_count):
            make_connection_between_islands(islands[0], islands[i])
        return

    if type == "clique":
        for i in range(islands_count):
            for j in range(islands_count):
                if i != j:
                    make_connection_between_islands(islands[i], islands[j])
        return

    if type == "torus":
        width = int(np.sqrt(islands_count))
        height = int(np.ceil(islands_count / width))
        #print('width: ' + str(width))
        #print('height: ' + str(height))
        for i in range(islands_count):
            row = int(i / width)
            col = int(i % width)
            if i >= islands_count - (islands_count % width):
                make_connection_between_islands(islands[i], islands[row*width + (col+1)%(islands_count % width)])
                make_connection_between_islands(islands[i], islands[((row+1) % height)*width + col])
            else:
                make_connection_between_islands(islands[i], islands[row*width + (col+1)%width])
                if islands_count % width == 0 or col < islands_count % width:
                    make_connection_between_islands(islands[i], islands[((row+1) % height)*width + col])
                else:
                    make_connection_between_islands(islands[i], islands[((row+1) % (height-1))*width + col])
        return
    
    raise(Exception("Unknown topology"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('islands_count', default=5, type=int)
    parser.add_argument('--topology', default='star')
    parser.add_argument('-mtbr', default=min_time_between_restarts, type=int, help='Minimal time between restarts')
    parser.add_argument('-msp', default=message_sending_probability, type=float, help='Message sending probability')
    parser.add_argument('-ttl', default=message_ttl, type=int, help='Message ttl')
    parser.add_argument('-ma', default=min_angle, type=float, help='Minimal angle between islands')
    parser.add_argument('-isrp', default=individual_soft_restart_probability, type=float, help='Individual soft restart probability')
    parser.add_argument('-srs', default=soft_restart_sigma, type=float, help='Soft restart sigma')
    

    args = parser.parse_args()
    islands_count = args.islands_count
    topology = args.topology
    min_time_between_restarts = args.mtbr
    message_sending_probability = args.msp
    message_ttl = args.ttl
    min_angle = args.ma
    individual_soft_restart_probability = args.isrp
    soft_restart_sigma = args.srs

    print("islands: {} epochs: {}, topology: {}, min_time_between_restarts: {}\n"
          "message_sending_probability: {}, message_ttl: {}, min_angle: {}\n"
          "individual_soft_restart_probability: {}, soft_restart_sigma: {}".format(islands_count, max_iter, topology,
                                                                                   min_time_between_restarts,
                                                                                   message_sending_probability,
                                                                                   message_ttl, min_angle,
                                                                                   individual_soft_restart_probability,
                                                                                   soft_restart_sigma))

    diversity_logger = logging.getLogger("diversityLogger")
    diversity_logger.setLevel(logging.INFO)
    dfh = logging.FileHandler("results/diversity.csv", mode='w')
    dfh.setLevel(logging.INFO)
    diversity_logger.addHandler(dfh)
    diversity_logger.info("epoch,diversity")

    controlIslands = [Island(toolbox, tools, population_size, i, min_time_between_restarts, 0, message_ttl,
                             min_angle, dimension, x_min, x_max, mutation_probability, crossover_probability, 
                             individual_soft_restart_probability, soft_restart_sigma) for i in range(islands_count)]
               
    islands = [Island(toolbox, tools, population_size, i + islands_count, min_time_between_restarts, message_sending_probability,
                      message_ttl, min_angle, dimension, x_min, x_max, mutation_probability, crossover_probability,
                      individual_soft_restart_probability, soft_restart_sigma) for i in range(islands_count)]

    make_topology(topology, islands, islands_count)
    #print_neighbours()

    for rep in range(experiment_repetitions):
        start_time = time.perf_counter()
        for it in range(max_iter):

            for island in controlIslands:
                island.evolution_step()

            for island in islands:
                island.evolution_step()

            # migrate 
            if np.random.rand() < migration_probability:
                populations = [island.get_population() for island in controlIslands]
                tools.migRing(populations, k=3, selection=toolbox.selectMig, replacement=toolbox.selectRepl)

                populations = [island.get_population() for island in islands]
                tools.migRing(populations, k=3, selection=toolbox.selectMig, replacement=toolbox.selectRepl)

            positions = [island.estimate_position()[0] for island in islands]
            mean_position_std = np.std(positions)
            diversity_logger.info("{}, {}".format(it, mean_position_std))

        results = [island.get_results() for island in islands]

        best_result = min([island.get_avg_fitness() for island in islands])
        duration = time.perf_counter() - start_time

        print("{0},{1:.4f},{2:.4f}".format(islands_count, duration, best_result))

        #  restart islands for next iteration
        for island in islands:
            island.restart_population()
            island.current_generation = 0
            island.last_restart_time = 0

        for island in controlIslands:
            island.restart_population()
            island.current_generation = 0
