import random
from deap import base, creator, tools, algorithms
from benchmark_functions import rastrigin
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import logging
import csv

from island import Island

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


def islands_too_close(a, b):
    curr_pos, curr_sd = a.estimate_position()
    next_pos, next_sd = b.estimate_position()
    dist = np.linalg.norm(curr_pos - next_pos)
    if next_sd > 0 and np.random.rand() > dist / next_sd:
        return True
    return False


if __name__ == "__main__":
    # islands_count = 10
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('islands_count', default=10, type=int)
    parser.add_argument('--show_plot', dest='show_plot', action='store_true')

    args = parser.parse_args()
    islands_count = args.islands_count
    show_plot = args.show_plot
    max_iter = 200
    epsilon = 0.01
    min_sd = abs(x_max - x_min) * epsilon

    logger = logging.getLogger("islandsLogger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("results/experiment_pop_{}_dim_{}_circle.csv".format(population_size, dimension))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("epoch,fitness,island")

    islands = [Island(toolbox, tools, population_size, i, logger) for i in range(islands_count)]
    restart_probability = 0.1

    start_time = time.perf_counter()
    for it in range(max_iter):
        for island in islands:
            island.evolution_step()

        for i in range(0, islands_count):
            mean, std = islands[i].estimate_position()

            if std < min_sd and np.random.rand() < restart_probability and islands[i].get_population_age() > 70:
                islands[i].restart_population()
            else:
                if islands_too_close(islands[i], islands[(i + 1) % islands_count]) and islands[i].get_population_age() > 70:
                    islands[i].restart_population()

    results = [island.get_results() for island in islands]
    for i, r in enumerate(results):
        plt.plot(r[0], r[2], label="average for island {}".format(i))

    best_result = min([island.get_best_fitness() for island in islands])
    duration = time.perf_counter() - start_time

    print("{0},{1:.4f},{2:.4f}".format(islands_count, duration, best_result))
    # # if show_plot:
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.yscale("log")
    # plt.legend(loc="upper right")
    # plt.title('Population size: {} epsilon: {}'.format(population_size, epsilon))
    # plt.show()

