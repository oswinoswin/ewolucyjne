import random
from deap import base, creator, tools
from benchmark_functions import rastrigin
import numpy as np
import argparse
import time
import logging


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('islands_count', default=10, type=int)

    args = parser.parse_args()
    islands_count = args.islands_count
    max_iter = 500
    min_time_between_restarts = 5

    message_sending_probability = 0.01
    default_ttl = 1
    min_angle = np.pi/4
    experiment_repetitions = 10

    print("islands: {} epochs: {}, min_time_between_restarts: {}\n"
          "message_sending_probability: {}, message_ttl: {}, similarity: {}".format(islands_count, max_iter, min_time_between_restarts, message_sending_probability, default_ttl,min_angle))

    diversity_logger = logging.getLogger("diversityLogger")
    diversity_logger.setLevel(logging.INFO)
    dfh = logging.FileHandler("results/diversity.csv", mode='w')
    dfh.setLevel(logging.INFO)
    diversity_logger.addHandler(dfh)
    diversity_logger.info("epoch,diversity")

    islands = [Island(toolbox, tools, population_size, i, min_time_between_restarts, message_sending_probability, default_ttl, min_angle, dimension) for i in range(islands_count)]
    controlIsland = Island(toolbox, tools, population_size, -1, min_time_between_restarts, message_sending_probability, default_ttl, min_angle, dimension)  # normal evolution here

    ## set up a topology
    for i in range(islands_count -1):
    	for j in range(islands_count-1):
    		if i != j:
        		make_connection_between_islands(islands[i], islands[j])

    for rep in range(experiment_repetitions):
        start_time = time.perf_counter()
        for it in range(max_iter):
            controlIsland.evolution_step()
            for island in islands:
                island.evolution_step()
                # checking for stuff is in island

            # migrate
            migration_factor = 0.3
            if np.random.rand() < migration_factor:
                populations = [island.get_population() for island in islands]
                tools.migRing(populations, k=1, selection=tools.selBest )

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

        controlIsland.restart_population()
        controlIsland.current_generation = 0
