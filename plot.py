import csv
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum

class Feature(Enum):
	fitness = 'fit'
	diversity = 'div'
	
	def __str__(self):
		return self.value

parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('islands_count', default=10, type=int, help='Count of islands')
parser.add_argument('feature_plotted', type=Feature, choices=list(Feature), help='Type of feature to be plotted')
parser.add_argument('-l', '--log', action='store_true', required=False, help='Plot Y axis in logarithmic scale')

args = parser.parse_args()
islands_count = args.islands_count
feature_plotted = args.feature_plotted
logarithmic_scale = args.log

islands = {}

for num in range(-1, islands_count):
	with open('results/islands/island_' + str(num) + '.csv') as f:
		reader = csv.reader(f, delimiter=',')
		islands[num] = []
		for row in reader:
			islands[num] += [row]
		islands[num] = np.array(islands[num][1:], dtype=np.float64)

fig, ax = plt.subplots()

ax.set(xlabel='generation')
if feature_plotted == Feature.fitness:
	ax.set(ylabel='avg. fitness')
	ax.set(title='Populations\' Fitness')
if feature_plotted == Feature.diversity:
	ax.set(ylabel='diversity')
	ax.set(title='Populations\' Diversity')

if logarithmic_scale == True:
	ax.set_yscale('log')

ax.grid()

for num in range(-1, islands_count):
	if feature_plotted == Feature.fitness:
		ax.plot(islands[num][:,1], label='Island ' + str(num))
	if feature_plotted == Feature.diversity:
		ax.plot(islands[num][:,2], label='Island ' + str(num))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.show()
