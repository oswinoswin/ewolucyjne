import csv
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum
from collections import OrderedDict
import math

class Feature(Enum):
	fitness = 'fit'
	diversity = 'div'
	
	def __str__(self):
		return self.value

parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('islands_count', default=10, type=int, help='Count of islands')
parser.add_argument('feature_plotted', type=Feature, choices=list(Feature), help='Type of feature to be plotted')
parser.add_argument('-l', '--log', action='store_true', required=False, help='Plot Y axis in logarithmic scale')
parser.add_argument('-i', '--per-island', action='store_true', required=False, help="Plot diagrams on per-island basis instead of one averaged")
parser.add_argument('-o', '--output-file', required=False, help='Output file name')

args = parser.parse_args()
islands_count = args.islands_count
feature_plotted = args.feature_plotted
logarithmic_scale = args.log
per_island_plot = args.per_island
output_file = args.output_file

if per_island_plot == True:

	islands = {}

	for num in range(0, 2*islands_count):
		with open('results/islands/island_' + str(num) + '.csv') as f:
			reader = csv.reader(f, delimiter=',')
			islands[num] = []
			for row in reader:
				islands[num] += [row]
			islands[num] = np.array(islands[num][1:], dtype=np.float64)

	fig, ax = plt.subplots()

	ax.set(xlabel='generation')
	if feature_plotted == Feature.fitness:
		ax.set(ylabel='best fitness')
		ax.set(title='Populations\' Fitness')
	if feature_plotted == Feature.diversity:
		ax.set(ylabel='diversity')
		ax.set(title='Populations\' Diversity')

	if logarithmic_scale == True:
		ax.set_yscale('log')

	ax.grid()

	for num in range(0, 2*islands_count):
		if feature_plotted == Feature.fitness:
			ax.plot(islands[num][:,1], label='Island ' + str(num))
		if feature_plotted == Feature.diversity:
			ax.plot(islands[num][:,2], label='Island ' + str(num))

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
	
	if output_file == None:
		plt.show()
	else:
		fig.savefig(output_file + '.png')
	
else:

	# all islands except control island
	
	islands = OrderedDict()

	for num in range(islands_count, 2*islands_count):
		with open('results/islands/island_' + str(num) + '.csv') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				if not row[0] in islands:
					islands[row[0]] = []
				islands[row[0]] += [row]

	islands_copy = islands
	islands = OrderedDict()
	for key in islands_copy:
		islands[key] = np.array(islands_copy[key], dtype=np.float64)

	means = []
	lbounds = []
	ubounds = []
	for key in islands:
		if feature_plotted == Feature.fitness:
			val = islands[key][:,1].mean()
			lb = islands[key][:,1].min()
			ub = islands[key][:,1].max()
			#std = islands[key][:,1].std() / math.sqrt(len(islands[key][:,1]))
		if feature_plotted == Feature.diversity:
			val = islands[key][:,2].mean()
			lb = islands[key][:,2].min()
			ub = islands[key][:,2].max()
			#std = islands[key][:,2].std() / math.sqrt(len(islands[key][:,2]))
		means += [val]
		lbounds += [lb]
		ubounds += [ub]
	means = np.array(means, dtype=np.float64)
	lbounds = np.array(lbounds, dtype=np.float64)
	ubounds = np.array(ubounds, dtype=np.float64)

	# control islands
	
	islands = OrderedDict()
	
	for num in range(0, islands_count):
		with open('results/islands/island_' + str(num) + '.csv') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				if not row[0] in islands:
					islands[row[0]] = []
				islands[row[0]] += [row]

	islands_copy = islands
	islands = OrderedDict()
	for key in islands_copy:
		islands[key] = np.array(islands_copy[key], dtype=np.float64)

	control_means = []
	control_lbounds = []
	control_ubounds = []
	for key in islands:
		if feature_plotted == Feature.fitness:
			val = islands[key][:,1].mean()
			lb = islands[key][:,1].min()
			ub = islands[key][:,1].max()
			#std = islands[key][:,1].std() #/ math.sqrt(len(islands[key][:,1]))
		if feature_plotted == Feature.diversity:
			val = islands[key][:,2].mean()
			lb = islands[key][:,2].min()
			ub = islands[key][:,2].max()
			#std = islands[key][:,2].std() #/ math.sqrt(len(islands[key][:,2]))
		control_means += [val]
		control_lbounds += [lb]
		control_ubounds += [ub]
	control_means = np.array(control_means, dtype=np.float64)
	control_lbounds = np.array(control_lbounds, dtype=np.float64)
	control_ubounds = np.array(control_ubounds, dtype=np.float64)
	
	fig, ax = plt.subplots(figsize=(12,8))

	ax.set(xlabel='generation')
	if feature_plotted == Feature.fitness:
		ax.set(ylabel='best fitness')
		ax.set(title='Populations\' Fitness')
	if feature_plotted == Feature.diversity:
		ax.set(ylabel='diversity')
		ax.set(title='Populations\' Diversity')

	if logarithmic_scale == True:
		ax.set_yscale('log')

	plt.xlim(0,1500)
	
	if feature_plotted == Feature.fitness:
		plt.ylim(0.000001,1000)
	if feature_plotted == Feature.diversity:
		plt.ylim(0.00001,10)

	ax.grid()
	
	# control islands
	ax.plot(control_lbounds, label='Control lower bound')
	
	if feature_plotted == Feature.fitness:
		ax.plot(control_means, label='Control mean best fitness')
	if feature_plotted == Feature.diversity:
		ax.plot(control_means, label='Control mean diversity')
		
	ax.plot(control_ubounds, label='Control upper bound')
	
	# test islands
	ax.plot(lbounds, label='Lower bound')
	
	if feature_plotted == Feature.fitness:
		ax.plot(means, label='Mean best fitness')
	if feature_plotted == Feature.diversity:
		ax.plot(means, label='Mean diversity')
		
	ax.plot(ubounds, label='Upper bound')
	

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
	
	if output_file == None:
		plt.show()
	else:
		fig.savefig(output_file + '.png')
