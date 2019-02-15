import subprocess
import numpy as np
import os

islands                             = 10

min_time_between_restarts           = [10, 30, 100]
message_sending_probability         = [0.003, 0.01, 0.03]
message_ttl                         = [1, 3, 10, 30, 100]
min_angle                           = [np.pi/6, np.pi/8, np.pi/12]
individual_soft_restart_probability = [1.0]
soft_restart_sigma                  = [6.0, 8.0]

calls = []

for srs in soft_restart_sigma:
    for isrp in individual_soft_restart_probability:
        for ma in min_angle:
            for ttl in message_ttl:
                for msp in message_sending_probability:
                    for mtbr in min_time_between_restarts:
                        calls += [[['python3', 'main.py', str(islands),
                                   '--topology', 'torus',
                                   '-mtbr', str(mtbr),
                                   '-msp', str(msp),
                                   '-ttl', str(ttl),
                                   '-ma', str(ma),
                                   '-isrp', str(isrp),
                                   '-srs', str(srs)],
                                   ['python3', 'plot.py', str(islands), 
                                   'fit', '-l',
                                   '-o', '../results/fig_' + str(islands) + 
                                         '-' + str(mtbr) +
                                         '-' + str(msp) +
                                         '-' + str(ttl) +
                                         '-' + str(ma) +
                                         '-' + str(isrp) +
                                         '-' + str(srs)],
                                   ['python3', 'plot.py', str(islands), 
                                   'fit', '-l', '-i',
                                   '-o', '../results/fig_' + str(islands) + 
                                         '-' + str(mtbr) +
                                         '-' + str(msp) +
                                         '-' + str(ttl) +
                                         '-' + str(ma) +
                                         '-' + str(isrp) +
                                         '-' + str(srs) +
                                         '_i'],
                                   ['python3', 'plot.py', str(islands), 
                                   'div', '-l',
                                   '-o', '../results/fig_' + str(islands) + 
                                         '-' + str(mtbr) +
                                         '-' + str(msp) +
                                         '-' + str(ttl) +
                                         '-' + str(ma) +
                                         '-' + str(isrp) +
                                         '-' + str(srs) +
                                         '_div']]]


os.chdir('/home/maciek/agh/intob/ewolucyjne')

for i in range(len(calls)):
    print('\nRunning: ' + str(i+1) + '/' + str(len(calls)) + '\n')
    for subcall in calls[i]:
        subprocess.run(subcall)

