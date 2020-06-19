import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PD_python

# simulation time step
dt = 0.1
# number of neuron per layer
N = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
# layer specific external DC current per layer in (pA)
DC_BG = np.array([561.97, 526.85, 737.59, 667.34, 702.47, 667.34, 1018.58, 737.59])
# cumulative sum of N. used to determine neuron id range
N_cumsum = N.cumsum()
# create array of neuron ids
source = np.array(range(1,N_cumsum[-1]+1))
# load connecivity data. each line represents connection values of each neuron
target = np.load('target.npy')
weight = np.load('weight.npy')
delay = np.load('delay.npy')

# create network 
net_layer = PD_python.Network(N=N)

# set layer specific external current input 
# to change a neuron parameter, the set_neuron_params method of the layers must be called
for idx, layer_ in enumerate(net_layer.layer):
    layer_.set_neuron_params({'I_0': DC_BG[idx]})
# create connections. This will create a dictionary containing conection parameters
net_layer.create_connection(source, target, weight, delay)
# simulate network for 1000 ms
net_layer.simulate(1000)

# matrix of start and end id of neurons for each layer.
# used for specify axis label position
# and to slice data
id_range = np.array([[1, N_cumsum[0]],
                    [N_cumsum[0]+1, N_cumsum[1]],
                    [N_cumsum[1]+1, N_cumsum[2]],
                    [N_cumsum[2]+1, N_cumsum[3]],
                    [N_cumsum[3]+1, N_cumsum[4]],
                    [N_cumsum[4]+1, N_cumsum[5]],
                    [N_cumsum[5]+1, N_cumsum[6]],
                    [N_cumsum[6]+1, N_cumsum[7]]])

# id of the last neuron in the network
highest_id = id_range[-1][-1]
# calculate yaxis label position
id_changed = abs(id_range - highest_id) + 1
L23_label_pos = (id_changed[0][0] + id_changed[1][1])/2
L4_label_pos = (id_changed[2][0] + id_changed[3][1])/2
L5_label_pos = (id_changed[4][0] + id_changed[5][1])/2
L6_label_pos = (id_changed[6][0] + id_changed[7][1])/2
ylabels = ['L23', 'L4', 'L5', 'L6']

# dark color for excitatory neurons, light color for inhibitory neurons
color_list = [
    '#000000', '#888888', '#000000', '#888888',
    '#000000', '#888888', '#000000', '#888888'
    ]

# plot raster plot
Fig1 = plt.figure(1, figsize=(8, 6))
for i in list(range(len(id_range))):
    # get index of neurons for each layer
    idx = (net_layer.spk_neuron >= id_range[i][0]) * \
            (net_layer.spk_neuron <= id_range[i][1])
    times = net_layer.spk_time[idx]
    neurons = np.abs(net_layer.spk_neuron[idx] - highest_id) + 1
    plt.plot(times, neurons, '.', color=color_list[i])
plt.xlabel('time [ms]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(
    [L23_label_pos, L4_label_pos, L5_label_pos, L6_label_pos],
    ylabels, rotation=10, fontsize=18
    )
plt.xlim([600,800])
# save figure
plt.savefig('raster_plot.png', dpi=300)
plt.show()
