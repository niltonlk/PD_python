import numpy as np
import time
#import cProfile

import PD_python

tic = time.time()
# scale
scale = 0.1
# simulation time step
dt = 0.1
# number of neuron per layer
N = (np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])*scale).astype(int)
# layer specific output delay (fixed depending on neuron type - exc or inh)
layer_specific_delay = np.array([1.5, 0.8, 1.5, 0.8, 1.5, 0.8, 1.5, 0.8])
# layer specific external input per layer in (number of external inputs)
layer_specific_connection = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
poisson_bg_rate = 8.0
# layer specific external DC current per layer in (pA)
DC_BG = layer_specific_connection * 0.3512 #np.array([561.97, 526.85, 737.59, 667.34, 702.47, 667.34, 1018.58, 737.59])
# cumulative sum of N. used to determine neuron id range
N_cumsum = N.cumsum()
# create array of neuron ids
source = np.array(range(1,N_cumsum[-1]+1))
# load connecivity data. each line represents connection values of each neuron
target = np.load('target_01.npy', allow_pickle=True)
weight = np.load('weight_01.npy', allow_pickle=True)
delay = np.load('delay_01.npy', allow_pickle=True)
# target = np.load('/scratch/nilton/PD_var_delay/target.npy', allow_pickle=True)
# weight = np.load('/scratch/nilton/PD_var_delay/weight.npy', allow_pickle=True)
# delay = np.load('/scratch/nilton/PD_var_delay/delay.npy', allow_pickle=True)


#with cProfile.Profile() as pr:
# create network 
net_layer = PD_python.Network(N=N, fname='spike_recorder.txt')

# set layer specific poisson external input firing rate
# to change a neuron parameter, the set_neuron_params method of the layers must be called
for idx, layer_ in enumerate(net_layer.layer):
    layer_.set_neuron_params({'poisson_rate': layer_specific_connection[idx]*poisson_bg_rate}) #,
    #                               'syn_delay': layer_specific_delay[idx]})
    #     layer_.set_neuron_params({'I_0': DC_BG[idx]})
    # create connections. This will create a dictionary containing conection parameters
net_layer.create_connection(source, target, weight, delay)
create_time = time.time() - tic
print("Time to create the connections: {:.2f} s".format(create_time))
tic = time.time()
# simulate network for 60500 ms
#t_sim = 60500
# simulate network for 1000 ms
t_sim = 1000
net_layer.simulate(t_sim)
sim_time = time.time() - tic
print("Time to simulate: {:.2f} s".format(sim_time))
    
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

#pr.print_stats()


'''
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
#plt.savefig('raster_plot_var_delay.png', dpi=300)

end=t_sim; begin=t_sim - 1000;
if begin < 500:
    begin = 500

rates_all = []
rates_averaged_all = []
rates_std_all = []
idx_t500 = np.where((net_layer.spk_time >= begin) & (net_layer.spk_time <=end))[0]
count_of_n = np.bincount(net_layer.spk_neuron[idx_t500].astype(int))

for idx, id_range_ in enumerate(id_range):
    count_of_n_fil = count_of_n[id_range[idx][0]:id_range[idx][1]+1]
    rate_each_n = count_of_n_fil * 1000. / (end - begin)
    rate_averaged = np.mean(rate_each_n)
    rate_std = np.std(rate_each_n)
    rates_all.append(rate_each_n)
    rates_averaged_all.append(float('%.3f' % rate_averaged))
    rates_std_all.append(float('%.3f' % rate_std))
print(rates_averaged_all)
print(rates_std_all)

rates_all_rev = []
for idx in range(1,len(rates_all)+1):
    rates_all_rev.append(rates_all[-idx])

pop_names = ['L23e','L23i','L4e','L4i','L5e','L5i','L6e','L6i']
label_pos = list(range(len(N), 0, -1))
color_list = ['#888888', '#000000']
medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
fig, ax1 = plt.subplots(figsize=(10, 6))
bp = plt.boxplot(rates_all_rev, 0, 'rs', 0, medianprops=medianprops)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
for h in list(range(len(N))):
    boxX = []
    boxY = []
    box = bp['boxes'][h]
    for j in list(range(5)):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    k = h % 2
    boxPolygon = Polygon(boxCoords, facecolor=color_list[k])
    ax1.add_patch(boxPolygon)
plt.xlabel('firing rate [Hz]', fontsize=18)
plt.yticks(label_pos, pop_names, fontsize=18)
plt.xticks(fontsize=18)
plt.savefig('box_plot_var_delay.png', dpi=300)

plt.show()
'''
