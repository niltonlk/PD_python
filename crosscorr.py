import numpy as np
import os
import matplotlib.pyplot as plt
import math

def cross_int(T, R, lag=1000, max_steps=1000000, binsize=1, dt=0.1):
    # cross_int( T, R, lag=1000, max_time=1000000)
    #
    # Input:
    #   R: list or array of spike times in number of steps (integer); Reference
    #   T: list or array of spike times in number of steps (integer); Test
    #   lag: distance in steps to calculate the crosscorrelogram (in steps; integer)
    #   max_steps: simulation duration (in steps)
    
    # binsize must be odd
    if binsize%2 == 0:
        binsize += 1
        print('binsize changed to {}. must be an odd number'.format(binsize))

    # Determine binned lag size to determine binned results length
    bin_not_multiple = (2*lag+1)%binsize # flag if length of result is not multiple of binsize
    binned_lag = lag//binsize + (1 if lag%binsize > binsize//2 else 0)
    res = [0 for i in range(2*binned_lag+1)] # list containing the unbinned results
    # Extend lag to include full binsize on edges if 
    new_lag = binned_lag*binsize+binsize//2 if bin_not_multiple else lag

    for r in R: # we take all the spikes in reference list
        if new_lag < r < max_steps-new_lag: # make sure there is enought room on both sides
            for t in T: # we take all the spikes in test list
                if r-new_lag <= t <= r+new_lag: # t is within the window of the reference spike
                    res[(t-r+new_lag)//binsize] += 1 # classical cross-correlogram
    
    # Stabilize the variance following Brillinger (Brillinger, Bryant & Segundo.
    # (1976) Identification of Synaptic Interactions. Biol. Cybernetics 22, 213-228
    res_stab = 2*np.sqrt(res)
    
    # plot results:
    ll = list(range(-binned_lag*binsize, (binned_lag+1)*binsize, binsize))

    plt.plot(ll, res_stab)
    plt.show()
    return ll, res_stab

    
path = 'data' # data path
name = 'spike_detector' # spike data prefix

files = []
for file in os.listdir(path):
    if file.startswith(name):
        temp = file.split('-')[0] + '-' + file.split('-')[1]
        if temp not in files:
            files.append(temp)
files = sorted(files) # sort files

# read GIDs data. matrix containing first and last neuron ID of each layer
gids = np.loadtxt(os.path.join(path, 'population_GIDs.dat')).astype(int)

# read spike data 
lnames = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i'] # layer names
data = {}
begin = 500.0 # discart first 500 ms
for i, file in enumerate(files):
    temp3 = [f for f in os.listdir(path) if f.startswith(file)]
    data_temp = [np.loadtxt(os.path.join(path, f)) for f in temp3]
    data_concatenated = np.concatenate(data_temp)
    data_concatenated[:, 1] = (data_concatenated[:, 1]-begin)*10 # multiply time by 10 to discard decimals
    data_concatenated = data_concatenated.astype(int) # convert to int to save memory and time
    data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
    idx = data_raw[:, 1] > 0
    data[lname[i]] = data_raw[idx]

spk = {} # empty dict for spike times
spk_count = {} # empty dict for spike counts
for i, lname in enumerate(lnames): # take all layers
    spk[lname] = [] # empty list
    spk_count[lname] = [] # emply list
    for j in range(gids[i][0], gids[i][1]+1): # for all neurons in layer
        idx = data[lname][:, 0] == j # find all indices for neuron j
        spk[lname].append(data[lname][idx, 1]) # add spike times
        spk_count[lname].append(len(data[name][idx,1])) # add number of spikes

