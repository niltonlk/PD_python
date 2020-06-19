# PD_python
Potjans and Diesmann microcircuit model (PD) implementation in python

# Files:
- PD_python.py: python class describint the PD model
- run_PD_python.py: python script to run the microcircuit whith layer specific external current input
- delay.npy: numpy array containing connection delay information. each line contains the delay from a single source to its respective target stored in the target.npy data below
- target.npy: numpy array containing connection target information. each line contains the targer from a single source. weight and delay are described in a different array
- weight.npy: numpy array containing connection weight information. each line contains the weight from a single source to its respective target. see above

# How to run:

python3 run_PD_python.py

# Output
- spike_recorder.txt: text file containig two colums with neuron id and spike time in (ms)
- raster_plot.png: raster plot of the 