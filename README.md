# PD_python
Potjans and Diesmann microcircuit model (PD) implementation in python

# Files:
- PD_python.py: python class to create and simulate a neuron network model with the GL neuron and poisson external input
- run_PD_python.py: python script to run the microcircuit whith layer specific poisson external input
- delay.npy: numpy array containing connection delay information. each line contains the delay from a single source to its respective target stored in the target.npy data below
- target.npy: numpy array containing connection target information. each line contains the targer from a single source. weight and delay are described in a different array
- weight.npy: numpy array containing connection weight information. each line contains the weight from a single source to its respective target. see above

# Requirements:

- python 3.6
- numpy > 1.17

# How to run:

python3 run_PD_python.py

# Output
- spike_recorder.txt: text file containig two colums with neuron id and spike time in (ms)

# Scaled down version
There is a scaled down version in scaled branch which contains connectivity data and sample output