"""
filename: sim_params.py

Simulation parameters for the Potjans and Diesmann microcircuit model

This code is part of STENDHAL
A NeuroMat package for simulate neuron networks, and analyse data.

Director:
Antonio Galves

Developers:
Nilton L. Kamiji
Renan Shimoura
Christophe Pouzat

Contributors:
Aline Duarte
Karine Guimaraes
Antonio Roque
Jorge Stolfi

July 28 2020
"""

import numpy as np


# Define Network Parameters extracted from original Potjans and Diesmann model
sim_dict = {
    # simulation time (ms)
    't_sim': 1000.0,
    # simulation step size (ms)
    'dt': 0.1,
    # number of threads
    'n_threads': 1,
    # number of MPI processes
    'n_mpi': 1,
    }
