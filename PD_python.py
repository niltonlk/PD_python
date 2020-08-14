'''
filename: PD_python.py

This code simulates the PD microcircuit model (Potjans and Diesmann, 2014)
with the GL (Galves and Locherbach, 2013) neuron in discrete time.

This code is part of STENDHAL package.
A NeuroMat package for simulate neuron networks, and analyse data.

Director:
Antonio Galves

Developers:
Nilton L. Kamiji
Christophe Pouzat

Contributors:
Renan Shimoura
Karine Guimaraes
Aline Duarte
Antonio Roque

July 09 2020
'''

import numpy as np

class Layer:
    id_0 = 0 # initial neuron ID of the layer
    N = 10 # number of neurons in the layer
    net = None
    
    class P: # model parameters
        
        def __init__(self):
            self.tau_m = 10.0 # membrane time constant in (ms)
            self.t_ref = 2.0 # refractory period in (ms)
            self.V_reset = 0.0 # reset membrane potential in (mV)
            self.V_th = 15.0 # threshold potential in (mV)
            self.C_m = 250.0 # membrance capacitance im (pF)
            self.tau_syn_ex = 0.5 # excitatory synaptic time constant in (ms)
            self.tau_syn_in = 0.5 # inhibitory synaptic time constant in (ms)
            self.I_0 = 0.0 # constant current in (pA)
            self.gamma = 0.1 # slope of the firing probability funtion in (1/mV)
            self.r = 0.4 # curvature of the firing probability function (unitless)
            self.V_rheo = 15.0 # rheobase potential, potential in which firing probability becomes > 0 in (mV)
            self.poisson_rate = 8.0*10 # rate of poisson spike input in (Hz)
            self.poisson_weight = 87.8 # weight of poisson input in (pA)
        
        def set_params(self, neuron_params):
            if 'tau_m' in neuron_params.keys():
                self.tau_m = neuron_params['tau_m']
            if 't_ref' in neuron_params.keys():
                self.t_ref = neuron_params['t_ref']
            if 'V_reset' in neuron_params.keys():
                self.V_reset = neuron_params['V_reset']
            if 'V_th' in neuron_params.keys():
                self.V_th = neuron_params['V_th']
            if 'C_m' in neuron_params.keys():
                self.C_m = neuron_params['C_m']
            if 'tau_syn_ex' in neuron_params.keys():
                self.tau_syn_ex = neuron_params['tau_syn_ex']
            if 'tau_syn_in' in neuron_params.keys():
                self.tau_syn_in = neuron_params['tau_syn_in']
            if 'I_0' in neuron_params.keys():
                self.I_0 = neuron_params['I_0']
            if 'gamma' in neuron_params.keys():
                self.gamma = neuron_params['gamma']
            if 'r' in neuron_params.keys():
                self.r = neuron_params['r']
            if 'V_rheo' in neuron_params.keys():
                self.V_rheo = neuron_params['V_rheo']
            if 'poisson_rate' in neuron_params.keys():
                self.poisson_rate = neuron_params['poisson_rate']


    class S: # state variables
        
        def __init__(self, N=None):
            self.N = N
            self.V_m = np.zeros(N) # membrane potential in (mV)
            self.I_syn_ex = np.zeros(N) # synaptic current in (pA)
            self.I_syn_in = np.zeros(N) # synaptic current in (pA)
            self.I_ext = np.zeros(N) # external input in (pA)
            self.is_ref = np.zeros(N).astype(int) # flag that indicates neuron refractory period
            self.poisson = np.zeros(N).astype(int) # number of poisson spike input per neuron
        
        def reset_state(self):
            self.V_m = np.zeros(self.N)
            self.I_syn_ex = np.zeros(self.N) # synaptic current in (pA)
            self.I_syn_in = np.zeros(self.N) # synaptic current in (pA)
            self.I_ext = np.zeros(self.N) # external input in (pA)
            self.is_ref = np.zeros(self.N).astype(int) # flag that indicates neuron refractory period
            self.poisson = np.zeros(self.N).astype(int) # number of poisson spike input per neuron
        
        def set_params(self, params):
            if 'V_m' in params.keys():
                try:
                    if len(params['V_m']) != len(self.V_m):
                        raise Exception('V_m array must be of length {}'.format(len(self.V_m)))
                    self.V_m = params['V_m']
                except:
                    print('V_m array must be of length {}'.format(self.V_m))
            if 'I_ext' in params.keys():
                try:
                    if len(params['I_ext']) != len(self.I_ext):
                        raise Exception('I_ext array must be of length {}'.format(len(self.I_ext)))
                    self.I_ext = params['I_ext']
                except:
                    print('I_ext array must be of length {}'.format(self.I_ext))
                
        
    class V: # auxiliary variables
        
        def __init__(self, dt=0.1, P=None):
            self.calibrate(dt=dt, P=P)
            
        def calibrate(self, dt=0.1, P=None):
            self.ref_count = int(P.t_ref / dt)
            self.rho = np.exp(-dt/P.tau_m)
            self.xi_ex = np.exp(-dt/P.tau_syn_ex)
            self.xi_in = np.exp(-dt/P.tau_syn_in)
            self.zeta_ex = -P.tau_m/( P.C_m * (1.0 - P.tau_m/P.tau_syn_ex) ) * \
                    np.exp(-dt/P.tau_syn_ex) * \
                    ( np.exp( dt * (1.0/P.tau_syn_ex - 1.0/P.tau_m) ) - 1.0 )
            self.zeta_in = -P.tau_m/( P.C_m * (1.0 - P.tau_m/P.tau_syn_in) ) * \
                    np.exp(-dt/P.tau_syn_in) * \
                    ( np.exp( dt * (1.0/P.tau_syn_in - 1.0/P.tau_m) ) - 1.0 )
            self.zeta_DC = P.tau_m / P.C_m * ( 1.0 - self.rho )
    
    
    class B: # input buffer
        
        def __init__(self, bl=5.0, dt=0.1, N=10):
            self.dt = dt
            self.N = N
            self.buffer_length = bl # buffer length in (ms)
            self.reset_buffer()
        
        def reset_buffer(self):
            self.weighted_spikes_ex = np.zeros((int(self.buffer_length/self.dt), self.N))
            self.weighted_spikes_in = np.zeros((int(self.buffer_length/self.dt), self.N))
            
        def shift_buffer(self):
            for idx in range(len(self.weighted_spikes_ex)-1):
                self.weighted_spikes_ex[idx] = self.weighted_spikes_ex[idx+1]
                self.weighted_spikes_in[idx] = self.weighted_spikes_in[idx+1]
            self.weighted_spikes_ex[-1][:] = 0.0
            self.weighted_spikes_in[-1][:] = 0.0

            
    def __init__(self, net=None, N=10, id_0=0, seed_phi=1234, seed_poisson=2345):
        self.id_0 = id_0 # initial neuron ID of the layer
        self.N = N # number of neurons in layer
        self.net = net # simulation time step in (ms)
        self.rng = np.random.default_rng(seed=seed_phi) # needs numpy > 1.17 random number generator for phi
        self.poisson_rng = np.random.default_rng(seed=seed_poisson) # needs numpy > 1.17 random number generator for poisson generator
#         np.random.seed(1234)

        # initialize state variables
        self.P_ = self.P()
        self.S_ = self.S(N)
        self.V_ = self.V(net.dt, self.P_)
        self.B_ = self.B(bl=20.0, dt=net.dt, N=N)
        
        
    def set_neuron_params(self, neuron_params):
        self.P_.set_params(neuron_params)
        self.S_.set_params(neuron_params)
        
        self.V_.calibrate(dt=self.net.dt, P=self.P_)
    
    
    def set_state(self, state_val):
        self.S_.set_params(state_val)

        
    def receive_spike(self, target, weight, delay):
#         print(target, weight, delay, target-self.id_0-1)
        if weight > 0:
            self.B_.weighted_spikes_ex[int(delay/self.net.dt)] \
                    [int(target-self.id_0-1)] += weight
        else:
            self.B_.weighted_spikes_in[int(delay/self.net.dt)] \
                    [int(target-self.id_0-1)] += weight

    
    def shift_buffer(self):
        self.B_.shift_buffer()
        
    
    def phi(self):
        V_diff = self.S_.V_m - self.P_.V_rheo
        # set negative values to 0.0; same as setting phi(V) = 0 if V_m < V_rheo
        idx_neg = np.where(V_diff < 0) 
        V_diff[idx_neg] = 0.0
        # it is not necessary to clip phi to 1.0, as phi(V) > 1.0 will act the same as 1.0
        return np.power(self.P_.gamma * V_diff, self.P_.r)
    
    
    def evaluate(self):
        # evolve membrane potential to the next value
        # States at the right is at time t
        # V_m is now at time t+net.dt
        # algorithm based on Rotter and Diesmann (1999)
        self.S_.V_m = self.S_.V_m * self.V_.rho + \
                      self.S_.I_syn_ex * self.V_.zeta_ex + \
                      self.S_.I_syn_in * self.V_.zeta_in + \
                      (self.P_.I_0 + self.S_.I_ext) * self.V_.zeta_DC

        # generate poisson spike train for time window dt
        # convert time from ms to s (poisson_rate is in Hz, while dt is in ms)
        lambda_ = self.P_.poisson_rate * self.net.dt * 1e-3
        # draw sample from a poisson distribution
        self.S_.poisson = self.poisson_rng.poisson(lambda_, self.N) 

        # evolve synaptic currents (weighted_spikes_ex[0] must contain spikes ariving at time t+dt)
        self.S_.I_syn_ex = self.S_.I_syn_ex * self.V_.xi_ex + \
                           self.B_.weighted_spikes_ex[0] + \
                           self.S_.poisson*self.P_.poisson_weight
        self.S_.I_syn_in = self.S_.I_syn_in * self.V_.xi_in + \
                           self.B_.weighted_spikes_in[0]
        
        # decrement is_ref by 1; is_ref is now at time t+dt
        self.S_.is_ref -= 1
       
        # get index of neurons in refractory period
        idx_ref = np.where(self.S_.is_ref > 0)[0]
        # Set V_m to V_reset for neurons in refractory period
        # as V_m might have changed due to synaptic inputs
        # V_m at t+dt will continue at V_reset if neuron is still refractory at time t+net.dt
        self.S_.V_m[idx_ref] = self.P_.V_reset

        # test threshold crossing at time t+dt
        # idx_spiked = np.where(self.S_.V_m >= self.P_.V_th)[0] (deterministic case)
        # Calculate Phi for all neurons
        phi = self.phi()
        # draw an array of uniform random number and test which neuron spiked
        U_i = self.rng.random(self.N) # needs numpy > 1.17
#         U_i = np.random.uniform(size=self.N)
        idx_spiked = np.where(phi >= U_i)[0]
        
        # set refractory flag; note that is_ref is already at time t+dt
        self.S_.is_ref[idx_spiked] = self.V_.ref_count
        # Reset membrane potential in synaptic input of neurons that spiked.
        # reset of synaptic input is not present in the original PD model.
        self.S_.V_m[idx_spiked] = self.P_.V_reset
        self.S_.I_syn_ex[idx_spiked] = 0.0
        self.S_.I_syn_in[idx_spiked] = 0.0

        # Send spike event
        # id_0 contains the neuron id previous to the first neuron of the layer
        # thus, the true neuron id is the index in the array + id_0 + 1
        # because neuron id starts at 1, however index starts at 0
        self.net.send_spike(idx_spiked + self.id_0 + 1)   
        
            
class Network:
    layer = []
    spk_time = np.array([])
    spk_neuron = np.array([])
    
    class Connection: # class storing connection dictionary
        conn = {}
        
        def __init__(self, source, target, weight, delay):
            self.add_connection(source, target, weight, delay)
            
        # key of the dictionary is the neuron id
        # members is also a dictionary containing arrays of
        # target, weight and delay
        # it accepts as function input an array of all variables
        # where each element is alinged or
        # lists of arrays for target, weight and delay containning
        # the values for each source on each row
        def add_connection(self, source, target, weight, delay):
            if type(source) == int:
                if type(target) == int:
                    assert type(weight) == type(delay) == float
                else:
                    assert len(target) == len(weight) == len(delay)
                if source in conn.keys():
                    self.conn[source]['target'] = np.append(self.conn[source]['target'], target)
                    self.conn[source]['weight'] = np.append(self.conn[source]['weight'], weight)
                    self.conn[source]['delay'] = np.append(self.conn[source]['delay'], delay)
                else:
                    self.conn.update({source:{'target':np.array(target),
                                              'weight':np.array(weight),
                                              'delay':np.array(delay)}})
            elif type(source) == tuple or type(source) == list or type(source) == np.ndarray:
                assert len(source) == len(target) == len(weight) == len(delay)
                for idx, s_ in enumerate(source):
                    if s_ in self.conn.keys():
                        self.conn[int(s_)]['target'] = np.append(
                            self.conn[int(s_)]['target'], target[idx])
                        self.conn[int(s_)]['weight'] = np.append(
                            self.conn[int(s_)]['weight'], weight[idx])
                        self.conn[int(s_)]['delay'] = np.append(
                            self.conn[int(s_)]['delay'], delay[idx])
                    else:
                        self.conn.update({int(s_):{'target':np.array(target[idx]),
                                                   'weight':np.array(weight[idx]),
                                                   'delay':np.array(delay[idx])}})
            
            
    def __init__(self, dt=0.1, N=[], fname='spike_recorder.txt', seed_phi=1234, seed_poisson=2345):
        self.t = 0.0
        self.dt = dt
        self.N = np.array([])
        self.create_layer(N)
        self.C_ = None
        self.fname = fname
        self.seed_phi = seed_phi
        self.seed_poisson = seed_poisson
        
    
    # create layers. to change a neuron parameter, the layer method 'set_neuron_params' must be called
    # multiples layer can be created by providing a list containing number of neurons per layer
    def create_layer(self, N=[]):
        if type(N) == int:
            assert N > 0
            if len(self.N):
                self.layer.append(Layer(self, N, int(np.cumsum(self.N)[-1]),
                                        seed_phi=self.seed_phi+len(self.N),
                                        seed_poisson=self.seed_poisson+len(self.N)))
            else:
                self.layer.append(Layer(self, N, 0,
                                        seed_phi=self.seed_phi+len(self.N),
                                        seed_poisson=self.seed_poisson+len(self.N)))
            self.N = np.append(self.N, N)
        else:
            for n_ in N:
                assert n_ > 0
                if len(self.N):
                    self.layer.append(Layer(self, n_, int(np.cumsum(self.N)[-1])))
                else:
                    self.layer.append(Layer(self, n_, 0))
                self.N = np.append(self.N, n_)
        self.N_layer = len(self.N)
        self.N_cumsum = np.cumsum(self.N)
        self.last_neuron_id = self.N_cumsum[-1]
        
    # create connection dictionary given list of sources and
    # list of arrays containing targets, weights and delays for each source.
    # See Connection above for detail
    def create_connection(self, source, target, weight, delay):
        self.C_ = self.Connection(source, target, weight, delay)
    
    # create connection from conn_params dict
#    def create_connection(self, conn_params={}):

    # send spike event to the layer containing the target neuron.
    # buffer of the target neuron at the given delay is updated.
    # note that index 0 of the buffer is at time t+dt
    # and the spike event is also at time t+dt, thus, a delay of dt indicates
    # buffer position at 1
    def send_spike(self, id_):
        for idx, source_id in enumerate(id_):
            if self.C_:
                for jdx, target_id in enumerate(self.C_.conn[source_id]['target']):
                    ldx = np.where(target_id <= self.N_cumsum)[0][0]
                    self.layer[ldx].receive_spike(target_id, self.C_.conn[source_id]['weight'][jdx],
                                                  self.C_.conn[source_id]['delay'][jdx])
            self.spike_recorder.write('{:6d} {:12.1f}\n'.format(source_id, self.t))
            self.spk_neuron = np.append(self.spk_neuron, source_id)
            self.spk_time = np.append(self.spk_time, self.t)
#             print(source_id, self.t)
                
    # simulate network for time (ms)
    # if simulate is called consecutively data is appended to file.
    # this may allow changing network parameters during simulation
    def simulate(self, time):
        if self.t==0:
            self.spike_recorder = open(self.fname,'w')
        else:
            self.spike_recorder = open(self.fname,'a')
        while (self.t <= time):
            # evolve time
            self.t += self.dt
            # evolve membrane potentials of eache layer
            for layer_ in self.layer:
                layer_.evaluate()
            # shift synaptic input buffer of each layer if there is synaptic connection.
            # Conditions where there is no synaptic connection: simulate N trials of a single neuron
            if self.C_:
                for layer_ in self.layer:
                    layer_.shift_buffer()
        self.spike_recorder.close()
        
                
