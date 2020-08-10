from network_params import net_dict
from sim_params import sim_dict
import numpy as np

def PD_microcircuit(seed=0):
    # Create random number generator:
    rng = np.random.default_rng(seed=seed)
    # store cumulative sum of number of neurons appended by initial 0
    N_cumsum = np.append(0, net_dict['N_full'].cumsum())
    # create 2D array with neuron id range per layer
    id_range = np.stack((N_cumsum[:-1], N_cumsum[1:]),axis=-1)
    # connection dict outdegree
    conn_dict = [[{} for i in range(len(net_dict['N_full']))]\
                 for j in range(len(net_dict['N_full']))]
    '''
    conn_dict = {i:{'target':np.array([]),
                    'weight':np.array([]),
                    'delay':np.array([])}
                 for i in range(N_cumsum[-1])}
    '''
    '''
    # connection dict indegree
    conn_dict_in = {i:{'source':np.array([]),
                       'weight':np.array([]),
                       'delay':np.array([])}
                    for i in range(N_cumsum[-1])}
    '''
    for (j,i),Ca in np.ndenumerate(net_dict['conn_probs']):
        print(i,j,Ca)
        # number of pre-synaptic neurons
        Npre = net_dict['N_full'][i]
        # number of post-synaptic neurons
        Npost = net_dict['N_full'][j]
        # total number of synaptic connections
        K = int(np.log(1-Ca)/np.log(1-(1/(Npre*Npost))))
        if i%2 == 0: # pre is excitatory
            w_mean = net_dict['PSP_e'] # mean excitatory synaptic weight
            w_sd = w_mean*net_dict['PSP_sd'] # standard deviation of synaptic weight
            d_mean = net_dict['mean_delay_exc'] # mean excitatory synaptic delay
            d_sd = d_mean*net_dict['rel_std_delay'] # standard deviation of synaptic delay
        else: # pre is inhibitory
            w_mean = net_dict['g']*net_dict['PSP_e'] # mean inhibitory synaptic weight
            w_sd = -w_mean*net_dict['PSP_sd'] # standard deviation of synaptic weight
            d_mean = net_dict['mean_delay_inh'] # mean inhibitory synaptic delay
            d_sd = d_mean*net_dict['rel_std_delay'] # standard deviation of synaptic delay
        if (i==2) and (j==0): # from L4e to L23e, double w
            w_mean *= 2.0 # synaptic weight from L4E to L23E is doubled
            w_sd = w_mean*net_dict['PSP_sd']

        # generate random list of pre-synaptic neurons
        '''
        pre_list = rng.integers(id_range[i][0],
                                high=id_range[i][1],
                                size=K)
        '''
        conn_dict[j][i]['source'] = rng.integers(id_range[i][0],
                                                 high=id_range[i][1],
                                                 size=K)
        pre_list = conn_dict[j][i]['source']

        # generate random list of post-synaptic neurons
        '''
        post_list = rng.integers(id_range[j][0],
                                 high=id_range[j][1],
                                 size=K)
        '''
        conn_dict[j][i]['target'] = rng.integers(id_range[j][0],
                                                 high=id_range[j][1],
                                                 size=K)
        post_list = conn_dict[j][i]['target']

        # generate truncated normal distributed weight array
        # the truncation algorithm is not the best solution,
        # alternative method is provided by scipy
        # weight_list = rng.normal(loc=w_mean, scale=w_sd, size=K)
        conn_dict[j][i]['weight'] = rng.normal(loc=w_mean, scale=w_sd, size=K)
        weight_list = conn_dict[j][i]['weight']
        # index of weight <= 0. All synapses must be active
        if w_mean>0: # excitatory
            idx_redraw = np.where(weight_list<=0)[0]
        else: # inhibitory
            idx_redraw = np.where(weight_list>=0)[0]
        # re-draw number and replace in weight_list until all weights are truncated at zero (exclusive)
        while len(idx_redraw)>0:
            weight_list[idx_neg] = rng.normal(loc=w_mean,
                                              scale=w_sd,
                                              size=len(idx_neg))
            if w_mean>0: # excitatory
                idx_redraw = np.where(weight_list<=0)[0]
            else: # inhibitory
                idx_redraw = np.where(weight_list>=0)[0]
                
        # generate truncated normal distributed delay array
        # the truncation algorithm is not the best solution,
        # alternative method is provided by scipy
        # delay_list = rng.normal(loc=d_mean, scale=d_sd, size=K)
        conn_dict[j][i]['delay'] = rng.normal(loc=d_mean, scale=d_sd, size=K)
        delay_list = conn_dict[j][i]['delay']
        # index of delays < dt (there must have a delay of at least one simulation step.
        # No spike transmission is spontaneous
        idx_redraw = np.where(delay_list<sim_dict['dt'])[0]
        # re-draw number and replace in delay_list untill all delays are >= dt
        while len(idx_redraw)>0:
            delay_list[idx_redraw] = rng.normal(loc=d_mean,
                                             scale=d_sd,
                                             size=len(idx_redraw))
            idx_redraw = np.where(delay_list<sim_dict['dt'])[0]

        # add connection to dict
        # first get unique sources
        '''
        u_keys = np.unique(pre_list)
        for idx,i_pre in enumerate(u_keys):
            print('{}/{}'.format(idx,len(u_keys)),end='\r')
            conn_dict[i_pre]['target'] = np.concatenate(
                (conn_dict[i_pre]['target'], post_list[pre_list==i_pre]))
            conn_dict[i_pre]['weight'] = np.concatenate(
                (conn_dict[i_pre]['weight'], weight_list[pre_list==i_pre]))
            conn_dict[i_pre]['delay'] = np.concatenate(
                (conn_dict[i_pre]['delay'], delay_list[pre_list==i_pre]))
        '''

        '''
        # do the same but now with target as key; create an indegree connection dict; for test purpose. may not be necessary
        u_keys = np.unique(post_list)
        for i_post in u_keys:
            conn_dict_in[i_post]['source'] = np.concatenate(
                (conn_dict_in[i_post]['source'], pre_list[post_list==i_post]))
            conn_dict_in[i_post]['weight'] = np.concatenate(
                (conn_dict_in[i_post]['weight'], weight_list[post_list==i_post]))
            conn_dict_in[i_post]['delay'] = np.concatenate(
                (conn_dict_in[i_post]['delay'], delay_list[post_list==i_post]))
        '''        
            
    return conn_dict #, conn_dict_in
