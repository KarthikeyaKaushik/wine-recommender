#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:46:08 2022
@author: karthikeyarkaushik
"""

from src.strategies import naive_knn_rep, baseline, weighted_averaging
from src.data_processing import merge_data, combine_adj
from src.config import NUM_EXPERTS, NUM_AMATEURS, NUM_REPS, PARAMETERS, COMPUTATION, GROUP, STRATEGY
import numpy as np
import os.path
import warnings
warnings.filterwarnings('ignore')
import pdb

if __name__ == '__main__':
    print("computation, group : ", COMPUTATION, GROUP)
    [dataframe, num_experts, num_amateurs] = merge_data()
    assert (num_experts == NUM_EXPERTS) and (num_amateurs == NUM_AMATEURS)
    # dataframe contains the merged dataset
    storage_path = 'results/simulations'
    # init a performances matrix to store all performances of shape - NUM_RATERS x NUM_STRATEGIES
    num_strategies = len(PARAMETERS['k'])* len(PARAMETERS['rho'])
    if STRATEGY == 'weighted_averaging':
        num_strategies = 1 * len(PARAMETERS['rho'])
    all_performance = np.zeros((NUM_EXPERTS + NUM_AMATEURS, 
                             num_strategies))
    if STRATEGY == 'baseline':
        all_performance = np.zeros((NUM_EXPERTS + NUM_AMATEURS,))

    # create empty list to store all adjacencies, update them as needed
    all_adjacency = [0 for i in range(num_strategies)] 
    for rep in range(NUM_REPS):
        if STRATEGY == 'naive_knn':
            performance, adjacency = naive_knn_rep(dataframe)
        elif STRATEGY == 'baseline':
            performance, adjacency = baseline(dataframe)
        elif STRATEGY == 'weighted_averaging':
            performance, adjacency = weighted_averaging(dataframe)
            
        if COMPUTATION == 'performance':
            all_performance = all_performance + performance
        else:
            for i in range(num_strategies):                
                all_adjacency[i] = combine_adj(current_adj=adjacency[i], master_adj=all_adjacency[i])
        print('Repetition : ', rep)
        
        
    if COMPUTATION == 'performance':
        if (STRATEGY == 'baseline') or (STRATEGY == 'weighted_averaging'):
            np.savetxt(os.path.join(storage_path, GROUP,
                                STRATEGY + '.csv'), all_performance/NUM_REPS)
        else:
            np.savetxt(os.path.join(storage_path, GROUP,
                            'performance.csv'), all_performance/NUM_REPS)
    else :
        for param in range(num_strategies):
            np.savetxt(os.path.join(storage_path, GROUP, 'adjacencies', str(param) + '.csv'), 
                      all_adjacency[param])

    
# temp = np.nanmean(performance, axis=0)
# t_k = [temp[i*7 : (i*7 + 7)] for i in range(17)]
# t_k = [np.around(t, 3) for t in t_k]
# t_k = [np.argmax(t) for t in t_k]






