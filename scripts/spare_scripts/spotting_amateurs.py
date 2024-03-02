#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:24:52 2022

@author: karthikeyarkaushik
"""

import numpy as np
import pandas as pd
from config import *
import pdb
import os

if __name__ == '__main__':
    # pick out a few interesting amateurs
    stats_df = pd.read_csv('results/stats.csv', index_col = 0)
    individual_influence = np.zeros((NUM_TOTAL,))
    
    for id_k, k_val in enumerate(PARAMETERS['k']):
        for id_r, rho_val in enumerate(PARAMETERS['rho']):    
            filename_id = id_k*len(PARAMETERS['rho']) + id_r
            adjacency = pd.DataFrame(np.genfromtxt(os.path.join('results', 'simulations',
                                                                GROUP, 'adjacencies',
                                                                str(filename_id) + '.csv')), 
                                     columns = ['from', 'to', 'value'])
            adjacency = pd.pivot(adjacency, index='from', columns='to', values='value').to_numpy()
            # # normalise by rows - rowval = rowval/rowsum
            adjacency = adjacency / np.nansum(adjacency, axis=1, keepdims=True)
            individual_influence = individual_influence + np.sum(adjacency, axis=0)
    individual_influence = individual_influence/np.sum(individual_influence)
    stats_df['influence'] = individual_influence
    stats_df['bang_for_buck'] = individual_influence/stats_df['ratings']
    stats_df.to_csv('results/stats.csv')
    
