#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:15:59 2022

@author: karthikeyarkaushik

Run analysis on particular wines, chosen for reasons of popularity/price etc.
1. Bernadotte 2007, #192
2. Leoville barton 2013, #1217
3. Pontet Canet 2008, #1623
4. du Tertre 2004, #1860
"""

from src.strategies import naive_knn_rep, baseline, weighted_averaging, single_wine_rep
from src.data_processing import merge_data, combine_adj
from config import NUM_EXPERTS, NUM_AMATEURS, NUM_REPS, PARAMETERS, COMPUTATION, GROUP, STRATEGY
import numpy as np
import pandas as pd
import os.path
import warnings
warnings.filterwarnings('ignore')
import pdb


if __name__ == '__main__':
    [dataframe, num_experts, num_amateurs] = merge_data()
    wine_IDS = [192, 1217, 1623, 1860]
    storage_path = 'results/simulations/supplementary/single_wines'
    # only one strategy, for k = 1, rho = 1
    for wine_ID in wine_IDS:
        all_adjacency = [0] 
        for rep in range(NUM_REPS):
            print(rep)
            if STRATEGY == 'naive_knn':
                performance, adjacency = single_wine_rep(dataframe, wine_ID)                            
                all_adjacency[0] = combine_adj(current_adj=adjacency[0], master_adj=all_adjacency[0])
        np.savetxt(os.path.join(storage_path, 'adjacencies_' + str(wine_ID) + '.csv'), all_adjacency[0])
    
#array([  0,   1,   2,   3,   7,   9,  12,  25,  27,  45,  50,  79,  82,
#        93,  95,  96, 129, 133]),)
# temp = all_adjacency[0][all_adjacency[0][:,2] > 0,:]
# np.unique(temp[:,1])