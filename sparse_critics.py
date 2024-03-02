#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:51:46 2022

@author: karthikeyarkaushik
"""

from config import NUM_EXPERTS, NUM_AMATEURS, NUM_REPS, PARAMETERS, COMPUTATION, GROUP, STRATEGY, NUM_TOTAL, MIN_OVERLAP
from src.data_processing import merge_data
import numpy as np
import pandas as pd
import os.path
import warnings
warnings.filterwarnings('ignore')
import pdb
from random import sample
SPARSE_REPS = 100

if __name__ == '__main__':
    storage_path = 'results/simulations/supplementary'
    [dataframe, num_experts, num_amateurs] = merge_data()
    assert (num_experts == NUM_EXPERTS) and (num_amateurs == NUM_AMATEURS)


    amateur_density = np.sum(~np.isnan(dataframe[NUM_EXPERTS:,:])) / (dataframe.shape[1] * NUM_AMATEURS)
    amateur_correlations = pd.DataFrame(np.transpose(dataframe[NUM_EXPERTS:,])).corr(min_periods=MIN_OVERLAP).to_numpy()

    sparse_correlations_experts = np.empty((SPARSE_REPS, NUM_EXPERTS, NUM_EXPERTS))
    sparse_correlations_together = np.empty((SPARSE_REPS, NUM_TOTAL, NUM_TOTAL))

    for i in range(SPARSE_REPS):
        copied_dataframe = dataframe.copy()
        # for every expert, find current density, get number of ratings to remove, remove it
        for e in range(NUM_EXPERTS):
            expert_density = np.sum(~np.isnan(dataframe[e,:])) / dataframe.shape[1]
            num_to_remove = int((expert_density - amateur_density) * dataframe.shape[1])
            non_nans = np.where(~np.isnan(dataframe[e, :]))
            item_to_remove = sample(non_nans[0].tolist(), num_to_remove)
            copied_dataframe[e, item_to_remove] = np.float('NaN')
        correlation_together = pd.DataFrame(np.transpose(copied_dataframe)).corr(min_periods=MIN_OVERLAP).to_numpy()
        expert_correlations = pd.DataFrame(np.transpose(copied_dataframe[:NUM_EXPERTS,:])).corr(min_periods=MIN_OVERLAP).to_numpy()
        sparse_correlations_together[i, :, :] = correlation_together
        sparse_correlations_experts[i, :, :] = expert_correlations
    sparse_correlations_together = np.nanmean(sparse_correlations_together, axis=0)
    sparse_correlations_separate = np.zeros((NUM_TOTAL, NUM_TOTAL))
    sparse_correlations_separate[:NUM_EXPERTS, :NUM_EXPERTS] = np.nanmean(sparse_correlations_experts, axis=0)
    sparse_correlations_separate[NUM_EXPERTS:, NUM_EXPERTS:] = amateur_correlations
    np.savetxt(os.path.join(storage_path,
                    'sparse_separate.csv'), sparse_correlations_separate)
    np.savetxt(os.path.join(storage_path,
                    'sparse_together.csv'), sparse_correlations_together)    
        
    
    
    
    
    
