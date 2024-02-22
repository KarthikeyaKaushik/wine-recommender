#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:30:24 2022

@author: karthikeyarkaushik
"""
import numpy as np
import pandas as pd
from config import *
from src.data_processing import merge_data
import pdb

if __name__ == '__main__':
    # Get essential stats about simulated results
    stats_df = pd.DataFrame()
    stats_df['e_a'] = ['expert' for i in range(NUM_EXPERTS)] + \
                        ['amateur' for i in range(NUM_AMATEURS)]
    dataframe, _, _ = merge_data()
    corr_matrix = pd.DataFrame(np.transpose(dataframe)).corr(min_periods=MIN_OVERLAP).to_numpy()
    np.fill_diagonal(corr_matrix, float('NaN'))
    stats_df['mean_corr'] = np.nanmean(corr_matrix, axis=1)
    stats_df['disp_corr'] = np.nanstd(corr_matrix, axis=1)
    temp = np.nanvar(corr_matrix, axis=1)
    stats_df['ratings'] = np.sum(dataframe == dataframe, axis=1)
    stats_df.to_csv('results/stats.csv')
    pd.DataFrame(corr_matrix).to_csv('results/correlation.csv')
    pdb.set_trace()