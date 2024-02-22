#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions here are called from run.py

The naive_knn_rep is one repetition of the naive-knn algorithm for a given dataframe.
Its workflow is :
    - Split the dataset into train and test 
    - For every single combination of parameter values, and for every single rater,
    get the result of applying the algorithm on the raters test set.

The get_results module collects results for one rater, for one set of pararmeters
In get_results, 
    - obtain the ordering and values of correlations between given user and the rest
    - Make a prediction on the test items for the user based on the ordering, values, 
    and training set from the rest of the group
    - This prediction also gets a list of contributors (influencers), and the amount of
    influence they weild for that particular test set, and that particular rater
    - Compare the predicted value and target value, and get an error
"""
import pandas as pd
import numpy as np
from src.config import PARAMETERS, MIN_OVERLAP, COMPUTATION, GROUP, STRATEGY, NUM_TOTAL, NUM_AMATEURS, NUM_EXPERTS
from src.data_processing import gen_traintest_split, post_processing
from src.compute import get_correlation, get_prediction
from src.evaluate import compare_choices
import pdb

def get_results(user,corr_mat, test_data, train_data, rho,kval):
    '''
    Perform the steps one after another - 
    get sorted correlations
    get predictions
    get rmse on predictions
    '''
    sorted_values, sorted_order = get_correlation(user, corr_mat, GROUP)
    predictions,target,longform = get_prediction(sorted_values, sorted_order,
                                                 user,test_data,train_data,rho,
                                                 kval, GROUP)
    error = compare_choices(prediction=predictions, target=target)
    return [error, longform]


def naive_knn_rep(df):
    '''
    Every time this function is called, use the test strategy to first pick out training and test split
    Then use the parameters from the parameter set to calculate rmse and adjacency for all agents
    '''
    
    k_vals = PARAMETERS['k'] # [1,3,5...,29]
    rho_vals = PARAMETERS['rho']    # [0, .25, ... 1.5]
    grand_adjacencies = []
    train_df, test_df = gen_traintest_split(data_mat=df, computation=COMPUTATION)
    correlation_matrix = pd.DataFrame(np.transpose(train_df)).corr(min_periods=MIN_OVERLAP).to_numpy()
    performance_strategies = np.zeros((df.shape[0],len(k_vals)*len(rho_vals))) # shape - (NUM_TOTAL, num_strategies)
    # TODO : Change the order of user, k, rho to prevent recalc of corrs
    for id_k, k_val in enumerate(k_vals):
        for id_r, rho_val in enumerate(rho_vals):
            all_outputs = []
            for user in range(performance_strategies.shape[0]):
                all_outputs.append(get_results(user,correlation_matrix, 
                                               test_df, train_df, rho_val, k_val))
            all_rmses, all_adjacencies = post_processing(all_outputs) 
            performance_strategies[:,id_k*len(rho_vals) + id_r] = all_rmses
            grand_adjacencies.append(all_adjacencies)
    if COMPUTATION == 'performance':
        return performance_strategies, None
    else:
        return None, grand_adjacencies

def single_wine_rep(df, wine_ID):
    """
    Every time this function is called, only get adjacencies for particular wine_ID

    """
    grand_adjacencies = []
    train_df, test_df = gen_traintest_split(data_mat=df, computation=COMPUTATION)
    to_remove = [i for i in range(df.shape[1])]
    to_remove.remove(wine_ID)
    correlation_matrix = pd.DataFrame(np.transpose(train_df)).corr(min_periods=MIN_OVERLAP).to_numpy()
    all_outputs = []
    test_df[:, to_remove] = np.float('NaN')
    test_df[:, wine_ID] = 0.0 # just get a dummy rating in its place
    for user in range(df.shape[0]):
        all_outputs.append(get_results(user,correlation_matrix, 
                                       test_df, train_df, 1.0, 5))
    all_rmses, all_adjacencies = post_processing(all_outputs) 
    grand_adjacencies.append(all_adjacencies)
    return None, grand_adjacencies
        

def baseline(df):
    '''
    Baseline repetition with average prediction
    '''
    assert STRATEGY == 'baseline' and COMPUTATION == 'performance'
    k_vals = PARAMETERS['k'] # [1,3,5...,29]
    rho_vals = PARAMETERS['rho']    # [0, .25, ... 1.5]
    train_df, test_df = gen_traintest_split(data_mat=df, computation=COMPUTATION)
    performance_strategies = np.zeros((df.shape[0],len(k_vals)*len(rho_vals))) # shape - (NUM_TOTAL, num_strategies)
    # TODO : Change the order of user, k, rho to prevent recalc of corrs
    mean_preds = np.nanmean(train_df,axis=0)
    all_errors = []
    for user in range(performance_strategies.shape[0]):
        all_errors.append(compare_choices(prediction=mean_preds, target=test_df[user,:]))
    all_errors = np.array(all_errors)
    return all_errors, None
    
def weighted_averaging(df):
    '''
    Weighted averaging instead of simply baseline
    '''
    assert STRATEGY == 'weighted_averaging' and COMPUTATION == 'performance'
    if GROUP == 'both':
        k_vals = [NUM_TOTAL-1] # get for one kval = max number of raters
    elif GROUP == 'amateurs':
        k_vals = [NUM_AMATEURS] # get for one kval = max number of raters
    elif GROUP == 'experts':
        k_vals = [NUM_EXPERTS] # get for one kval = max number of raters
    rho_vals = PARAMETERS['rho']    
    train_df, test_df = gen_traintest_split(data_mat=df, computation=COMPUTATION)
    correlation_matrix = pd.DataFrame(np.transpose(train_df)).corr(min_periods=MIN_OVERLAP).to_numpy()
    performance_strategies = np.zeros((df.shape[0],len(k_vals)*len(rho_vals)))
    for id_k, k_val in enumerate(k_vals):
        for id_r, rho_val in enumerate(rho_vals):
            all_outputs = []
            for user in range(performance_strategies.shape[0]):
                all_outputs.append(get_results(user,correlation_matrix, 
                                               test_df, train_df, rho_val, k_val))
            all_rmses, _ = post_processing(all_outputs) 
            performance_strategies[:,id_k*len(rho_vals) + id_r] = all_rmses
    return performance_strategies, None
    










