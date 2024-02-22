#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is the workhorse containing essential computation functions
get_correlation : 
    takes in the correlation matrix and user, 
    returns the ordered correlation values and indices of those values
get_row_prediction :
    for a given item,and rater, get the prediction as sum of the 
    relevant training set times the relevant weighting divided by the sum of relevant weighting
get_prediction :
    for a given set of items and rater, first get the weighting (ordererd correlation^rho)
    then get a row prediction for every item in that set of items based on this weighting
get_homophily_individual :
    Get the homophily index of an individual, and also the baseline associated with them
    The homophily index should approach the baseline as k tends to the maximum number of influencers
get_homophily_group :
    Similar to above, just run on the entire group at once
"""

import pandas as pd
import numpy as np
from src.config import *
import pdb


def get_correlation(user, corr_mat, group='both'):
    '''
    Use the corr_mat, and the user index to get the arranged correlations 
    w.r.t other raters corr_mat has to be a 2D numpy array.
    Set self correlation to nan
    Set nan correlation to mean correlation
    Set self correlation to -inf
    Set correlations to -inf depending upon the group
    Arrange correlations, and return indices of the arranged values
    '''
    agent_correlations = corr_mat[user,:].copy() 
    agent_correlations[user] = np.float('Nan') # first set to nan so as to not mess up the average correlations
    agent_correlations[np.isnan(agent_correlations)] = np.nanmean(agent_correlations) # make the unknowns average instead of "unknown"
    agent_correlations[user] = -np.float('Inf') # set self correlation to -Inf to set to bottom of ranked list 
    # set all the correlations with the non target group to -np.float('Inf')
    if group == 'experts':
        agent_correlations[NUM_EXPERTS:] = -np.float('Inf')
    elif group == 'amateurs':
        agent_correlations[0:NUM_EXPERTS] = -np.float('Inf')
    assert sum(np.isnan(agent_correlations)) == 0
    agent_values = np.flip(np.sort(agent_correlations))
    agent_ranks = np.flip(np.argsort(agent_correlations))
    agent_values = np.expand_dims(np.array(agent_values),axis=1)
    agent_ranks = agent_ranks.tolist() 
    return [agent_values,agent_ranks] 


def get_row_prediction(idx, recos, group, 
                       sorted_ord, sorted_vals, 
                       np_test_data, mask_w, kval, rho, user, weighting,
                       reason='performance'):
    '''
    Get prediction given item idx 
    See get_prediction for general idea - it calls this for a particular index
    '''
    predicted_row = np_test_data[:,idx] # this is the entire test data on item = idx
    mask_r = ~np.isnan(predicted_row)
    mask_total = mask_w & mask_r # get total valid mask to apply on both test data and weighting
    if reason == 'performance':
        try: # to fill the recos with top k values, if not enough k values, fill anyway, also default behavior
            prediction = np.dot(predicted_row[mask_total][0:kval], weighting[mask_total][0:kval])/sum(abs(weighting[mask_total][0:kval]))
        except:
            prediction = np.dot(predicted_row[mask_total], weighting[mask_total])/sum(abs(weighting[mask_total]))
            # if the K_VALs aren't possible to achieve, prediction becomes nan.
        if np.isnan(prediction):
            if group == 'experts':
                prediction = np.nanmean(predicted_row[mask_total][0:kval])
            elif group == 'amateurs':
                prediction = np.nanmean(predicted_row[mask_total][0:kval])
            else:
                prediction = np.nanmean(predicted_row[mask_total][0:kval])
        if np.isnan(prediction):
            prediction = 0.0
        assert ~np.isnan(prediction)
    else :
        prediction = 1.0
    if np.sum(mask_total) > 0: # If there are items that can be recommended, and non-nan weights
        temp = np.zeros(mask_total.shape) # temp contains all those places mask != nan
        temp[np.where(1*mask_total == 1)[0][:kval]] = 1 # mask all nans, and choose only top kval of those masks
        # weighting to be added - first normalise and then add -
        normed_weighting = np.nan_to_num(weighting) * temp
        recos = recos + normed_weighting # add the weighting to the recos
    else: # if there are no items to be recommended, but there are non-nan weights
        temp = np.zeros(mask_w.shape) # temp contains all those places mask != nan
        temp[np.where(1*mask_w == 1)[0][:kval]] = 1 # mask all nans, and choose only top kval of those masks
        normed_weighting = np.nan_to_num(weighting) * temp
        normed_weighting = normed_weighting / np.sum(normed_weighting)
        recos = recos + normed_weighting # add the weighting to the recos
    return prediction, recos


def get_prediction(sorted_vals,sorted_ord,user,
                   test_data,train_data,rho,kval,
                   group,reason=COMPUTATION):
    '''
    Use the train_data values of all agents to get a prediction for test_data 
    where values are not nan. Putting get_row_prediction in another block
    '''    
    weighting = sorted_vals[:,0] # get weight to multiply rating with
    # wherever weighting is 0, set it to 0 after raising to rho
    where_zeros = weighting < -1 # this only sets -Inf to 0
    where_nans = (weighting < 0) & (weighting >= -1)
    weighting = weighting ** rho
    weighting[where_nans] = 0.0
    weighting[where_zeros] = 0.0
    # set negatives to nans    
    mask_w = ~np.isinf(sorted_vals[:,0]) # getting rid of -inf correlations
    predictions = np.zeros((test_data.shape[1],))
    target = test_data[user,:]
    target_indices = np.where(~np.isnan(target))
    # rearrange test data according to priority
    # TODO : See if np.ix_ works better?
    np_test_data = train_data[sorted_ord,:] 
    recos = np.zeros(test_data.shape[0])
    for idx in target_indices[0]: # go through all those items where user has non zero rating in test data
        prediction, recos = get_row_prediction(idx,recos,group,sorted_ord, sorted_vals,
                                               np_test_data, mask_w, kval, rho, user, weighting, 
                                               reason)    
        predictions[idx] = prediction
    longform = np.zeros((sum(recos>0),3))
    longform[:,0] = user
    idx_list = np.where(recos>0)[0].tolist()
    longform[:,1] = np.array(sorted_ord)[idx_list]
    longform[:,2] = recos[idx_list]#recos[recos>0]
    return predictions, target, longform



def get_homophily_individual(corr_matrix, rho_val, k_val, type_homophily = 'Real homophily', master_dataset = None):
    '''
    Given the adjacency matrix, rho_val and k_val, calculate the homophily index
    In case of potential homophily, the adjacency matrix is the correlation matrix. 
    Fill diagonals with 0s - (every row = og_row_array)
    For each row, replace nans by the average correlation with the rest of the agents
    Sort the correlation row (= row_sorted)
    Raise it to the power rho
    normalise to 1 (= weighting)
    pick out the top k indices in row_sorted
    '''
    if type_homophily == 'Potential homophily': 
        np.fill_diagonal(corr_matrix,0)
        stored_weights = []
        # pick out to k_val columns for every row, and . else let it be as it is
        for row_ind in range(corr_matrix.shape[0]):
            og_row_array = np.copy(corr_matrix[row_ind,:])
            positive_corr_mean = np.mean(og_row_array[og_row_array>0])
            og_row_array = np.nan_to_num(og_row_array, nan=positive_corr_mean, copy=True)
            og_row_array[og_row_array < 0] = 0.0 #positive_corr_mean
            row_sorted = np.flip(np.sort(og_row_array))
            where_zeros = og_row_array == 0
            weighting = og_row_array ** rho_val
            weighting[where_zeros] = 0.0
            weighting = weighting / np.sum(weighting)
            temp_weighting = np.zeros(weighting.shape)
            temp_weighting[og_row_array > row_sorted[k_val]] = weighting[og_row_array > row_sorted[k_val]]
            stored_weights.append(temp_weighting)
        corr_matrix = np.array(stored_weights)
            
    # calculate rating count baseline (not rater count baseline)
    rating_counts = np.sum(~np.isnan(master_dataset), axis=1)
    num_items = master_dataset.shape[1]
    train_set_realistic = [i * ((num_items - HOLDOUT)/num_items) for i in rating_counts]
    exp_df = pd.DataFrame(columns = ['group', 'kval', 'rhoval', 'Hi', 'method', 'baseline_1', 'baseline_2', 'rater'])
    for expert in range(0, NUM_EXPERTS):
        if type_homophily == 'Potential homophily':
            baseline= None
        else:
            baseline = (np.sum(train_set_realistic[:NUM_EXPERTS]) - rating_counts[expert])/(np.sum(train_set_realistic) - rating_counts[expert])
        s_expe = np.sum(np.sum(corr_matrix[expert, : NUM_EXPERTS]))
        d_expe = np.sum(np.sum(corr_matrix[expert, NUM_EXPERTS : ]))
        s_exp = s_expe / (s_expe + d_expe)

        s_exp = {'group': ['experts'], 'kval': [k_val], 'rhoval':[rho_val], 'Hi':[s_exp], 
                 'method':type_homophily, 'baseline_1' : (NUM_EXPERTS-1)/(NUM_TOTAL-1), 
                 'baseline_2' : baseline, 'rater':expert}
        temp_s_exp = pd.DataFrame(s_exp)
        exp_df = pd.concat([exp_df,temp_s_exp])
    
    ama_df = pd.DataFrame(columns = ['group', 'kval', 'rhoval', 'Hi', 'method', 'baseline_1', 'baseline_2', 'rater'])
    for amateur in range(NUM_EXPERTS,NUM_TOTAL):
        if type_homophily == 'Potential homophily':
            baseline = None
        else:
            baseline = (np.sum(train_set_realistic[NUM_EXPERTS:]) - rating_counts[amateur])/(np.sum(train_set_realistic) - rating_counts[amateur])
        s_amat = np.sum(np.sum(corr_matrix[amateur, NUM_EXPERTS : ]))
        d_amat = np.sum(np.sum(corr_matrix[amateur, : NUM_EXPERTS]))
        s_ama = s_amat / (s_amat + d_amat)
        s_ama = {'group': ['amateurs'], 'kval': [k_val], 'rhoval':[rho_val], 'Hi':[s_ama], 'method':type_homophily, 
                'baseline_1':(NUM_AMATEURS-1)/(NUM_TOTAL-1), 'baseline_2' : baseline, 'rater':amateur}
        temp_s_ama = pd.DataFrame(s_ama)
        ama_df = pd.concat([ama_df, temp_s_ama])
    
    homophily_df = pd.concat([exp_df, ama_df])
    return homophily_df




def get_homophily_group(corr_matrix, rho_val, k_val, type_homophily = 'Real homophily', master_dataset = None):
    '''
    See above for the individual case
    '''
    if type_homophily == 'Potential homophily': 
        np.fill_diagonal(corr_matrix,0)
        stored_weights = []
        # do some manipulation to corr_matrix. pick out to k_val columns for every row, and . else let it be as it is
        for row_ind in range(corr_matrix.shape[0]):
            og_row_array = np.copy(corr_matrix[row_ind,:])
            positive_corr_mean = np.mean(og_row_array[og_row_array>0])
            og_row_array = np.nan_to_num(og_row_array, nan=positive_corr_mean, copy=True)
            og_row_array[og_row_array < 0] = 0.0 #positive_corr_mean
            row_sorted = np.flip(np.sort(og_row_array))
            where_zeros = og_row_array == 0
            weighting = og_row_array ** rho_val
            weighting[where_zeros] = 0.0
            weighting = weighting / np.sum(weighting)
            temp_weighting = np.zeros(weighting.shape)
            temp_weighting[og_row_array > row_sorted[k_val]] = weighting[og_row_array > row_sorted[k_val]]
            stored_weights.append(temp_weighting)
        corr_matrix = np.array(stored_weights)
    
    # calculate rating count baseline (not rater count baseline)
    rating_counts = np.sum(~np.isnan(master_dataset), axis=1)
    num_items = master_dataset.shape[1]
    train_set_realistic = [i * ((num_items - HOLDOUT)/num_items) for i in rating_counts]

    if type_homophily == 'Potential homophily':
        baseline_2_e = None
        baseline_2_a = None
    else:
        baseline_2_e = (NUM_EXPERTS*np.sum(train_set_realistic[:NUM_EXPERTS]) - np.sum(rating_counts[:NUM_EXPERTS]))/ \
                        (NUM_EXPERTS*np.sum(train_set_realistic) - np.sum(rating_counts[:NUM_EXPERTS]))
        baseline_2_a = (NUM_AMATEURS*np.sum(train_set_realistic[NUM_EXPERTS:]) - np.sum(rating_counts[NUM_EXPERTS:]))/ \
                        (NUM_AMATEURS*np.sum(train_set_realistic) - np.sum(rating_counts[NUM_EXPERTS:]))
        
    # homophily index = si / (si + di)
    s_expe = np.sum(np.sum(corr_matrix[ : NUM_EXPERTS, : NUM_EXPERTS ]))
    d_expe = np.sum(np.sum(corr_matrix[ : NUM_EXPERTS, NUM_EXPERTS : ]))

    s_exp = s_expe / (s_expe + d_expe)
    s_exp = {'group': ['experts'], 'kval': [k_val], 'rhoval':[rho_val], 'Hi':[s_exp], 'method':type_homophily,
            'baseline_1' : (NUM_EXPERTS-1)/(NUM_TOTAL-1), 'baseline_2' : baseline_2_e }

    s_amat = np.sum(np.sum(corr_matrix[NUM_EXPERTS : , NUM_EXPERTS : ]))
    d_amat = np.sum(np.sum(corr_matrix[NUM_EXPERTS : , : NUM_EXPERTS ]))

    s_ama = s_amat / (s_amat + d_amat)
    a_ihi = (s_ama - (NUM_AMATEURS/NUM_TOTAL)) / (1 - (NUM_AMATEURS/NUM_TOTAL))
    s_ama = {'group': ['amateurs'], 'kval': [k_val], 'rhoval':[rho_val], 'Hi':[s_ama], 'IHi':[a_ihi], 'method':type_homophily, 
            'baseline_1':(NUM_AMATEURS-1)/(NUM_TOTAL-1), 'baseline_2' : baseline_2_a}
    return s_exp, s_ama    



def get_influence(adj_matrix, rho_val, k_val, type_influence = 'Real influence', master_dataset = None):
    """
    Get the influence exerted by every individual in a group on average (actual and potential).

    Args:
        adj_matrix (np array): The adjacency matrix.
        rho_val (float): weighting
        k_val (int): nearest neighbours
        type_influence (string): influence type - real or potential
        master_dataset (np array): full dataset

    Returns:
        dictionary with influence by group

    """
    if type_influence == 'Potential influence': 
        # drop the nan row
        adj_matrix = adj_matrix[np.sum(~np.isnan(adj_matrix), axis=1) > 0,:]
        adj_matrix = adj_matrix[:, np.sum(~np.isnan(adj_matrix), axis=0) > 0]
        np.fill_diagonal(adj_matrix,0)
        stored_weights = []
        # do some manipulation to adj_matrix. pick out to k_val columns for every row, and . else let it be as it is
        for row_ind in range(adj_matrix.shape[0]):
            og_row_array = np.copy(adj_matrix[row_ind,:])
            positive_corr_mean = np.mean(og_row_array[og_row_array>0])
            og_row_array = np.nan_to_num(og_row_array, nan=positive_corr_mean, copy=True)
            og_row_array[og_row_array < 0] = positive_corr_mean
            row_sorted = np.flip(np.sort(og_row_array))
            where_zeros = og_row_array == 0
            weighting = og_row_array ** rho_val
            weighting[where_zeros] = 0.0
            weighting = weighting / np.sum(weighting)
            temp_weighting = np.zeros(weighting.shape)
            temp_weighting[og_row_array > row_sorted[k_val]] = weighting[og_row_array > row_sorted[k_val]]
            stored_weights.append(temp_weighting)
        adj_matrix = np.array(stored_weights)
    
    # calculate rating count baseline (not rater count baseline)
    rating_counts = np.sum(~np.isnan(master_dataset), axis=1)
    if type_influence == 'Potential influence':
        baseline_2_e = None
        baseline_2_a = None
    else:
        baseline_2_e = np.sum(rating_counts[0:NUM_EXPERTS])/np.sum(rating_counts)
        baseline_2_a = np.sum(rating_counts[NUM_EXPERTS:])/np.sum(rating_counts)
    
    table_h_stats = {}
    table_h_stats['Ni'] = [NUM_EXPERTS, NUM_AMATEURS] # alright, do better later
    table_h_stats = pd.DataFrame(table_h_stats)
    table_h_stats['Wi'] = table_h_stats['Ni'] / sum(table_h_stats['Ni'])
    
    expert_influencer_proportion = np.sum(np.sum(adj_matrix, axis=0)[:NUM_EXPERTS])
    amateur_influencer_proportion = np.sum(np.sum(adj_matrix, axis=0)[NUM_EXPERTS:])
    total_influence = np.sum(np.sum(adj_matrix))
        
    s_exp = {'group': ['experts'], 'kval': [k_val], 'rhoval':[rho_val], 
             'proportion_per_capita':[(expert_influencer_proportion/total_influence)/NUM_EXPERTS], 
             'proportion_gross':[expert_influencer_proportion/total_influence], 'method':type_influence,
            'baseline_1' : 1/(NUM_TOTAL-1), 'baseline_2' : baseline_2_e }
    
    s_ama = {'group': ['amateurs'], 'kval': [k_val], 'rhoval':[rho_val], 
             'proportion_per_capita':[(amateur_influencer_proportion/total_influence)/NUM_AMATEURS], 
             'proportion_gross':[amateur_influencer_proportion/total_influence], 'method':type_influence, 
            'baseline_1':1/(NUM_TOTAL-1), 'baseline_2' : baseline_2_a}
    return s_exp, s_ama








