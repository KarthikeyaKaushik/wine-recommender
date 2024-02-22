#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions in this module are called before and after the actual computations
merge_data :
    Used to combine expert and amateur datasets, and get rid of raters with only nans
gen_traintest_split : 
    Split a dataset into train and test for a given mode of computation
post_processing : 
    The get_results function in compute.py returns a list of errors and adjacencies 
combine_adj :
    integrates adjacency matrix (in longform) over all repetitions
"""
import pandas as pd
import numpy as np
import random
import pdb
from src.config import SPLIT_HOW, HOLDOUT, NUM_TOTAL

def get_mask(indices, row_shape):
    '''
    Given a row_shape, fill up the array with 1s in the indices, and nans otherwise
    '''
    mask = np.empty(row_shape)
    mask[:] = np.nan
    mask[indices] = 1
    return mask

def merge_data():
    '''
    Merges the expert and amateur datasets.
    inputs have to be of shape :
            items x raters
    Assume the data is in ../data/ and named experts.csv, amateurs.csv (check if headers)
    Return merged dataset, number of experts and number of amateurs
    '''
    experts = pd.read_csv("data/experts.csv",index_col=0).to_numpy()
    amateurs = pd.read_csv("data/amateurs.csv", index_col = 0).to_numpy()
    assert experts.shape[0] == amateurs.shape[0]
    # check if there are any raters with nans, get them out
    nans_experts = np.sum(~np.isnan(experts), axis = 0) # sum to see how many non-nans
    nans_amateurs = np.sum(~np.isnan(amateurs), axis = 0) 
    if (nans_experts.any() == 0) and (nans_amateurs.any() == 0): 
        # if no nans, return merged value, with num experts and num amateurs
        return [np.hstack([experts, amateurs]).T, experts.shape[1], amateurs.shape[1]]
    else :
        if (len(np.where(nans_experts == 0)[0].tolist()) > 0):
            experts = np.delete(experts, np.where(nans_experts == 0), axis=1)
        if (len(np.where(nans_amateurs == 0)[0].tolist()) > 0):
            amateurs = np.delete(amateurs, np.where(nans_amateurs == 0), axis=1)
        return [np.hstack([experts, amateurs]).T, experts.shape[1], amateurs.shape[1]]

def gen_traintest_split(data_mat, computation='performance'):
    '''
    Generating the test and train data from the combined data matrix
    Use the split_how argument (either one out or given n) to split data_mat 
    into two parts. The matrices are of equal size, the only change being the 
    test and train's intersection is Null
    data_mat has shape n x m, 
    where n - number of agents, m is number of items
    To test performance, we split the data based on what items have been evaluated.
    That is, if we are leaving 10 out of the total items, they will be left out from
    the items that have a non-nan value. If not, they will be taken out from all 
    available indices.
    '''
    # create empty splits with identical shapes
    assert all(np.sum(~np.isnan(data_mat), axis=1) > HOLDOUT)  # number of items > split
    data_train, data_test = np.empty(data_mat.shape), np.empty(data_mat.shape)
    data_train[:,:] = np.float('NaN')
    data_test[:,:] = np.float('NaN')
    for row in range(data_mat.shape[0]):
        row_data = data_mat[row,:]
        if computation == 'performance': # split only on evaluated items
            rating_indices = np.where(~np.isnan(row_data))[0].tolist()
        else: # split on any item
            rating_indices = list(range(0,row_data.shape[0]))
        # get split number of these indices, put them into test_indices, rest into train_indices
        if SPLIT_HOW == 'one_out':
            test_indices = random.sample(rating_indices, HOLDOUT)
            train_indices = [idx for idx in rating_indices if idx not in test_indices]
        else:
            train_indices = random.sample(rating_indices, HOLDOUT)
            test_indices = [idx for idx in rating_indices if idx not in train_indices]
        # now put all these train_index values in data_train
        data_train[row,:] = row_data * get_mask(train_indices,row_data.shape) 
        if computation == 'performance':
            data_test[row,:] = row_data * get_mask(test_indices,row_data.shape) 
        else:
            data_test[row,:] = get_mask(test_indices,row_data.shape) 
    return data_train, data_test



def post_processing(all_outputs):
    '''
    Split up the collected (from parallel processes) outputs into errors and adjacencies
    '''
    errors = np.zeros((len(all_outputs),))
    errors[:] = np.nan
    adjacency_df = pd.DataFrame(columns = ['from_node','to_node','weight'])
    for ind,user in enumerate(all_outputs): # run through all users
        errors[ind] = user[0]
        user_adjacency = user[1]
        user_adjacency = pd.DataFrame(data=user_adjacency)
        user_adjacency.columns = ['from_node','to_node','weight']
        adjacency_df = adjacency_df.append(user_adjacency)
    adjacency_df = adjacency_df.to_numpy()
    return errors, adjacency_df



def combine_adj(current_adj, master_adj):
    if type(master_adj) == int: # for the first case
        arr = current_adj
        rows = arr[:,0].astype('int')
        cols = arr[:,1].astype('int')
        pivoted_adj = np.zeros((NUM_TOTAL, NUM_TOTAL))
        pivoted_adj[rows, cols] = arr[:,2]
        rows = np.repeat(list(range(NUM_TOTAL)), NUM_TOTAL, axis=0)
        cols = np.tile(list(range(NUM_TOTAL)), NUM_TOTAL)
        vals = pivoted_adj.reshape((NUM_TOTAL*NUM_TOTAL,))
        pivoted_adj = np.vstack([rows, cols, vals]).T
        return pivoted_adj
    else:
        m_arr, c_arr = master_adj, current_adj
        m_rows, m_cols = m_arr[:,0].astype('int'), m_arr[:,1].astype('int')
        c_rows, c_cols = c_arr[:,0].astype('int'), c_arr[:,1].astype('int')
        piv_master, piv_curr = np.zeros((NUM_TOTAL, NUM_TOTAL)), np.zeros((NUM_TOTAL, NUM_TOTAL))
        piv_master[m_rows, m_cols] = m_arr[:,2]
        piv_curr[c_rows, c_cols] = c_arr[:,2]
        piv_master = piv_master + piv_curr
        rows = np.repeat(list(range(NUM_TOTAL)), NUM_TOTAL, axis=0)
        cols = np.tile(list(range(NUM_TOTAL)), NUM_TOTAL)
        vals = piv_master.reshape((NUM_TOTAL*NUM_TOTAL,))
        piv_master = np.vstack([rows, cols, vals]).T
        return piv_master



