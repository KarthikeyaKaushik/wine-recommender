#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions here are called from get_results in the strategies module.
compare_choices :
    From a list of predicted ratings, make all possible pairs.
    From those paired up values, choose the one that is higher
    ie. generate a list of binary values for pairs of ratings
pair_up :
    helper function
get_rmes :
    Alternate error metric - simply the root mean square error
"""
import numpy as np
from itertools import combinations
import pdb


def pair_up(array):
    '''
    Generate pairwise comparisons between all elements of the array in a systematic way
    '''
    pairs = np.array(list(combinations(array, 2)))
    assert (len(pairs.shape) == 2) and (pairs.shape[0] == (len(array) * (len(array) - 1))/2)
    return np.array(pairs[:, 0] > pairs[:, 1]) * 1

def compare_choices(prediction, target):
    '''
    Perform binary choice evaluation between non nans in target vs those in prediction
    That is, get the non nan indices from target - if there are 10, you have 45 choices to make- 
    1-2, 1-3, 1-4.... 9-10
    '''
    target_indices = np.where(~np.isnan(target))[0]
    if target_indices.shape[0] <= 1:
        return np.float('NaN')  
    target = target[target_indices]
    prediction = prediction[target_indices]
    compared_targets = pair_up(target)
    compared_predictions = pair_up(prediction)
    score = np.sum(compared_predictions == compared_targets)/len(compared_predictions)
    return score

def get_rmse(predictions, target):
    '''
    from the prediction array and target array, get the mean of square of difference
    '''
    mask_p = ~np.isnan(predictions)
    mask_t = ~np.isnan(target)
    mask_total = mask_p & mask_t
    rmse = np.sqrt(np.sum((predictions[mask_total] - target[mask_total])**2)/ np.sum(mask_total))
    return rmse
