#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:27:07 2022

@author: karthikeyarkaushik
"""
import pandas as pd
import numpy as np
from src.compute import get_homophily_individual, get_homophily_group, get_influence
from src.data_processing import merge_data
from config import *
import os.path

if __name__ == '__main__':
    [dataframe, num_experts, num_amateurs] = merge_data()
    correlation_matrix = pd.DataFrame(np.transpose(dataframe)).corr(min_periods=MIN_OVERLAP).to_numpy()
    
    # storing homophily for different parameters
    # method is potential homophily or real
    # store both individual and group homophilies
    homophily_individual = pd.DataFrame(columns=['group', 'kval', 'rhoval', 'Hi', 'method', 'baseline_1',
                                         'baseline_2', 'rater'])
    homophily_df = pd.DataFrame(columns=['group', 'kval', 'rhoval', 'Hi', 'method', 'baseline_1',
                                         'baseline_2'])
    influence_df = pd.DataFrame(columns=['group', 'kval', 'rhoval', 'proportion_per_capita', 'proportion_gross',
                                         'method', 'baseline_1','baseline_2'])
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
            parameter_homophily = get_homophily_individual(adjacency, rho_val, k_val, 
                                                           type_homophily = 'Real homophily',
                                               master_dataset = dataframe)  
            homophily_individual = pd.concat([homophily_individual, parameter_homophily])           
            # now do group :
            s_exp, s_ama = get_homophily_group(adjacency, rho_val, k_val, type_homophily = 'Real homophily', 
                                      master_dataset = dataframe)
            ex_df = pd.DataFrame(s_exp)
            am_df = pd.DataFrame(s_ama)
            homophily_df = pd.concat([homophily_df, ex_df, am_df])
        
            s_exp, s_ama = get_homophily_group(correlation_matrix, rho_val, k_val, type_homophily = 'Potential homophily',
                                              master_dataset = dataframe)  
            ex_df = pd.DataFrame(s_exp)
            am_df = pd.DataFrame(s_ama)
            homophily_df = pd.concat([homophily_df, ex_df, am_df])       
            # also do influence calculations : Potential first
            s_exp, s_ama = get_influence(correlation_matrix, rho_val, k_val, type_influence = 'Potential influence',
                                              master_dataset = dataframe)  
            ex_df = pd.DataFrame(s_exp)
            am_df = pd.DataFrame(s_ama)
            influence_df = pd.concat([influence_df, ex_df, am_df])       
            # now do real influence
            s_exp, s_ama = get_influence(adjacency, rho_val, k_val, type_influence = 'Real influence',
                                              master_dataset = dataframe)  
            ex_df = pd.DataFrame(s_exp)
            am_df = pd.DataFrame(s_ama)
            influence_df = pd.concat([influence_df, ex_df, am_df])       
            
            
            
    homophily_individual.to_csv(os.path.join('results', 'homophily', GROUP, 'individual.csv'))    
    print('Done computing individual homophilies')
    homophily_df.to_csv(os.path.join('results', 'homophily', GROUP, 'group.csv'))
    print('Done computing group homophilies')    
    influence_df.to_csv(os.path.join('results','influence.csv'))
    print('Done computing influence')
    
    
    
    
    