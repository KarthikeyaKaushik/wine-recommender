#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:57:27 2022

@author: karthikeyarkaushik
"""

NUM_EXPERTS = 14
NUM_AMATEURS = 120
NUM_TOTAL = 134
NUM_REPS = 100
# change the parameters depending on your dataset
PARAMETERS = {'k':[1,2,3,5,7,9,11,13,17,19,23,29,50,75,100,125,NUM_EXPERTS+NUM_AMATEURS-1],
              'rho':[0,0.25,0.5,0.75,1,1.25,1.5]}
SPLIT_HOW = 'one_out' # the kind of data splitting used
HOLDOUT = 10 # Leave HOLDOUT, or keep HOLDOUT
GROUP = 'both' # testing the strategies on which group?
COMPUTATION = 'performance'#'performance' # can be performance, or computing influence
MIN_OVERLAP = 5 # overlap to get valid correlation
STRATEGY = 'baseline' # which strategy : naive_knn, baseline, weighted_averaging