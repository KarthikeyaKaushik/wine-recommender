# Scripts

## run.py :

Main script to simulate the different social learning strategies
Workflow :
- Read data
- Initialise matrices to store simulated results
- Rerun the strategies for a particular number of repetitions (in NUM_REPS)
- There are two primary modes
    - You can simulate strategies to get the amount of influence raters exert on
    each other. In this case, it does not matter if I have rated an item or not. If you
    have a recommendation, and if the strategy recommends you, I will consider you influential
    - You can simulate strategies to get the accuracy for raters for different values of k and rho.
    In this case, the algorithm calculates a value only on items I have already rated.
- run.py depends on modules that are in the src folder. Each of these modules is well commented and you can inspect them by
going through run.py and the relevant module in src.
    
## homophily.py :


This script calculates the homophily indices from a simulated dataset, for both actual and
 potential influence. It operates on top of data produced by run.py. The potential influence relies on the correlation matrix, 
while the actual homophily is computed from the simulated data
Workflow :
- Read simulated data
- For different values of k and rho, compute the actual homophily index as - 
    hi = si / (si + di)
- That is, the homophily index is the proportion of influence received from raters of the in-group
(given by si) to the total influence
- We consider two baselines
    - Group proportion : in-group-rater-count/total-rater-count
    - Count proportion : tldr; it is the ratio of the number of ratings made by one group to the
    other. 
    Longer explanation : To compute the theoretical influence a rater can exert, we need to consider
    the number of ratings they can have in the training dataset. This is stochastic, because in the 
    'actual' influence case (see first mode in run.py), we populate the test set with a fixed number of 
    random items without caring if it has been rated or not. This means there are many items in the 
    training set that have not been evaluated, and on those items, the rater cannot exert any influence.
    Therefore, the count proportion is calculated as :
    total-items = total number of items in the dataset
    actual-ratings = number of items a rater has evaluated (max(actual-ratings) = total-items)
    training-set-counts = actual-ratings * (total-items - HOLDOUT)/total-items
                        = actual-ratings * Probability(item being included in training set)
    
    
## stats.py :

This script gives us a few basic metrics about the dataset - we have number of raters in each group, 
the rating correlation means and dispersion, and also the number of ratings per rater.

## single_wines.py

In the previous simulations, we look at influence results aggregating over all wines. This script reproduces
those analyses for specifically chosen wines.

## sparse_critics.py

This is a sensitivity analysis presented in the first section of the supplementary material. Here, we introduce additional 
sparsity by randomly selecting critics' ratings, and setting them to NaN so that the number of ratings per critic
matches the average number of amateur ratings. 

## viz.R :

Visualization scripts in R, it operates on top of data produced by run.py, homophily.py and stats.py.

## network_viz.py

This script produces the (beautiful) network plots in the paper. A paramtrically generated layout
represents critics and amateurs as nodes on a graph. The directed network connections depend upon simulated data from run.py. 

## spare_scripts

This folder contains scripts that were eventually not used for the paper. 