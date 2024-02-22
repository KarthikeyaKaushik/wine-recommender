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
    
## homophily.py :

This script calculates the homophily indices from a simulated dataset, for both
actual and potential influence. The potential influence relies on the correlation matrix, 
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

## viz.R :

Visualization scripts in R
    