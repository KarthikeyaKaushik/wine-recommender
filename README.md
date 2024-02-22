# wine-final

Building a collaborative filtering algorithm for wines and understanding the role of expert and amateur influence

To run the algorithm, initialise the parameters in the config.py file :

NUM_EXPERTS : Number of experts
NUM_AMATEURS : Number of amateurs
NUM_TOTAL : Total number of raters
NUM_REPS : Number of independent simulation runs to aggregate results
PARAMETERS : Dictionary storing the k values (number of neighbours) and 
        rho values (weight associated with a neighbour's opinion)
SPLIT_HOW : How to split the dataset. Two options :
            - 'one_out' : a generalised version, where you leave n_out of the dataset as test
            - 'keep_n' : complementary way
            default is one_out
HOLDOUT : How many items to leave out (or keep)
GROUP : Testing the strategies on which group
COMPUTATION : You can either run the simulations to get accuracy (= 'performance')
              or you can run simulations to get influence measures (= 'influence')
MIN_OVERLAP : Minimum overlap needed to get a valid correlation in a sparse dataset


To run the simulations, do :

python scripts/run.py from the root     

Please go through the modules in src to get a sense of the workflow.


