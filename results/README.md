# Results

- Simulations contain the simulation results from runnings scripts/run.py. We have three different groupings in results :
    - In amateurs/performance.csv, we report simulation results for different rho and k values when drawing information only
    from amateurs for everyone
    - Similarly in experts/performance.csv, when drawing information only from experts for everyone
    - In both, we combine both pools of recommenders
        - We store adjacencies created as a result of simulations in adjacencies
            - We store each unique combination of k value and rho value in a separate file in both/adjacencies
            - We have 17 k values and 7 rho values (119 conditions in total, each with a separate csv file)
            - Each file is indexed by the combination of k and rho indices (k value index * 7 + rho value index)
            - These k and rho values can be found in src/config.py
            - E.g 15.csv is k = 3 (index=2) and rho = 0 (index=0)
            - Each of these files consist of all possible pairs of recommender - recommendees, totalling 134 * 134
            - The format is "to, from, weight"
        - Accuracies of recommendations are in performance.csv
    - In the supplementary folder, we have results from recommendations on single wines, and also results from the sensitivity analysis
    presented in figure 5    

- Additionally, we have a few files containing some basic results
    - In correlation.csv we have correlations between all pairs of individuals
    - In stats.csv we have their group labels, number of ratings etc.
    - influence.csv is used for group level influence presented in figure 12

- Homophily can be calculated at the aggregate and individual level, therefore Homophily contains two sub-folders 
    - individual.csv contains individual level homophily indices (reflecting the tendency of the algorithm to serve
     each individual information from their own group)
    - group.csv contains group level homophily indices (same as above but the indices are aggregated at the group level)
    - Both files contain information about the actual homophily used in the main paper, and the
    potential homophily reported in the supplementary material
    
- Visualization contains figures generated from scripts/viz.R and reported in the paper