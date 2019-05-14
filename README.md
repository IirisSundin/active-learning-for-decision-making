# Active Learning for Decision-Making from Imbalanced Observational data

This repository contains the code (and associated documentation and instructions) used in a paper 'Active Learning for Decision-Making from Imbalanced Observational data' by Iiris Sundin, Peter Schulam, Eero Siivola, Aki Vehtari, Suchi Saria and Samuel Kaski (ICML 2019). Preprint: https://arxiv.org/abs/1904.05268 

All code under licence BSD-3-Clause unless otherwise stated.

## Content
Main scripts in **bold**.
	
src/
- gp-comparison.stan: Stan model to fit two potential outcomes from observations and comparative feedback
- gpmodel.py: GP model class to fit two potential outcomes from observations and direct feedback on counterfactuals
- logit2.stan: Stan model for the logistic regression in the simulated experiment ('src/run_simple_example.py')
- model.py: Parent class for ITE models using active learning
- **run_active_GP_IHDP.py**: Run one LOOCV fold for the IHDP data (Section 5.2.2)
- **run_simple_example.py**: Run simulated example for one target unit (Section 5.2.1)
- run_simple_example_comparison.py: Run the simulated experiment with comparative feedback for one target unit (Section 5.2.5)
- simple_DM_GP.py: Correlation between observed and estimated Type S error rate and imbalance. (Section 5.1)
- stan_utility.py: Stan utility functions
- util.py: Functions used by other scripts
	
Note: the execution of some of the examples may be slow. Best run on a computation cluster using parallel batch jobs.
	
## Dependencies
- NumPy
- SciPy
- GPy (run_active_GP_IHDP.py)
- PyStan (run_simple_example.py, run_simple_example_comparison.py)


## Instructions how to run simulated experiment (Section 5.2.1)
1. Clone the repository
2. Ensure that you have PyStan installed together with other dependencies (see 'Dependencies' above).
3. Go to directory active-learning-for-decision-making/src/
4. Run run_simple_example.py [target_x] [n_train] [n_queries] [seed] [res_path] [acquisition] [stan_path]
	+ e.g. 'python run_simple_example.py -1.5 30 10 12345 ../results/-1_5+decerrig/ decerrig /stancodes/'
	for target x=-1.5, training sample size 30, 10 queries, seed 12345, save results to '../results/-1_5+decerrig/' and use D-M aware that explores. /stancodes/ points to a directory where cached stan models will be saved.

## Instructions how to run IHDP experimet (Section 5.2.2)
1. Clone the repository
2. Ensure that you have GPy installed together with other dependencies (see 'Dependencies' above).
3. Go to directory active-learning-for-decision-making/src/
4. Run run_active_GP_IHDP.py [target_idx] [n_queries] [seed] [res_path] [acquisition] 
	+ e.g. 'python run_active_GP_IHDP.py 1 5 12345 ../results/IHDP+1+decerrig/ decerrig'
	for target 1, 5 queries, seed 12345, and use D-M aware that explores ('decerrig')
	
