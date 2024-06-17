import numpy as np
import os
import pandas as pd
import pickle
import sys


"""
Example usage: python organize_results.py /path/to/pls/data
I.e., path should be to the /data directory created for any given analysis

Will create a /results directory in /data with dictionaries of various output metrics keyed by rotation method, stored as .pkl files
Just makes life easier when analyzing them later :)

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1]
rotate_methods = ['none', 'brain', 'behav', 'both']
alpha = 0.05 # significance threshold




# ---------- GET RESULTS ---------- #

# Generic function for loading in results and organizing them as a dictionary keyed by rotation method
def load_data(out_file, working_dir, rotate_methods=rotate_methods):
    steps = os.listdir(working_dir)
    steps = sorted([folder for folder in steps if 'step' in folder])

    data = {}

    # LEVEL 1 - METHOD (dict key)
    # For each rotation method, create a running list of values - will be of shape (teps, draws)
    for rotate_method in rotate_methods:
        method_data = []
    
        # LEVEL 2 - STEP (dim. 1)
        # For each step, create a running list of values - will be of shape (draws,)
        for step in steps:
            step_data = []

            draws = os.listdir(f'{working_dir}/{step}/')
            draws = sorted([folder for folder in draws if 'draw' in folder])

            # LEVEL 3 - DRAW (dim. 2)
            # Now - load data for each draw, fill step array accordingly
            for draw in draws:
                data_file = f'{working_dir}/{step}/{draw}/output/{rotate_method}/{out_file}.csv'
                draw_data = np.genfromtxt(data_file, delimiter=',')                 
                step_data.append(draw_data)

            step_data = np.array(step_data)            
            method_data.append(step_data)

        # Once each step is done, store full array as a key-value pair in a dict under this rotation method name 
        data[rotate_method] = np.array(method_data)
    
    return data


# Get a series of dictionaries, each keyed by rotation method
# Shapes of the numpy arrays for each method are commented
pvals = load_data('pvals', working_dir) # (steps, draws, lvs)
singvals = load_data('singvals', working_dir) # (steps, draws, lvs)

# These metrics are identical across runs, so we just load for the unrotated condition
varexp = load_data('varexp', working_dir, rotate_methods=['none']) # (steps, draws, lvs)
ucorr = load_data('ucorr', working_dir, rotate_methods=['none']) # (steps, draws, lvs, splits)
vcorr = load_data('vcorr', working_dir, rotate_methods=['none']) # (steps, draws, lvs, splits)




# ---------- DERIVE OTHER METRICS ---------- #

# Get permutation test pass rate by step (% of draws < 0.05)
passrates = {} # (steps, lvs)

for method in rotate_methods:
    # Load in p-vals for each method
    method_pvals = pvals[method]
    n_draws = method_pvals.shape[1]

    # Find no. of draws passing, divide by no. draws
    passed = np.sum(method_pvals < alpha, axis=1)
    method_passrates = np.round((passed / n_draws), 2)

    # Add to dict
    passrates[method] = method_passrates


# Get z-value of singular value in null distribution
z_scores = {} # (steps, draws, lvs)

for method in rotate_methods:
    # Load in singvals, perm singvals
    method_singvals = singvals[method]

    # Calculate mean, sigma of each distribution
    means = np.mean(method_perm_singvals, axis=3)
    sigmas = np.std(method_perm_singvals, axis=3)
    method_z_scores = np.divide(np.subtract(method_singvals, means), sigmas)
    
    # Add to dict
    z_scores[method] = method_z_scores




# ---------- SAVE EVERYTHING ---------- #

results_dir = f'{working_dir}/results'
os.makedirs(results_dir, exist_ok=True)

with open(f'{results_dir}/varexp.pkl', 'wb') as f:
    pickle.dump(varexp, f)
    
with open(f'{results_dir}/pvals.pkl', 'wb') as f:
    pickle.dump(pvals, f)

with open(f'{results_dir}/singvals.pkl', 'wb') as f:
    pickle.dump(singvals, f)

with open(f'{results_dir}/ucorr.pkl', 'wb') as f:
    pickle.dump(ucorr, f)

with open(f'{results_dir}/vcorr.pkl', 'wb') as f:
    pickle.dump(vcorr, f)

with open(f'{results_dir}/passrates.pkl', 'wb') as f:
    pickle.dump(passrates, f)

with open(f'{results_dir}/z_scores.pkl', 'wb') as f:
    pickle.dump(z_scores, f)
