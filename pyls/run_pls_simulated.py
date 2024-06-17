import glob
import numpy as np
import os
import sys

from pyls import behavioral_pls


"""
Usage: python run_pls_simulated.py /path/to/data/step_X/iter_Y

Will perform PLS, with four different permutation tests, on a given dataset.
Results will be stored in a new /output directory in the dataset folder.
PLS is performed using the pyls library, with custom modifications to implement the permutation tests and split-half stability analysis.
When analyzing a full set of datasets from a given analysis, can be submitted as a batch job on a HPC cluster.

Source: https://github.com/rmarkello/pyls

"""


dataset = sys.argv[1]
rotate_methods = ['both','brain','behav','none']
nperm = 10000
nsplit = 100

X = np.genfromtxt(glob.glob(f'{dataset}/X*.csv')[0], dtype='float', delimiter=',')
Y = np.genfromtxt(glob.glob(f'{dataset}/Y*.csv')[0], dtype='float', delimiter=',')


# Run PLS, with different rotation methods
for method in rotate_methods:
    out_dir = f'{dataset}/output/{method}'
    os.makedirs(out_dir, exist_ok=True)

    if method == 'both':
        pls_results = behavioral_pls(X, Y, n_perm=nperm, n_boot=0, n_split=0, test_split=None, seed=123, rotate=True, rotate_method='both')

    elif method == 'brain':
        pls_results = behavioral_pls(Y, X, n_perm=nperm, n_boot=0, n_split=0, test_split=None, seed=123, rotate=True, rotate_method='right')

    elif method == 'behav':
        pls_results = behavioral_pls(X, Y, n_perm=nperm, n_boot=0, n_split=0, test_split=None, seed=123, rotate=True, rotate_method='right')

    elif method == 'none':
        pls_results = behavioral_pls(X, Y, n_perm=nperm, n_boot=0, n_split=nsplit, test_split=None, seed=123, rotate=False)


    # Get p-vals and some other rotation-specific stuff
    pvals = pls_results['permres']['pvals']
    singvals = pls_results['singvals']
    perm_singvals = pls_results['permres']['perm_singvals']

    np.savetxt(f'{out_dir}/pvals.csv', pvals, delimiter=',')
    np.savetxt(f'{out_dir}/singvals.csv', singvals, delimiter=',')
    np.savetxt(f'{out_dir}/perm_singvals.csv', perm_singvals, delimiter=',')
    
    # Only get stability and strength metrics for no rotation (arbitrary choice) - otherwise redundant
    if method == 'none':
        varexp = pls_results['varexp']
        ucorr = pls_results['splitres']['ucorr']
        vcorr = pls_results['splitres']['vcorr']

        np.savetxt(f'{out_dir}/varexp.csv', varexp, delimiter=',')
        np.savetxt(f'{out_dir}/ucorr.csv', ucorr, delimiter=',')
        np.savetxt(f'{out_dir}/vcorr.csv', vcorr, delimiter=',')
    
    del pls_results
