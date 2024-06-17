import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

from pls_analysis_plots import plot_behav_loadings, plot_brain_bsr, plot_pvals


"""
Example usage: python analyze_ukbb_full.py /path/to/pepp/data
Analyzes results from the PLS performed on the supplementary PEPP dataset.
Again, intended to run locally with available data, and a minc-toolkit-v2 module implementation. Provided here for completeness.

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1] # should have a /input directory with data/descriptions and a /results directory for PLS results
out_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)

subjects = np.genfromtxt(f'{working_dir}/input/subjects_actual.txt', dtype='str')
subjects = [subj[0:4] for subj in subjects]

rotations = ['none', 'behav', 'brain', 'both']

sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid', {'grid.linestyle':'--', 'grid.color':'grey'})




# ---------- BRAIN/BEHAV PLOTS ---------- #

# Get x weights, yloadings, lvs for this PLS run
# Order is L, R, in the numeric order within each hemisphere
x_weights = np.genfromtxt(f'{working_dir}/results/bootres/bootres_x_weights_normed.csv', delimiter=',') # (xvars, lvs)
split_idx = int(x_weights.shape[0] / 2) # useful later for splitting into L/R

y_loadings = np.genfromtxt(f'{working_dir}/results/y_loadings.csv', delimiter=',') # (yvars, lvs)

# Y loading CIs stored in separate .csv
yvars = y_loadings.shape[0]
lvs = y_loadings.shape[1]

y_loadings_ci = [] # (yvars, lvs, 2)

for yvar in range(yvars):
    var_ci_file = f'{working_dir}/results/bootres/y_loadings_ci/bootres_y_loadings_ci_behaviour_{yvar}.csv'
    var_ci = np.genfromtxt(var_ci_file, delimiter=',')
    y_loadings_ci.append(var_ci)
              
y_loadings_ci = np.array(y_loadings_ci)

# Get behav vars used, use reference file to get interpretable names
behav_vars = np.genfromtxt(f'{working_dir}/input/behav_vars.txt', dtype='str')


for lv in range(lvs):
    # For each LV, get weights
    lv_bsr = x_weights[:,lv]
    left_bsr = lv_bsr[:split_idx]
    right_bsr = lv_bsr[split_idx:]

    # Get ID of this LV, output dir
    lv_id = str(lv+1)
    lv_id = lv_id.zfill(len(str(lvs)))
    lv_dir = f'{out_dir}/lvs/lv_{lv_id}'
    os.makedirs(lv_dir, exist_ok=True)

    # Use this command to take parcellated data and map it back to vertexwise space
    # module load minc-toolkit-v2
    plot_brain_bsr(left_stats=left_bsr, right_stats=right_bsr, out_dir=lv_dir)

    # Get loadings, CI for this LV, plot
    lv_y = y_loadings[:,lv]
    lv_y_ci = np.squeeze(y_loadings_ci[:,lv,:])
    plot_behav_loadings(y=lv_y, y_ci=lv_y_ci, behav_vars=behav_vars, lv_index=lv+1, out_dir=lv_dir)




# ---------- SUMMARY PLOTS ---------- #

# Dictionaries of stats by rotation method
pvals = {}
singvals = {}
perm_singvals = {}


# For each type of rotation, add p value, singular value, and the set of singular values from the permutation test to the respective dict
for rotation in rotations:
    rotation_pvals = np.genfromtxt(f'{working_dir}/results/{rotation}/output/summary/permres_pvals.csv', delimiter=',')
    rotation_singvals = np.genfromtxt(f'{working_dir}/results/{rotation}/output/summary/singvals.csv', delimiter=',')
    rotation_perm_singvals = np.genfromtxt(f'{working_dir}/results/{rotation}/output/summary/perm_singvals.csv', delimiter=',')

    # Get mean and sigma of H0 for each LV, calculate a corresponding z-score for the true singular value
    h0_means = np.mean(rotation_perm_singvals, axis=1)
    h0_sigmas = np.std(rotation_perm_singvals, axis=1)

    # Add results to dictionary, keyed by rotation method
    pvals[rotation] = rotation_pvals
    singvals[rotation] = rotation_singvals
    perm_singvals[rotation] = rotation_perm_singvals


# For each latent variable, make a plot of singular value distributions across permutations for each rotation type
lvs = len(pvals[rotations[0]])
null_dir = f'{out_dir}/null_distributions'
os.makedirs(null_dir, exist_ok=True)

for lv in range(lvs):
    lv_id = str(lv+1)
    lv_id = lv_id.zfill(2)

    # Get permuted singular values corresponding to this LV for each method
    to_plot = [values[lv,:] for keys, values in perm_singvals.items()]

    # Do some hacky stuff to make sure each of these maps to a column in a pandas DF
    to_plot = pd.DataFrame({rotations[i]: values for i, values in enumerate(to_plot)})[rotations]
    singval = singvals[list(singvals)[0]][lv]

    plot_singvals(to_plot, singval, lv_id, null_dir)    


# Also plot a line graph of p values by latent variable for each rotation method
plot_pvals(pvals, out_dir)
