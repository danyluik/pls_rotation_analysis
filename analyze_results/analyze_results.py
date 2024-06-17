import glob
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys

from pls_analysis_plots import plot_significance, plot_values, plot_passed


"""
Example usage: python analyze_simulated_data.py /path/to/pls/data
Path should be to the /data directory created for any given analysis

Will create a /figures directory under /data plotting various metrics by parameter (e.g., sample size) for each LV

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1]
figure_dir = f'{working_dir}/figures'
os.makedirs(figure_dir, exist_ok=True)


# Axis label for plots depending on parameter
if 'canonical_corr' in working_dir:
    plot_label = 'Canonical corr.'
elif 'noise' in working_dir:
    plot_label = 'Noise std. dev.'
elif ('sample_size' in working_dir) or ('white' in working_dir) or ('ukbb' in working_dir):
    plot_label = 'Sample size'

rotate_methods = ['none', 'behav', 'brain', 'both']

# Colors for strip plots
corr_color = sns.color_palette('husl', 8)[4]
var_color = sns.color_palette('husl', 8)[6]

sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid', {'grid.linestyle':'--', 'grid.color':'grey'})

alpha = 0.05




# ---------- LOAD RESULTS ---------- #

results_dir = f'{working_dir}/results'

# Of shape (datasets, )
vals = np.genfromtxt(glob.glob(f'{working_dir}/*.csv')[0], dtype='float')

# Basically format values to make plots look nice
# Either trim to 2 decimal places, or make integers
if np.all(vals <= 1):
    vals = ['{:.2f}'.format(val) for val in vals]
else:
    vals = ['{:.0f}'.format(val) for val in vals]


# For each method (key), of shape (steps, draws)
with open(f'{results_dir}/pvals.pkl', 'rb') as f:
    pvals = pickle.load(f)

# For each method (key), of shape (datasets, lvs)
with open(f'{results_dir}/passrates.pkl', 'rb') as f:
    passrates = pickle.load(f)

# For each method (key), of shape (datasets, repeats, lvs)
with open(f'{results_dir}/z_scores.pkl', 'rb') as f:
    z_scores = pickle.load(f)

# For 'none' (key), of shape (datasets, repeats, lvs, splits)
with open(f'{results_dir}/ucorr.pkl', 'rb') as f:
    ucorr = pickle.load(f)

# For 'none' (key), of shape (datasets, repeats, lvs, splits)
with open(f'{results_dir}/vcorr.pkl', 'rb') as f:
    vcorr = pickle.load(f)

# For 'none' (key), of shape (datasets, repeats, lvs)
with open(f'{results_dir}/varexp.pkl', 'rb') as f:
    varexp = pickle.load(f)




# ---------- PLOT RESULTS ---------- #

if 'ukbb' in working_dir:
    # Get average LVs passing permutation test by sample size for each method
    avg_passed = {}

    for method in rotate_methods:
        # Sum the number of LVs passing for each iter (10 * 100)
        method_pvals = pvals[method]
        method_passed = np.sum(method_pvals < alpha, axis=2)

        # Average across the 100 iterations to get one average for each sample size
        method_passed_avg = np.average(method_passed, axis=1)
        avg_passed[method] = method_passed_avg

    avg_passed = pd.DataFrame(avg_passed)
    avg_passed.insert(0, 'n', vals)
    plot_passed(df=avg_passed, x='n', xlabel='Sample size', out_dir=f'{figure_dir}')

# Number of total LVs in analysis
lvs = passrates['none'].shape[1]


# Plots by LV
for lv in range(lvs):
    # For each LV, get an LV-specific output directory
    lv_id = str(lv+1)
    lv_id = lv_id.zfill(len(str(lvs)))
    out_dir = f'{figure_dir}/lvs/lv_{lv_id}'
    os.makedirs(out_dir, exist_ok=True)

    # Make a dicionary of passrates by rotate method for this LV
    # Same for z-scores
    lv_passrates = {}
    lv_zscores = {}
    for method in rotate_methods:
        lv_passrates[method] = passrates[method][:,lv]
        lv_zscores[method] = np.squeeze(np.average(z_scores[method][:,:,lv], axis=1))

    # Convert to dataframe then plot passrate by method across datasets
    lv_passrates = pd.DataFrame(lv_passrates)
    lv_passrates.insert(0, 'vals', vals)
    plot_significance(df=lv_passrates, x='vals', xlabel=plot_label, ylabel='Pass rate', out_dir=out_dir)
    
    # Same for z-scores
    lv_zscores = pd.DataFrame(lv_zscores)
    lv_zscores.insert(0, 'vals', vals)
    plot_significance(df=lv_zscores, x='vals', xlabel=plot_label, ylabel='Average z-value', out_dir=out_dir, zval=True)


    # Then - make plots of u and v corr values by dataset
    # Basically, get splithalf values for this LV, then average across splits
    lv_ucorr = np.absolute(np.squeeze(ucorr['none'][:,:,lv,:]))
    lv_ucorr = np.average(lv_ucorr, axis=2) # of shape (datasets, repeats)
    lv_ucorr = pd.DataFrame(data=lv_ucorr.T, columns=vals)
    plot_values(df=lv_ucorr, x=vals, color=corr_color, xlabel=plot_label, ylabel='Avg. stability', descr='u_corr', out_dir=out_dir)

    lv_vcorr = np.absolute(np.squeeze(vcorr['none'][:,:,lv,:]))
    lv_vcorr = np.average(lv_vcorr, axis=2)
    lv_vcorr = pd.DataFrame(data=lv_vcorr.T, columns=vals)
    plot_values(df=lv_vcorr, x=vals, color=corr_color, xlabel=plot_label, ylabel='Avg. stability', descr='v_corr', out_dir=out_dir)


    # Finally - make plot of varexp by dataset
    # No averaging, just violin across repeats
    lv_varexp = varexp['none'][:,:,lv]
    lv_varexp = pd.DataFrame(data=lv_varexp.T, columns=vals)
    plot_values(df=lv_varexp, x=vals, color=var_color, xlabel=plot_label, ylabel='Cov. explained', descr='covexp', out_dir=out_dir)
