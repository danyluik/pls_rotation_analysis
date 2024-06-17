import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

from pls_analysis_plots import plot_behav_loadings, plot_brain_bsr, plot_pvals, line_plot, plot_values


"""
Usage: python analyze_ukbb_full.py /path/to/pls/data
Analyzes results from the final PLS performed on the full set of UKBB participants.
Again, intended to run locally with available data, and a minc-toolkit-v2 module implementation. Provided here for completeness.

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1] # path should have a /data folder (with PLS outputs) and a /input folder (various input data for UKBB analysis)
data_dir = f'{working_dir}/data/full/output'

out_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)

# Our CIVET data is regional not vertexwise
left_parcellation = np.loadtxt('civet/CIVET_2.1.0_dkt_left_short.txt')
right_parcellation = np.loadtxt('/civet/CIVET_2.1.0_dkt_right_short.txt')

rotations = ['none','brain','behav','both']

sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid', {'grid.linestyle':'--', 'grid.color':'grey'})
    
corr_color = sns.color_palette('husl', 8)[4]
var_color = sns.color_palette('husl', 8)[6]




# ---------- PLOT BRAIN, BEHAV ---------- #

# Get x weights, yloadings, lvs for this PLS run
# Order is L, R, in the numeric order within each hemisphere
x_weights = np.genfromtxt(f'{data_dir}/none/x_weights_normed.csv', delimiter=',') # (xvars, lvs)
split_idx = int(x_weights.shape[0] / 2) # useful later for splitting into L/R

y_loadings = np.genfromtxt(f'{data_dir}/none/y_loadings.csv', delimiter=',') # (yvars, lvs)

# Y loading CIs stored in separate .csv
yvars = y_loadings.shape[0]
lvs = y_loadings.shape[1]

y_loadings_ci = [] # (yvars, lvs, 2)

for yvar in range(yvars):
    var_ci_file = f'{data_dir}/none/y_loadings_ci/y_loadings_ci_{yvar}.csv'
    var_ci = np.genfromtxt(var_ci_file, delimiter=',')
    y_loadings_ci.append(var_ci)
              
y_loadings_ci = np.array(y_loadings_ci)

# Get behav vars used, use reference file to get interpretable names
behav_vars = np.genfromtxt(f'{working_dir}/input/behav_vars.csv', delimiter=',', dtype='str')
behav_ref = pd.read_csv(f'{working_dir}/input/behav_ref.csv', delimiter=',', dtype='str')
behav_vars = [behav_ref.loc[behav_ref['code']==var]['name'].values[0] for var in behav_vars]

max_weight = round(np.max(np.absolute(x_weights)), 2)

for lv in range(lvs):
    # 1. Brain BSR
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
    plot_brain_bsr(left_stats=left_bsr, right_stats=right_bsr, out_dir=lv_dir, left_parcellation=left_parcellation, right_parcellation=right_parcellation)


    # 2. Behav loadings
    lv_y = y_loadings[:,lv]
    lv_y_ci = np.squeeze(y_loadings_ci[:,lv,:])

    # Use custom plotting function
    plot_behav_loadings(y=lv_y, y_ci=lv_y_ci, behav_vars=behav_vars, lv_index=lv+1, out_dir=lv_dir)




# ---------- SUMMARY PLOTS ---------- #

pvals = {}

# For each type of rotation, add p values to dict
for rotation in rotations:
    rotation_pvals = np.genfromtxt(f'{data_dir}/{rotation}/pvals.csv', delimiter=',')
    pvals[rotation] = rotation_pvals
    
plot_pvals(pvals=pvals, out_dir=out_dir)


# Load in varexp, plot with respect to LV (consistent across rotation methods)
lv_labels = np.arange(start=1, stop=lvs+1)

varexp = np.genfromtxt(f'{data_dir}/none/varexp.csv', delimiter=',')
line_plot(x=lv_labels, y=varexp, color=var_color, ylabel='Covariance explained', descr='covexp', out_dir=out_dir)

# Do the same for ucorr/vcorr (taking absolute value to account for arbitrary sign flips)
ucorr = np.genfromtxt(f'{data_dir}/none/ucorr.csv', delimiter=',') # (lvs, splits)
vcorr = np.genfromtxt(f'{data_dir}/none/vcorr.csv', delimiter=',')

ucorr = pd.DataFrame(data=np.absolute(ucorr).T, columns=lv_labels)
vcorr = pd.DataFrame(data=np.absolute(vcorr).T, columns=lv_labels)

plot_values(df=ucorr, x=lv_labels, color=corr_color, xlabel='Latent variable', ylabel='Avg. stability', descr='u_corr', out_dir=out_dir)
plot_values(df=vcorr, x=lv_labels, color=corr_color, xlabel='Latent variable', ylabel='Avg. stability', descr='v_corr', out_dir=out_dir)
