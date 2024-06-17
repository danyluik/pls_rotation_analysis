from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys


"""
Example usage: python plot_null_distributions.py /path/to/single_dataset
I.e., path should end in /data/step_X/draw_Y/output, and have perm_singvals.csv inside each rotation method subfolder

Will create box plots of the null distributions by latent variable, separately for each rotation method, 
in the folder for the sample dataset specified.

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1]

rotate_methods = ['none', 'behav', 'brain', 'both']

sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid', {'grid.linestyle':'--', 'grid.color':'grey'})

cmap = sns.color_palette('husl', 8)
palette={'none':cmap[5], 'behav':cmap[3], 'brain':cmap[1], 'both':cmap[0]}




# ---------- PLOT NULL DISTRIBUTIONS ---------- #

perm_singvals = {}
for rotate_method in rotate_methods:
    perm_singvals[rotate_method] = np.genfromtxt(f'{working_dir}/{rotate_method}/perm_singvals.csv', delimiter=',')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, rotate_method in enumerate(rotate_methods):
    # For each method, load singvals by LV across all permuations
    method_singvals = perm_singvals[rotate_method]

    # Make a DF with columns as LVs and rows as singval by permutation
    lvs = np.arange(1, method_singvals.shape[0]+1, dtype=int)
    to_plot = pd.DataFrame(data=method_singvals.T, columns=lvs)

    # Turn into a boxpot by LV
    ax = sns.boxplot(data=to_plot,
        boxprops=dict(color=palette[rotate_method], edgecolor='black', linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1),
        medianprops=dict(color='black', linewidth=1),
        flierprops=dict(color='black', markeredgecolor='black', markersize=0.8),
        ax=axes[floor(idx/2), idx % 2])

    if dataset == 'white':
        ax.set_ylim([0.05, 0.15])
    elif dataset == 'ukbb':
        ax.set_ylim([0, 0.3])

    ax.set_xlabel('Latent variable', fontweight='bold', fontsize=14)
    ax.set_ylabel('Singular value', fontweight='bold', fontsize=14)
    ax.set_title(rotate_method.capitalize(), fontweight='bold', fontsize=16, pad=10)
    
    ax.grid(True, axis='both', alpha=0.3)

    # Avoid clutter
    n_lvs = len(lvs)

    if n_lvs > 10:
        ax.set_xticks(np.arange(0,len(lvs),2))
        ax.set_xticklabels(np.arange(1,len(lvs)+1,2))

    # Make boundaries thicker
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')


plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(f'{working_dir}_null_distribution.png', dpi=300, bbox_inches='tight')
