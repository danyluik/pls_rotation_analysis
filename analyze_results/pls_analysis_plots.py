import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd
import seaborn as sns


"""
Used to plot line graphs of significance metrics by parameter (e.g., sample size) for a given LV
    df: column 1 as parameter, 2:5 as values for each rotation method
    x: values for the parameter
    xlabel: label for x-axis
    ylabel: label for the y-axis
    out_dir: output directory
    zval: if True, makes some adjustments if plotting z-values vs. pass rates

"""

def plot_significance(df, x, xlabel, ylabel, out_dir, zval=False):
    fig, ax = plt.subplots(figsize=(4,3))

    cmap = sns.color_palette('husl', 8)
    palette={'none':cmap[5], 'behav':cmap[3], 'brain':cmap[1], 'both':cmap[0]}

    for column in df.columns[1:]:
        sns.lineplot(x=x, y=column, data=df, color=palette[column], label=column, linewidth=2)

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=16, labelpad=6)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=16, labelpad=6)

    ax.get_legend().set_visible(False)
    ax.grid(True, axis='both')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    # Set bounds if pass rate
    if not zval:
        ax.set_ylim(-0.1, 1.1)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')

    if zval:
        plt.savefig(f'{out_dir}/z_scores.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{out_dir}/passrates.png', dpi=300, bbox_inches='tight')

    plt.close()




"""
Used to plot values of covariance explained/split-half stability by parameter (e.g., sample size) for a given LV
    df: each column as a parameter (i.e., value on the x-axis)
    x: values for the parameter
    color: color of the violins
    xlabel: label for x-axis
    ylabel: label for the y-axis
    descr: name for the file (ucorr, vcorr, covexp)
    out_dir: output directory

"""

def plot_values(df, x, color, xlabel, ylabel, descr, out_dir):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.stripplot(data=df, palette=[color], size=3)

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=16, labelpad=6)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=16, labelpad=6)

    ax.grid(True, axis='both')
    plt.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    plt.xticks(rotation=45)

    # If we're plotting by LV for UKBB, declutter the x-axis, also standardize range with other plots
    xvars = len(x)
    if xvars > 10:
        ax.set_xticks(np.arange(1,xvars+1,2))
        ax.set_xticklabels(np.arange(1,xvars+1,2))

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')

    plt.savefig(f'{out_dir}/{descr}.png', dpi=300, bbox_inches='tight')
    plt.close()




"""
Plot avg number of LVs passed by parameter (e.g., sample size) for each rotation method
    df: column 1 as parameter, 2:5 as values for each rotation method
    x: parameter values
    xlabel: label for x-axis
    out_dir: output directory

"""

def plot_passed(df, x, xlabel, out_dir):
    fig, ax = plt.subplots(figsize=(4,3))

    cmap = sns.color_palette('husl', 8)
    palette={'none':cmap[5], 'behav':cmap[3], 'brain':cmap[1], 'both':cmap[0]}

    for column in df.columns[1:]:
        sns.lineplot(x=x, y=column, data=df, color=palette[column], label=column, linewidth=2)

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=14, labelpad=6)
    ax.set_ylabel('Average LVs passed', fontweight='bold', fontsize=14, labelpad=6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, axis='both', alpha=0.3)
    plt.xticks(rotation=45)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')

    plt.savefig(f'{out_dir}/passed.png', dpi=300, bbox_inches='tight')
    plt.close()
    



"""
Plot p-values by latent variable for a single run of PLS. Used for final UKBB run.
    pvals: pd DataFrame of p-values, with columns as rotation methods
    out_dir: output directory

"""

def plot_pvals(pvals, out_dir):
    fig, ax = plt.subplots(figsize=(4,3))

    order=['none', 'behav', 'brain', 'both']
    cmap = sns.color_palette('husl', 8)
    palette={'none':cmap[5], 'behav':cmap[3], 'brain':cmap[1], 'both':cmap[0]}

    # Get pvals as dataframe
    pvals = pd.DataFrame(pvals)
    pvals = pvals[order]
    
    sns.lineplot(data=pvals, dashes=False, palette=palette, linewidth=2)
    plt.axhline(y=0.05, xmin=0.04, xmax=0.96, color='k', linestyle='--', alpha=0.7) # significance thresh

    # Ticks at every other LV
    lvs = len(pvals)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(0,lvs,2))
    ax.set_xticklabels(np.arange(1,lvs+1,2))

    ax.set_xlabel('Latent variable', fontweight='bold', fontsize=14, labelpad=6)
    ax.set_ylabel('p-value', fontweight='bold', fontsize=14, labelpad=6)

    ax.grid(True, axis='both', alpha=0.3)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')

    plt.savefig(f'{out_dir}/pvals.png', dpi=300, bbox_inches='tight')
    plt.close()




"""
Plot some metric by latent variable for a single run of PLS. Used for covariance explained in final UKBB analysis.
    x: reference values (i.e., array of latent variable IDs)
    y: values to plot (i.e., covariance explained)
    color: color of line
    descr: file name
    ylabel: label for y-axis
    out_dir: output directory

"""

def line_plot(x, y, color, ylabel, descr, out_dir):
    fig, ax = plt.subplots(figsize=(4,3))
    plt.plot(x, y, color=color, linewidth=2)

    ax.set_xlabel('Latent variable', fontweight='bold', fontsize=14, labelpad=6)
    ax.set_ylabel(f'{ylabel}', fontweight='bold', fontsize=14, labelpad=6)

    ax.grid(True, axis='both', alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # If we're plotting by LV for UKBB, declutter the x-axis, also standardize range with other plots
    xvars = len(x)
    if xvars > 10:
        ax.set_xticks(np.arange(1,xvars+1,2))
        ax.set_xticklabels(np.arange(1,xvars+1,2))

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')

    plt.savefig(f'{out_dir}/{descr}.png', dpi=300, bbox_inches='tight')
    plt.close()




"""
Plot the null distributions by rotation method for a single dataset
    df: dataframe of permuted singular values, with each column corresponding to a rotation method
    singval: actual singular value from the initial PLS
    lv_id: for the filename
    out_dir: output directory

"""

def plot_singvals(df, singval, lv_id, out_dir):
    # Make histogram
    fig, ax = plt.subplots(figsize=(4,3))
    cmap = sns.color_palette('husl', 8)
    palette = {'none':cmap[5], 'behav':cmap[3], 'brain':cmap[1], 'both':cmap[0]}
    sns.histplot(df, palette=palette, alpha=0.8)
    plt.axvline(x=singval, color='r', linewidth=2.5)

    ax.set_xlabel('Singular value', fontweight='bold', fontsize=14)
    ax.set_ylabel('Count', fontweight='bold', fontsize=14)
    ax.yaxis.get_major_locator().set_params(integer=True)

    xmin = 0.9*df.to_numpy().min()
    xmax = 0.8*df.to_numpy().max()
    plt.xlim(xmin, xmax)

    ax.grid(True, axis='both', alpha=0.3)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')

    plt.savefig(f'{out_dir}/lv_{lv_id}.png', dpi=300, bbox_inches='tight')
    plt.close()




"""
Horizontal bar chart of behavioural loadings with CI for a given LV
    y: y loadings, of shape (yvars, )
    y_ci: CIs on y loadings, of shape (yvars, 2) - low first, high second
    behav_vars: interpretable variable names for y, of shape (yvars, )
    lv_index: ID for LV, for plot title
    out_dir: output directory for figure

"""

def plot_behav_loadings(y, y_ci, behav_vars, lv_index, out_dir):
    # Array to store high/low error for each behav var based on loading and CI
    error = []
    total_vars = y.shape[0]
    behav_indices = range(0, total_vars) # for plot later


    # ERROR: For each behav var - get error for barchart based on CI
    for var in range(total_vars):
        # Shape (LVs, 2) - row 1 is CI low bound, row 2 is CI high bound
        ci = y_ci[var,:]

        # Get high and low error for this variable, append to running list
        var_err = [math.fabs((y[var] - ci[0])), math.fabs((y[var] - ci[1]))]
        error.append(var_err)
        
    error = np.array(error)
    error = error.T # now of shape (2, behav_vars)
        

    # COLORS: Green if significantly positive, red if significantly negative, grey if not significant
    colors = []
    cmap = sns.color_palette('deep',10)

    for var in range(total_vars):
        ci_low = y[var] - error[0, var]
        ci_high = y[var] + error[1, var]

        if (ci_low >= 0):
            colors.append(cmap[2]) # green

        elif (ci_high <= 0):
            colors.append(cmap[3]) # red

        else: 
            colors.append(cmap[7]) # grey


    # SORTING: most positive at top of the horizontal bar graph
    sort_indices = np.argsort(y)
    y = y[sort_indices]
    error = error[:,sort_indices]
    colors = np.array(colors)[sort_indices]
    behav_vars = np.array(behav_vars)[sort_indices]


    # PLOT - horizontal bar chart, variables on y-axis
    fig, ax = plt.subplots(figsize=(6,6))
    ax.barh(behav_indices, y, xerr=error, align='center', color=colors, error_kw={'elinewidth':3})
    ax.set_yticks(behav_indices)
    ax.set_yticklabels(behav_vars)

    ax.grid(True, axis='both', alpha=0.3, linewidth=2)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('black')
        ax.spines[axis].set_linewidth(1.2)
        
    ax.tick_params(axis='y', colors='white') # Make tick labels invisible
    plt.axvline(x=0, color='black', linewidth=2) # Add a vertical bar at loadings = 0

    # Set colors of y-axis labels the same as bars (green, red, grey)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), colors):
        ticklabel.set_color(tickcolor)
        ticklabel.set_fontweight('bold')

    ax.set_xlabel('Loadings', fontweight='bold', labelpad=10)
    ax.xaxis.label.set_fontsize(20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=20)

    plt.savefig(f'{out_dir}/behav_loadings', bbox_inches='tight') 
    plt.close()




"""
Plots regional weights according to some parcellation using create_civet_image.sh
    left_stats: some weight (BSR) on each left hem region (ordered by parcel label)
    right_stats: some weight (BSR) on each left hem region (ordered by parcel label)
    out_dir: output directory for figure
    left_parcellation: vertex-wise parcel labels for the left hem
    right_parcellation: vertex-wise parcel labels for the right hem

If no parcellation is given, the plot is made vertex wise, and it assumes the inputs exclude masked vertices

Before running the command:
    module load minc-toolkit-v2

"""

def plot_brain_bsr(left_stats, right_stats, out_dir, left_parcellation=None, right_parcellation=None):
    # If we have a parcellation, map parcel-wise weights to all vertices within the parcel
    if (left_parcellation is not None) and (right_parcellation is not None):
        left_labels = np.unique(left_parcellation)
        right_labels = np.unique(right_parcellation)

        # Define vertex-wise arrays for plotting with create_civet_image.sh
        # Basically will map feature weight to every vertex included within a region
        left_vertices = np.zeros(np.shape(left_parcellation))
        right_vertices = np.zeros(np.shape(right_parcellation))

        # For left: loop through each parcel, get stat, fill vertices corresponding to label with stat
        for (idx, label) in enumerate(left_labels):
            stat = left_stats[idx]
            to_fill = np.where(left_parcellation == label)
            left_vertices[to_fill] = stat

        # Repeat for right hem
        for (idx, label) in enumerate(right_labels):
            stat = right_stats[idx]
            to_fill = np.where(right_parcellation == label)
            right_vertices[to_fill] = stat
    

    # Otherwise plot vertexwise, assuming masked vertices are excluded
    else:
        left_mask = np.genfromtxt('civet/CIVET_2.1_mask_left_short.txt')
        right_mask = np.genfromtxt('civet/CIVET_2.1_mask_right_short.txt')

        # Define vertex-wise arrays, fill where we have valid vertices
        left_vertices = np.zeros(np.shape(left_mask))
        right_vertices = np.zeros(np.shape(right_mask))

        left_vertices[np.where(left_mask==1)] = left_stats
        right_vertices[np.where(right_mask==1)] = right_stats


    # Save vertex-wise measurements as pd Series
    # For some reason create_civet_image.sh likes it if I do it this way
    left_vertices = pd.Series(left_vertices)
    right_vertices = pd.Series(right_vertices)
    left_vertices.to_csv(f'{out_dir}/left_vertex_weights.csv', header=['BSR'])
    right_vertices.to_csv(f'{out_dir}/right_vertex_weights.csv', header=['BSR'])

    # Get max weight across hems for plot
    stats = np.concatenate((left_stats, right_stats), axis=0)
    max_weight = np.round(np.max(np.absolute(stats)), decimals=2)

    # Create a create_civet_image.sh command to visualize results
    image_command = (f"bash civet/create_civet_image.sh --left-statmap {out_dir}/left_vertex_weights.csv:\"BSR\" --right-statmap {out_dir}/right_vertex_weights.csv:\"BSR\" --colourmap /civet/RMINC_blue.lut,/civet/RMINC_red.lut --left-thresholds 0,{math.ceil(max_weight)} --right-thresholds 0,{math.ceil(max_weight)} --colourbar-labels \"0\",\"{max_weight}\" --left-mask civet/CIVET_2.1_mask_left_short.txt --right-mask civet/CIVET_2.1_mask_left_short.txt --no-annotate-directions /civet/CIVET_2.0_icbm_avg_mid_sym_mc_left.obj /civet/CIVET_2.0_icbm_avg_mid_sym_mc_right.obj {out_dir}/brain_bsr.png")
    os.system(image_command)

    os.remove(f'{out_dir}/left_vertex_weights.csv')
    os.remove(f'{out_dir}/right_vertex_weights.csv')
