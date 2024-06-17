import numpy as np
import os
import sys

np.random.seed(123)


"""
Usage: python get_ukbb_data.py /path/to/outputs

Draws subsamples from the broader pool of UKBB subjects, previously organized with clean_ukbb_data.py (100 datasets at each of 10 sample sizes).
Intended to be run locally on already downloaded data. It's just provided here for completeness.

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1]
brain_matrix = np.genfromtxt(f'{working_dir}/input/ukbb_brain.csv', dtype='float', delimiter=',')
behav_matrix = np.genfromtxt(f'{working_dir}/input/ukbb_behav.csv', dtype='float', delimiter=',')

steps = 10 # number of steps of N
draws = 100 # separate draws of subjects for each N




# ---------- DRAW SUBSAMPLES ---------- #

# Get number of steps to make, in terms of % of sample to use, then convert to actual N
n = behav_matrix.shape[0]
step_n = np.logspace(start=np.log10(51), stop=np.log10(20000), num=steps, dtype=int)

# Make an output directory, save this info
data_dir = f'{working_dir}/data'
os.makedirs(data_dir, exist_ok=True)
np.savetxt(f'{data_dir}/n.csv', step_n , delimiter=',')


# Recursive function that chooses n subjects and gets corresponding brain/behav matrices
# Keeps going until it finds a matrix with enough non-zero variance values
def get_subset(n, current_n, brain_matrix, behav_matrix):
    subset = sorted(np.random.choice(a=n, size=current_n, replace=False))
    brain_subset = brain_matrix[subset, :]
    behav_subset = behav_matrix[subset, :]

    # Variance by column
    column_var = np.var(behav_subset, axis=0)

    # Try again if any of them have no variance
    if np.any(column_var == 0):
        return get_subset(n, current_n, brain_matrix, behav_matrix)

    else:
        return brain_subset, behav_subset



# For each step (N subjects):
for step in range(steps):
    # Get ID, so steps are sorted ascending
    step_id = str(step+1)
    step_id = step_id.zfill(len(str(steps)))
    print(step_id)

    current_n = step_n[step]

    # Then, for each draw, draw N subjects, save a brain/behav matrix
    for draw in range(draws):
        # Call function defined above to get brain/behav subsets
        brain_subset, behav_subset = get_subset(n, current_n, brain_matrix, behav_matrix)

        # Get ID for this draw, make a step/draw output dir
        draw_id = str(draw+1)
        draw_id = draw_id.zfill(len(str(draws)))
        out_dir = f'{data_dir}/step_{step_id}/draw_{draw_id}'
        os.makedirs(out_dir, exist_ok=True)

        # Save in our subset-specific output directory
        np.savetxt(f'{out_dir}/brain_matrix.csv', brain_subset, delimiter=',')
        np.savetxt(f'{out_dir}/behav_matrix.csv', behav_subset, delimiter=',')
