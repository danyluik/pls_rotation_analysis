import numpy as np
import os
import pandas as pd
import sys


"""
Usage: python clean_ukbb_data.py /path/to/outputs

Reorganizes some local UKBB data (some reshaping, removing columns with not enough variance)
Intended to be run locally on already downloaded data. It's just provided here for completeness.

"""


working_dir = sys.argv[1]

# Brain: participant-wise DKT regional CT values, should be of shape (28 804, 64)
brain_matrix = pd.read_parquet(f'{working_dir}/ref/DKTcortthick.parquet')
brain_matrix = brain_matrix.T # otherwise (regions * participants)
brain_matrix.to_csv(f'{working_dir}/ukbb_brain.csv', index=False, header=False)


# Behav: participant-wise lifestyle risk factors, should be of shape (28 804, 38)
# Make sure index is first column (ID), save variables for reference
behav_matrix = pd.read_csv(f'{working_dir}/ref/20230828_subject_selection_rm+derived2_imputed.csv', index_col=0)
original_behav_vars = pd.Series(behav_matrix.columns)

behav_matrix = behav_matrix.loc[:, behav_matrix.var() >= 1] # Remove any columns with low variance
behav_vars = pd.Series(behav_matrix.columns)

removed = original_behav_vars[~original_behav_vars.isin(behav_vars)]

behav_vars.to_csv(f'{working_dir}/behav_vars.csv', index=False, header=False)
removed.to_csv(f'{working_dir}/removed_vars.csv', index=False, header=False)
behav_matrix.to_csv(f'{working_dir}/ukbb_behav.csv', index=False, header=False)
