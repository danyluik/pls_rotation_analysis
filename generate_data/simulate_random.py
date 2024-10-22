import numpy as np
import os
import pandas as pd
import sys

np.random.seed(99999)


"""
Usage: python simulate_random.py /path/to/outputs

Generates datasets of values randomly drawn from a normal distribution with mean 0 and variance 1
(100 datasets at each of 10 sample sizes). Each pair of X and Y matrices is saved under the working 
directory, in random/data/step_X/draw_Y

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1]

steps = 10 # Number of different sample sizes
draws = 100 # Number of draws per step

n = np.logspace(np.log10(100), np.log10(10000), num=steps) # sample size
n = np.round(n, decimals=0).astype(int) # make all n integers

p_x = 90 # x features
p_y = 10 # y features

working_dir = f'{working_dir}/random'
os.makedirs(working_dir, exist_ok=True)

np.savetxt(f'{working_dir}/n.csv', n, delimiter=',')

description = 'Vary N, 10 values log spaced between 10 and 10 000. 100 draws of each. \n \
               90 X, 10 Y, of randomly generated, whitened data.' 

out_file = open(f'{working_dir}/run_description.txt', 'w')
out_file.write(f'{description} \n')
out_file.close()




# ---------- SIMULATE DATA ---------- #

for step in range(steps):
    # Get ID for this step, filled with 0s so sorted properly
    step_id = str(step+1)
    step_id = step_id.zfill(len(str(steps)))
    print(step_id)

    current_n = n[step]


    # Save model parameters for reference
    model_params = {'n': current_n,
                    'p_x': p_x,
                    'p_y': p_y}

    model_params = pd.Series(model_params)
    step_dir = f'{working_dir}/step_{step_id}'
    os.makedirs(step_dir, exist_ok=True)
    model_params.to_csv(f'{step_dir}/model_params.csv', header=False)


    # For each iteration, generate a PLS covariance matrix and a corresponding dataset with n subjects
    for draw in range(draws):
        draw_id = str(draw+1)
        draw_id = draw_id.zfill(len(str(draws)))
        
        # Generate random X and Y
        X = np.random.randn(current_n, p_x)
        Y = np.random.randn(current_n, p_y)

        # Save in run- and dataset-specific output directory
        draw_dir = f'{step_dir}/draw_{draw_id}'
        os.makedirs(draw_dir, exist_ok=True)

        np.savetxt(f'{draw_dir}/X_{step_id}_{draw_id}.csv', X_white, delimiter=',')
        np.savetxt(f'{draw_dir}/Y_{step_id}_{draw_id}.csv', Y_white, delimiter=',')
