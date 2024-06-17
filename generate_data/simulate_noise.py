import numpy as np
import os
import pandas as pd
import sys

from gemmr.generative_model import GEMMR

np.random.seed(22222)


"""
Usage: python simulate_noise.py /path/to/outputs

Generates datasets with GEMMR, with varying nosie levels (100 datasets at each of 10 standard deviations of Gaussian noise).
Each pair of X and Y matrices is saved under the working directory, in noise/data/step_X/draw_Y

"""


# ---------- SETUP ----------#

steps = 10 # Number of different noise levels
draws = 100 # Number of draws per step

noise_std = np.linspace(0, 1, steps) # sigma of Gaussian noise

n = 1000 # sample size
p_x = 90 # x features
p_y = 10 # y features
m = 1 # cross-block encodings
r_between = 0.3 # strength of strongest cross-block encoding

working_dir = f'{working_dir}/noise'
os.makedirs(working_dir, exist_ok=True)

np.savetxt(f'{working_dir}/noise_std.csv', noise_std, delimiter=',')

description = 'Vary std of Gaussian added to X and Y, 10 values linearly spaced between 0 and 1. 100 draws of each. \n \
               90 X, 10 Y, 1000 samples, CC of 0.3. All other params author-recommended.'

out_file = open(f'{working_dir}/run_description.txt', 'w')
out_file.write(f'{description} \n')
out_file.close()




# ---------- SIMULATE DATA ---------- #

for step in range(steps):
    # Get ID for this step, filled with 0s so sorted properly
    step_id = str(step+1)
    step_id = step_id.zfill(len(str(steps)))
    print(step_id)

    current_noise = noise_std[step]


    # Save model parameters for reference
    model_params = {'n': n,
                    'p_x': p_x,
                    'p_y': p_y,
                    'm': m,
                    'r_between': r_between,
                    'noise_std': current_noise}

    model_params = pd.Series(model_params)
    step_dir = f'{working_dir}/step_{step_id}'
    os.makedirs(step_dir, exist_ok=True)
    model_params.to_csv(f'{step_dir}/model_params.csv', header=False)


    # For each iteration, generate a PLS covariance matrix and a corresponding dataset with n subjects
    for draw in range(draws):
        draw_id = str(draw+1)
        draw_id = draw_id.zfill(len(str(draws)))

        gm = GEMMR(model='pls', px=p_x, py=p_y, m=m, r_between=r_between, random_state=np.random.choice(10000))
        
        X, Y = gm.generate_data(n)

        # Generate an X- and Y-sized matrix of Gaussian noise
        X_noise = np.random.normal(0, current_noise, size=(n,p_x))
        Y_noise = np.random.normal(0, current_noise, size=(n,p_y))

        # Add to original matrix
        X = np.add(X, X_noise)
        Y = np.add(Y, Y_noise)

        # Save in run- and dataset-specific output directory
        draw_dir = f'{step_dir}/draw_{draw_id}'
        os.makedirs(draw_dir, exist_ok=True)

        np.savetxt(f'{draw_dir}/X_{step_id}_{draw_id}.csv', X, delimiter=',')
        np.savetxt(f'{draw_dir}/Y_{step_id}_{draw_id}.csv', Y, delimiter=',')
