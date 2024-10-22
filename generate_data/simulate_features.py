import numpy as np
import os
import pandas as pd
import sys

from gemmr.generative_model import GEMMR

np.random.seed(55555)


"""
Usage: python simulate_features.py /path/to/outputs

Generates datasets with GEMMR, with varying dimensionalities of the Y matrix (100 datasets at each of 10 dimensionalities)
Each pair of X and Y matrices is saved under the working directory, in features/data/step_X/draw_Y

"""


# ---------- SETUP ---------- #

working_dir = sys.argv[1]

steps = 10 # Number of different effect strengths
draws = 100 # Number of draws per step

p_y = np.logspace(np.log10(2), np.log10(200), num=steps) # y features

n = 1000 # sample size
p_x = 90 # x features
r_between = 0.3 # strength of strongest cross-block encoding
m = 1 # cross-block encodings

working_dir = f'{working_dir}/features'
os.makedirs(working_dir, exist_ok=True)

np.savetxt(f'{working_dir}/p_y.csv', p_y, delimiter=',')

description = 'Vary Y features, 10 values logarithmically spaced between 2 and 200. 100 draws of each. \n \
               90 X, 1000 samples, CC of 0.3. All other params author-recommended.'

out_file = open(f'{working_dir}/run_description.txt', 'w')
out_file.write(f'{description} \n')
out_file.close()



# ---------- SIMULATE DATA ---------- #

for step in range(steps):
    # Get ID for this step, filled with 0s so sorted properly
    step_id = str(step+1)
    step_id = step_id.zfill(len(str(steps)))
    print(step_id)

    current_y = p_y[step]


    # Save model parameters for reference
    model_params = {'n': n,
                    'p_x': p_x,
                    'p_y': current_y,
                    'm': m,
                    'r_between': r_between}

    model_params = pd.Series(model_params).
    step_dir = f'{working_dir}/step_{step_id}'
    os.makedirs(step_dir, exist_ok=True)
    model_params.to_csv(f'{step_dir}/model_params.csv', header=False)


    # For each iteration, generate a PLS covariance matrix and a corresponding dataset with n subjects
    for draw in range(draws):
        draw_id = str(draw+1)
        draw_id = draw_id.zfill(len(str(draws)))

        gm = GEMMR(model='pls', px=p_x, py=current_y, m=m, r_between=r_between, random_state=np.random.choice(10000))
        
        X, Y = gm.generate_data(n)

        # Save in run- and dataset-specific output directory
        draw_dir = f'{step_dir}/draw_{draw_id}'
        os.makedirs(draw_dir, exist_ok=True)

        np.savetxt(f'{draw_dir}/X_{step_id}_{draw_id}.csv', X, delimiter=',')
        np.savetxt(f'{draw_dir}/Y_{step_id}_{draw_id}.csv', Y, delimiter=',')
