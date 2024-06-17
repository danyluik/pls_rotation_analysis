## Description

This repository contains the code necessary to reproducibly generate and analyze the simulated data described in (link submission).

All simulated data were generated using the [GEMMR](https://github.com/murraylab/gemmr) package, and all PLS analyses were performed using a modified version of [pyls](https://github.com/rmarkello/pyls). 



## Usage

1. Environment setup
The packages required and installation instructions can be found in ``requirements.txt`.


2. Simulating data
In ``generate_data``, any of the ``simulate_X.py`` files can be run with:

```
python simulate_X.py /path/to/outputs
```

The output directory will contain 1000 simulated X and Y matrices, each in a separate directory. The directory structure will have 10 "step" folders (e.g., one per sample size), and 100 "draw" folders inside each (i.e., one for each dataset within a given sample size).


3. Running PLS
In ``pyls``, ``run_pls_simulated.py``` can be run with:

```
python run_pls_simulated.py /path/to/single/dataset
```

This will create an ``output`` folder within the single dataset folder, containing various PLS outcome metrics (covariance explained, split-half stability, significance, etc.)

To speed up processing, a list of jobs can be created and submitted to a high-performance computing cluster.


4. Analyzing results

From the ``analyze_results`` folder, a set of PLS results across the 1000 can be organized into single ``.pkl`` files using:

```
python organize_data.py /path/to/simulated/data
```

The output can then be analyzed, creating the plots seen in the manuscript, with:

```
python analyze_results.py /path/to/simulated_data
```



## Other files

The scripts used to analyze the UK BioBank and PEPP datasets, which cannot be openly shared, are also provided for transparency. 
