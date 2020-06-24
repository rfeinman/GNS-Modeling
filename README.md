# Generative Neuro-Symbolic (GNS) Modeling

## Setup

This code repository requires Python 3 with PyTorch >= 1.0.0. 
You must have the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) installed in order to call the bottom-up parser from the [BPL matlab code](https://github.com/brendenlake/BPL). 
Make sure that you have downloaded the BPL matlab repository and created an environment variable `BPL_PATH` that points to the base directory.
This is the only external non-python utility that we use.

In order to use the provided sources code modules, make sure to add the `src/` folder to your python path (Unix machines):
 ```
export PYTHONPATH="/path/to/GNSModeling/src:$PYTHONPATH"
```

## Running Experiments

After completing the setup steps above, navigate to the `experiments/` directory to find scripts for running our experiments with the GNS model. 
Each experiment has its own unique sub-directory and a README with directions for running the experiments.
We are still working to clean up some of the experiment code and appologize if some pieces are missing or unclear.
