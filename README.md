# Generative Neuro-Symbolic (GNS) Modeling

## Setup

This code repository requires Python 3 with PyTorch. We recommend using PyTorch >= 1.5.0, as this version was used for development and testing.

In order to use the provided sources code modules, make sure to add the GNS-Modeling root folder to your python path (Unix machines):
 ```
export PYTHONPATH="/path/to/GNS-Modeling:$PYTHONPATH"
```

#### pyBPL setup

In order to use this code, you will need to pre-install the [pyBPL python library](https://github.com/rfeinman/pyBPL), a separate Python library that we also maintain. Make sure to download the repository and add its root folder to your python path, e.g.:
 ```
export PYTHONPATH="/path/to/pyBPL:$PYTHONPATH"
```

At the moment, pyBPL uses the bottom-up parser from the [BPL matlab code](https://github.com/brendenlake/BPL) during inference. Therefore to use our inference algorithms you will need to download the BPL matlab repository and install the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) to call its functions from python. Make sure you have created an environment variable `BPL_PATH` that points to the BPL base directory.
NOTE: we are working on re-implementing the bottom-up parser in pyBPL. Once complete, pyBPL will be a fully self-contained python package.

## Running Experiments

After completing the setup steps above, navigate to the `experiments/` directory to find scripts for running our experiments with the GNS model. 
Each experiment has its own unique sub-directory and a README with directions for running the experiment scripts.

## Development Notes
We are still working to clean up some of the experiment code and appologize if some pieces are missing or unclear.
