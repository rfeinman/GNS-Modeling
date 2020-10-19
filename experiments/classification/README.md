# One-shot classification

## 1) Download dataset
Before you begin, unzip the file `one-shot-classification.zip` to produce folder `one-shot-classification/`. This is a folder of Omniglot images for the 20-way within-alphabet classification task of Lake et al. (2015). It was taken directly from the [Omniglot repository](https://github.com/brendenlake/omniglot/tree/master/python/one-shot-classification).

## 2) Run experiments (on cluster)
The one-shot classification experiments require a multi-GPU cluster to complete in reasonable time. I've included a python script to execute cluster jobs using the [submitit](https://github.com/facebookincubator/submitit) package for Slurm cluster scheduling. To execute all jobs for the forward classification score (explained in paper Section 4), run the following command:

```
python run_classification.py
```

Once completed, you will need to re-run this script for the reverse direction to obtain the full two-way criterion of Eq. 5:

```
python run_classification.py --reverse
```

As a reminder, computation for each the forward and reverse direction involves a series of 3 steps (explained in paper Section 4 & Appendix B):
1. Compute 'base' parses for all training images `i`
2. Optimize parses for all training images `i`
3. Refit each training parse `i` to each test image `j`