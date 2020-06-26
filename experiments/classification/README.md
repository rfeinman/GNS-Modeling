# One-shot classification

Before you begin, unzip the file `one-shot-classification.zip` to produce folder `one-shot-classification/`. This is a folder of Omniglot images for the 20-way within-alphabet classification task of Lake et al. (2015). It was taken directly from the [Omniglot repository](https://github.com/brendenlake/omniglot/tree/master/python/one-shot-classification).

### Step 1: compute 'base' training parses for all images

To start, run the following:

```
sbatch --array=0-19 multi_parsing.sh
```

### Step 2: optimize training parses

Once step 1 has complete, run the following:

```
sbatch --array=0-19 multi_optimize.sh
```

### Step 3: refit training parses to test images

Once step 2 has complete, run the following:
```
sbatch --array=0-19 multi_refit_gpu.sh
```