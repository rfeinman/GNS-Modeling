# Generate new exemplars

Before you begin, unzip the file `targets.zip` to produce folder `targets/`.

### Step 1: get parses for target images

To compute the posterior parses for all 50 target images, first run the following slurm script:

```
sbatch --array=0-4 get_parses.sh
```

We have included pre-computed parses obtained from this step, accessed by unzipping `parses.zip`.


### Step 2: generate new exemplars

To show a grid of 9 new GNS exemplars for each of the 50 target images, run the following:

```
python generate.py
```