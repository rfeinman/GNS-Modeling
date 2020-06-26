#!/usr/local/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='turing|volta'
#SBATCH --time=8:00:00
#SBATCH --job-name=osg
#SBATCH --output=osg_%A_%a.out

module purge
module load cuda-10.2

cd /path/to/generate_exemplars
python get_parses.py --batch_ix=$SLURM_ARRAY_TASK_ID