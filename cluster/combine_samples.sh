#!/bin/bash

#SBATCH --job-name=comb
#SBATCH --output=log_combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

./combine_samples.py --regex train "train_\d+" --dir /scratch/jb6504/StrongLensing-Inference
./combine_samples.py --regex test_point "test_\d+" --dir /scratch/jb6504/StrongLensing-Inference
./combine_samples.py --regex test_prior "test_prior\d+" --dir /scratch/jb6504/StrongLensing-Inference
