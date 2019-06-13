#!/bin/bash

#SBATCH --job-name=comb
#SBATCH --output=log_combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# ./combine_samples.py --regex train_fix "train_fix_\d+" --dir /scratch/jb6504/StrongLensing-Inference
#./combine_samples.py --regex test_fix_point "test_fix_\d+" --dir /scratch/jb6504/StrongLensing-Inference
#./combine_samples.py --regex test_fix_prior "test_fix_prior_\d+" --dir /scratch/jb6504/StrongLensing-Inference

./combine_samples.py --regex train_mass "train_mass_\d+" --dir /scratch/jb6504/StrongLensing-Inference
./combine_samples.py --regex test_mass_point "test_mass_\d+" --dir /scratch/jb6504/StrongLensing-Inference
./combine_samples.py --regex test_mass_prior "test_mass_prior_\d+" --dir /scratch/jb6504/StrongLensing-Inference

# ./combine_samples.py --regex train_align "train_align_\d+" --dir /scratch/jb6504/StrongLensing-Inference
# ./combine_samples.py --regex test_align_point "test_align_\d+" --dir /scratch/jb6504/StrongLensing-Inference
# ./combine_samples.py --regex test_align_prior "test_align_prior_\d+" --dir /scratch/jb6504/StrongLensing-Inference

# ./combine_samples.py --regex train_full "train_full_\d+" --dir /scratch/jb6504/StrongLensing-Inference
# ./combine_samples.py --regex test_full_point "test_full_\d+" --dir /scratch/jb6504/StrongLensing-Inference
# ./combine_samples.py --regex test_full_prior "test_full_prior_\d+" --dir /scratch/jb6504/StrongLensing-Inference
