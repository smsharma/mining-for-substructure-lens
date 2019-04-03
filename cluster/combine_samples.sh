#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --output=log_combine2.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# ./combine_samples.py --regex train "train\d+" --dir /scratch/jb6504/StrongLensing-Inference
# ./combine_samples.py --regex train_point "train_point\d+" --dir /scratch/jb6504/StrongLensing-Inference
# ./combine_samples.py --regex test "test\d+" --dir /scratch/jb6504/StrongLensing-Inference
./combine_samples.py --regex test_prior "test_prior\d+" --dir /scratch/jb6504/StrongLensing-Inference
