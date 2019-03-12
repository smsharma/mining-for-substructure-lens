#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --output=log_combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

./combine_samples.py --regex train "train\d+"
./combine_samples.py --regex test "test\d+"
