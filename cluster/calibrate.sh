#!/bin/bash

#SBATCH --job-name=cal
#SBATCH --output=log_calibrate.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u calibrate.py carl_grid --dir /scratch/jb6504/StrongLensing-Inference
python -u calibrate.py alice_grid --dir /scratch/jb6504/StrongLensing-Inference
python -u calibrate.py alices_grid --dir /scratch/jb6504/StrongLensing-Inference
