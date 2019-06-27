#!/bin/bash

#SBATCH --job-name=tr-c-e
#SBATCH --output=log_train_carl_exp.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py carl train_fix carl_fix_largelr --lr 0.01 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_mass carl_mass_largelr --lr 0.01 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_align carl_align_largelr --lr 0.01 --dir /scratch/jb6504/StrongLensing-Inference

python -u train.py carl train_fix carl_fix_smalllr --lr 0.0001 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_mass carl_mass_smalllr --lr 0.0001 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_align carl_align_smalllr --lr 0.0001 --dir /scratch/jb6504/StrongLensing-Inference
