#!/bin/bash

#SBATCH --job-name=tr-c-pre
#SBATCH --output=log_train_carl_pre.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py carl train_mass carl_mass_pre --load carl_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_align carl_align_pre --load carl_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_full carl_full_pre --load carl_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_full carl_full_pre_aux -z --load carl_fix --dir /scratch/jb6504/StrongLensing-Inference
