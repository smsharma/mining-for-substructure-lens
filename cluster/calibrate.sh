#!/bin/bash

#SBATCH --job-name=cal
#SBATCH --output=log_calibrate.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

DIR=/Users/johannbrehmer/work/projects/strong_lensing/StrongLensing-Inference
# DIR=/scratch/jb6504/StrongLensing-Inference/

source activate lensing
cd $DIR

python -u calibrate.py carl_aux_grid carl_aux_calibrate --dir $DIR
python -u calibrate.py alice_aux_grid alice_aux_calibrate --dir $DIR
python -u calibrate.py alices_aux_grid alices_aux_calibrate --dir $DIR

python -u calibrate.py carl_pointref_aux_grid carl_pointref_aux_calibrate --dir $DIR
python -u calibrate.py alice_pointref_aux_grid alice_pointref_aux_calibrate --dir $DIR
python -u calibrate.py alices_pointref_aux_grid alices_pointref_aux_calibrate --dir $DIR
