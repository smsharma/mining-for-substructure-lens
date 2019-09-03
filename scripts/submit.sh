#!/bin/bash

cd /scratch/jb6504/recycling_strong_lensing/cluster


############################################################
# Simulation
############################################################

sbatch --array=0-499 simulate_train.sh
sbatch --array=0-624 simulate_calibration.sh
sbatch simulate_calibration_ref.sh
sbatch --array=0-9 simulate_test.sh


############################################################
# Combination
############################################################

sbatch combine_samples.sh


############################################################
# Training
############################################################

sbatch train_carl.sh
sbatch train_alices.sh
sbatch train_alpha.sh
sbatch train_lr.sh
sbatch train_sgd.sh
sbatch train_other.sh


############################################################
# Evaluation
############################################################

sbatch eval_carl.sh
sbatch eval_alices.sh
sbatch eval_alpha.sh
sbatch eval_lr.sh
sbatch eval_sgd.sh
sbatch eval_other.sh


############################################################
# Calibration
############################################################

sbatch calibrate.sh
