#!/bin/bash

cd /scratch/jb6504/recycling_strong_lensing/cluster


############################################################
# Simulation
############################################################

sbatch --array=0-499 simulate_train.sh
# sbatch --array=0-624 simulate_calibration.sh
# sbatch simulate_calibration_ref.sh
# sbatch --array=0-9 simulate_test.sh


############################################################
# Combination
############################################################

# sbatch combine_samples.sh


############################################################
# Training
############################################################

# sbatch train_carl.sh
# sbatch train_alices.sh
# sbatch train_carl_full.sh
# sbatch train_alices_full.sh
# sbatch train_carl_pre.sh
# sbatch train_alices_pre.sh


############################################################
# Evaluation
############################################################

# sbatch eval_carl.sh
# sbatch eval_alices.sh


############################################################
# Calibration
############################################################

# sbatch calibrate.sh
