#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster


############################################################
# Simulation
############################################################

## sbatch --array=0-99 simulate_train.sh
## sbatch --array=0-624 simulate_calibration.sh
## sbatch simulate_calibration_ref.sh
## sbatch --array=0-9 simulate_test.sh


############################################################
# Combination
############################################################

sbatch combine_samples.sh # align and mass


############################################################
# Training
############################################################

sbatch train_carl.sh  # align
sbatch train_alice.sh
sbatch train_alices.sh
sbatch train_carl2.sh  # mass
sbatch train_alice2.sh
sbatch train_alices2.sh 


############################################################
# Evaluation
############################################################

sbatch eval_carl.sh  # align
sbatch eval_alice.sh 
sbatch eval_alices.sh
sbatch eval_carl2.sh  # mass
sbatch eval_alice2.sh
sbatch eval_alices2.sh
## sbatch eval_carl3.sh
## sbatch eval_alice3.sh
## sbatch eval_alices3.sh


############################################################
# Calibration
############################################################

## sbatch calibrate.sh
