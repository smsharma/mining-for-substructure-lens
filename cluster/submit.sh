#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster


############################################################
# Simulation
############################################################

# sbatch --array=0-999 simulate_train.sh
# sbatch --array=0-99 simulate_train_pointref.sh

# sbatch --array=0-624 simulate_calibration.sh
# sbatch simulate_calibration_ref.sh

# sbatch --array=0-9 simulate_test_point.sh
# sbatch --array=0-9 simulate_test_prior.sh


############################################################
# Combination
############################################################

# sbatch combine_samples.sh


############################################################
# Training
############################################################

# sbatch train_carl.sh
# sbatch train_alice.sh
# sbatch train_alices.sh

# sbatch train_carl_aux.sh
# sbatch train_alice_aux.sh
# sbatch train_alices_aux.sh


############################################################
# Evaluation
############################################################

sbatch eval_carl.sh
sbatch eval_alice.sh
sbatch eval_alices.sh

# sbatch eval_carl_aux.sh
# sbatch eval_alice_aux.sh
# sbatch eval_alices_aux.sh


############################################################
# Calibration
############################################################

# To do
