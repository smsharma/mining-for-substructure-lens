#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster

sbatch --array=0-99 simulate_train.sh
sbatch --array=0-9 simulate_test.sh

# sbatch combine_samples.sh

# sbatch train_carl.sh
# sbatch train_carl_log.sh
# sbatch train_alice.sh
# sbatch train_alice_log.sh
# sbatch train_alices_alpha_0.1.sh
# sbatch train_alices_log_alpha_0.1.sh
# sbatch train_alices_alpha_1.sh
# sbatch train_alices_log_alpha_1.sh
# sbatch train_alices_alpha_10.sh
# sbatch train_alices_log_alpha_10.sh
