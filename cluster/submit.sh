#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster

# sbatch --array=0-99 simulate_train.sh
# sbatch --array=0-9 simulate_train_point.sh
# sbatch --array=0-9 simulate_test.sh

# sbatch combine_samples.sh

# sbatch train_debug.sh
sbatch train_carl.sh
sbatch train_alice.sh
sbatch train_carl_point.sh
sbatch train_alice_point.sh
sbatch train_carl_log.sh
sbatch train_alice_log.sh
