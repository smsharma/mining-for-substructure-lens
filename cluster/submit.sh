#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster



# sbatch --array=0-99 simulate_train.sh
# sbatch --array=0-9 simulate_train_point.sh
# sbatch --array=0-9 simulate_test.sh
sbatch --array=0-9 simulate_test_prior.sh

# sbatch combine_samples.sh



sbatch train_debug.sh

sbatch train_carl.sh
sbatch train_alice.sh
sbatch train_alices.sh

sbatch train_carl_deep.sh
sbatch train_alice_deep.sh
sbatch train_alices_deep.sh

sbatch train_carl_point.sh
sbatch train_alice_point.sh
