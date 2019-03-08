#!/bin/bash

cd /scratch/jb6504/StrongLensing-Inference/cluster

sbatch --array=0-99 simulate_train.sh
sbatch --array=0-9 simulate_test.sh
