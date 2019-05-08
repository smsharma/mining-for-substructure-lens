#!/bin/bash

#SBATCH --job-name=e-a
#SBATCH --output=log_eval_alice.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

for i in {0..624}
do
    echo ""
    echo ""
    echo ""
    echo "EVALUATING CALIB $i"
    echo ""
    python -u test.py alice calibrate_theta$i alice_calibrate_theta$i --dir /scratch/jb6504/StrongLensing-Inference
done
    echo ""
    echo ""
    echo ""
    echo "EVALUATING CALIB REF"
    echo ""
python -u test.py alice calibrate_ref alice_calibrate_ref --dir /scratch/jb6504/StrongLensing-Inference

# python -u test.py alice test_prior alice_prior --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py alice test_prior alice_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference
# python -u test.py alice test_point alice_grid --grid --dir /scratch/jb6504/StrongLensing-Inference
