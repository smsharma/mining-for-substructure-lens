#!/bin/bash

#SBATCH --job-name=e-as-a
#SBATCH --output=log_eval_alices_aux.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/
#
# for i in {0..624}
# do
#     echo ""
#     echo ""
#     echo ""
#     echo "EVALUATING CALIB $i"
#     echo ""
#     python -u test.py alices_aux calibrate_theta$i alices_aux_calibrate_theta$i --aux z --dir /scratch/jb6504/StrongLensing-Inference
# done

echo ""
echo ""
echo ""
echo "EVALUATING CALIB REF"
echo ""
python -u test.py alices_aux calibrate_ref alices_aux_calibrate_ref --grid --aux z --dir /scratch/jb6504/StrongLensing-Inference

# echo ""
# echo ""
# echo ""
# echo "EVALUATING PRIOR SAMPLE"
# echo ""
# python -u test.py alices_aux test_prior alices_aux_prior --aux z --dir /scratch/jb6504/StrongLensing-Inference
#
# echo ""
# echo ""
# echo ""
# echo "EVALUATING PRIOR SAMPLE (SHUFFLED)"
# echo ""
# python -u test.py alices_aux test_prior alices_aux_shuffledprior --aux z --shuffle --dir /scratch/jb6504/StrongLensing-Inference

# echo ""
# echo ""
# echo ""
# echo "EVALUATING POINT SAMPLE ON PARAM GRID"
# echo ""
# python -u test.py alices_aux test_point alices_aux_grid --aux z --grid --dir /scratch/jb6504/StrongLensing-Inference
