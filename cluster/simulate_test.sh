#!/bin/bash

#SBATCH --job-name=sim-ept
#SBATCH --output=log_simulate_testpoint_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py --fixz --fixm --fixalign -n 10000 --name test_fix_${SLURM_ARRAY_TASK_ID} --test --point --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py --fixz --fixalign -n 10000 --name test_mass_${SLURM_ARRAY_TASK_ID} --test --point --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py --fixz --fixm -n 10000 --name test_align_${SLURM_ARRAY_TASK_ID} --test --point --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py -n 10000 --name test_full_${SLURM_ARRAY_TASK_ID} --test --point --dir /scratch/jb6504/StrongLensing-Inference

python -u simulate.py --fixz --fixm --fixalign -n 10000 --name test_fix_prior_${SLURM_ARRAY_TASK_ID} --test --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py --fixz --fixalign -n 10000 --name test_mass_prior_${SLURM_ARRAY_TASK_ID} --test --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py --fixz --fixm -n 10000 --name test_align_prior_${SLURM_ARRAY_TASK_ID} --test --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py -n 10000 --name test_full_prior_${SLURM_ARRAY_TASK_ID} --test --dir /scratch/jb6504/StrongLensing-Inference

