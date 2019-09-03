#!/bin/bash

#SBATCH --job-name=sim-cal
#SBATCH --output=log_simulate_calibration_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm --fixalign -n 5000 --name calibrate_fix_theta_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/recycling_strong_lensing
python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixalign -n 5000 --name calibrate_mass_theta_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/recycling_strong_lensing
python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm -n 5000 --name calibrate_align_theta_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/recycling_strong_lensing
python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} -n 5000 --name calibrate_full_theta_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/recycling_strong_lensing
