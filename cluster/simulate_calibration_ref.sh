#!/bin/bash

#SBATCH --job-name=sim-calref
#SBATCH --output=log_simulate_calibration_ref.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py --calref --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm --fixalign -n 5000 --name calibrate_fix_ref --dir /scratch/jb6504/recycling_strong_lensing
python -u simulate.py --calref --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixalign -n 5000 --name calibrate_mass_ref --dir /scratch/jb6504/recycling_strong_lensing
python -u simulate.py --calref --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm -n 5000 --name calibrate_align_ref --dir /scratch/jb6504/recycling_strong_lensing
python -u simulate.py --calref --theta ${SLURM_ARRAY_TASK_ID} -n 5000 --name calibrate_full_ref --dir /scratch/jb6504/recycling_strong_lensing
