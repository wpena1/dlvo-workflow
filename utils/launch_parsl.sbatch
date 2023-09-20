#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 2GB
#SBATCH -t 60:00:00

source /scratch/wpc252/miniconda3/etc/profile.d/conda.sh
conda activate wepy3

python run_parameter_sweep_dlvo.py -a 5 6 -DL 5.25 -FP 0.5 -LS 6 7 7.5 -RP 85 -RN 105 -seed 10
