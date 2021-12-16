#!/bin/bash
#SBATCH --time=00:04:00
#SBATCH -p class --gres=gpu:1
#SBATCH -o data_generator_log_slurm.log
{ time ./data_generator_run.sh; } > ./data_generator_log_blend.log 2>&1
