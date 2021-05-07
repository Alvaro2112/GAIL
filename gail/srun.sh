#!/bin/env bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 12      # cores requested
#SBATCH --mem=100000  # memory in Mb
#SBATCH -t 15:00:00  # time requested in hour:minute:second
#SBATCH --partition=gpu # time requested in hour:minute:second
#SBATCH --gres=gpu:1

unset DISPLAY XAUTHORITY
echo "Running ${@}"
"${@}"