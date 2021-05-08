#!/bin/env bash
#SBATCH --output=res.txt
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 12      # cores requested
#SBATCH --mem=100000  # memory in Mb
#SBATCH -t 100:00:00  # time requested in hour:minute:second
# request gpu partition specfically
python runner.py
