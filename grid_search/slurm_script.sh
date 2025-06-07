#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davide.morelli@unifr.ch

source .venv/bin/activate
python3 grid_search.py