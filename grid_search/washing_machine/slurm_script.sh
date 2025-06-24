#!/bin/bash
#SBATCH --job-name=washing_machine_grid_search
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davide.morelli@unifr.ch

# Activate your Python virtual environment
source .venv/bin/activate
module load cuda/11.8

# Run the script
python3 grid_search.py
