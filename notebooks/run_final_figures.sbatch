#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=figures_hr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project2/tas1/gmsarro/mse_cyclone-energetics/notebooks/run_final_figures_%j.out
#SBATCH --error=/project2/tas1/gmsarro/mse_cyclone-energetics/notebooks/run_final_figures_%j.err

cd /project2/tas1/gmsarro/mse_cyclone-energetics/notebooks
source activate notebook_env 2>/dev/null || conda activate notebook_env

# Fix outputs then execute notebook (no save = no validation failure)
python3 run_figures_execute_only.py

echo "Notebook execution finished. Check outputs above."
