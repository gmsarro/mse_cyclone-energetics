#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=smooth
#SBATCH --output=smooth_%A_%a.out
#SBATCH --error=smooth_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --array=2000-2022

module load python

YEAR=${SLURM_ARRAY_TASK_ID}

cyclone-energetics smooth \
    --input-directory /project2/tas1/gmsarro/ERA5 \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/smoothed \
    --variables TE \
    --year-start "${YEAR}" \
    --year-end "$((YEAR + 1))"
