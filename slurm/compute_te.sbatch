#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=compute_te
#SBATCH --output=compute_te_%A_%a.out
#SBATCH --error=compute_te_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array=2000-2022

module load python

YEAR=${SLURM_ARRAY_TASK_ID}

cyclone-energetics compute-te \
    --era5-base-directory /project2/tas1/gmsarro/ERA5 \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/TE \
    --year-start "${YEAR}" \
    --year-end "$((YEAR + 1))"
