#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=integrate
#SBATCH --output=integrate_%A_%a.out
#SBATCH --error=integrate_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array=2000-2022

module load python

YEAR=${SLURM_ARRAY_TASK_ID}

cyclone-energetics integrate-fluxes \
    --te-directory /project2/tas1/gmsarro/cyclone_energetics/output/TE \
    --dhdt-directory /project2/tas1/gmsarro/tend \
    --vint-directory /project2/tas1/gmsarro/vint \
    --radiation-directory /project2/tas1/gmsarro/ERA5/rad \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/integrated \
    --year-start "${YEAR}" \
    --year-end "$((YEAR + 1))"
