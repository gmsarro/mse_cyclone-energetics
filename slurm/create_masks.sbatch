#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=masks
#SBATCH --output=masks_%A_%a.out
#SBATCH --error=masks_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --array=0-1

module load python

HEMISPHERES=("NH" "SH")
HEMI=${HEMISPHERES[$SLURM_ARRAY_TASK_ID]}

cyclone-energetics create-masks \
    --hemisphere "${HEMI}" \
    --vorticity-directory /project2/tas1/gmsarro/track/vorticity \
    --track-directory /project2/tas1/gmsarro/cyclone_energetics/output/tracks \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/masks_"${HEMI}" \
    --year-start 2000 \
    --year-end 2023
