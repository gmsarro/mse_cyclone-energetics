#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=composites
#SBATCH --output=composites_%j.out
#SBATCH --error=composites_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

module load python

cyclone-energetics build-composites \
    --track-directory /project2/tas1/gmsarro/track/raw \
    --integrated-flux-directory /project2/tas1/gmsarro/cyclone_energetics/output/integrated \
    --mask-directory-nh /project2/tas1/gmsarro/cyclone_energetics/output/masks_NH \
    --mask-directory-sh /project2/tas1/gmsarro/cyclone_energetics/output/masks_SH \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/composites \
    --year-start 2000 \
    --year-end 2023
