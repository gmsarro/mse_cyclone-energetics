#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=assign
#SBATCH --output=assign_%j.out
#SBATCH --error=assign_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

module load python

cyclone-energetics assign-fluxes \
    --integrated-flux-directory /project2/tas1/gmsarro/cyclone_energetics/output/integrated \
    --mask-directory-nh /project2/tas1/gmsarro/cyclone_energetics/output/masks_NH \
    --mask-directory-sh /project2/tas1/gmsarro/cyclone_energetics/output/masks_SH \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/assigned \
    --year-start 2000 \
    --year-end 2023
