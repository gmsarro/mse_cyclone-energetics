#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=tracks
#SBATCH --output=tracks_%j.out
#SBATCH --error=tracks_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G

module load python

cyclone-energetics process-tracks \
    --track-directory /project2/tas1/gmsarro/track/raw \
    --output-directory /project2/tas1/gmsarro/cyclone_energetics/output/tracks
