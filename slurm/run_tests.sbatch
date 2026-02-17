#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=tests
#SBATCH --output=tests_%j.out
#SBATCH --error=tests_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G

module load python

cd /project2/tas1/gmsarro/cyclone_energetics
pip install -e ".[dev]" --quiet
pytest tests/ -v --tb=short
