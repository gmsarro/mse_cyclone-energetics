#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=validate
#SBATCH --output=validate_%j.out
#SBATCH --error=validate_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

module load python

cd /project2/tas1/gmsarro/cyclone_energetics
echo "===== Running validation tests ====="
echo "Python: $(python3 --version)"
echo "NumPy: $(python3 -c 'import numpy; print(numpy.__version__)')"
echo "Start: $(date)"
python3 tests/validate_against_reference.py
EXIT_CODE=$?
echo "End: $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
