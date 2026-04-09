#!/bin/bash
#SBATCH --account=pi-tas1
#SBATCH --partition=broadwl
#SBATCH --job-name=nb_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/midway2/gmsarro/nb_full_%j.out
#SBATCH --error=/scratch/midway2/gmsarro/nb_full_%j.err

set -uo pipefail

module unload python 2>/dev/null || true
module load python/miniforge-25.3.0 2>/dev/null || true
source /software/python-miniforge-25.3.0-el7-x86_64/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /home/gmsarro/.conda/envs/notebook_env 2>/dev/null || conda activate notebook_env 2>/dev/null || true

echo "Python: $(which python) — $(python --version 2>&1)"

REPO=/project2/tas1/gmsarro/mse_cyclone-energetics
SCRATCH_OUT=/scratch/midway2/gmsarro/final_figures
mkdir -p "$SCRATCH_OUT"

cd "$REPO/notebooks"

python3 - "$SCRATCH_OUT" <<'PYSCRIPT'
import json
import sys

out_dir = sys.argv[1]
nb = json.load(open("final_figures.ipynb", encoding="utf-8"))
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        for i, line in enumerate(cell["source"]):
            if 'OUT_DIR = "."' in line or "OUT_DIR = '.'" in line:
                cell["source"][i] = 'OUT_DIR = "%s"\n' % out_dir
                print("Patched OUT_DIR in cell %d -> %s" % (idx, out_dir))
last_code = max(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
nb["cells"][last_code]["source"] = ["# skipped transport figure for speed\n"]
print("Blanked last code cell %d (transport figure) to save time." % last_code)
with open("/tmp/final_figures_scratch.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f)
print("Prepared notebook.")
PYSCRIPT

echo "=== Executing full notebook at $(date) (per-cell timeout 3h) ==="
jupyter nbconvert --to notebook --execute \
    --output /tmp/final_figures_executed.ipynb \
    --ExecutePreprocessor.timeout=10800 \
    --ExecutePreprocessor.kernel_name=python3 \
    /tmp/final_figures_scratch.ipynb
rc=$?
echo "=== nbconvert exit code: $rc at $(date) ==="

if [ ! -f /tmp/final_figures_executed.ipynb ]; then
    echo "No executed notebook output."
    exit 1
fi

python3 - "$REPO/notebooks/final_figures.ipynb" "$SCRATCH_OUT" <<'FIXOUT'
import json
import sys

nb_path, out_dir = sys.argv[1], sys.argv[2]
executed = json.load(open("/tmp/final_figures_executed.ipynb", encoding="utf-8"))
for cell in executed["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if "OUT_DIR = " in line and out_dir.rstrip("/") in line:
                cell["source"][i] = 'OUT_DIR = "."\n'
                break
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(executed, f, indent=1)
    f.write("\n")
print("Saved executed notebook to %s (OUT_DIR restored to default)." % nb_path)
FIXOUT

echo "=== Copying figures from scratch to repo notebooks/ ==="
cp "$SCRATCH_OUT"/*.png "$REPO/notebooks/" 2>/dev/null || true
cp "$SCRATCH_OUT"/*.pdf "$REPO/notebooks/" 2>/dev/null || true
ls -lh "$SCRATCH_OUT"/*.png 2>/dev/null | head -20 || true

if [ "$rc" -ne 0 ]; then
    echo "WARNING: nbconvert reported failure; notebook file was still written."
    exit "$rc"
fi

echo "=== Git commit and push (notebook with embedded outputs) ==="
GIT=/software/git-2.10-el7-x86_64/bin/git
cd "$REPO"
"$GIT" add notebooks/final_figures.ipynb
if "$GIT" diff --cached --quiet; then
    echo "Nothing staged; skipping commit."
else
    TREE=$("$GIT" write-tree)
    PARENT=$("$GIT" rev-parse HEAD)
    COMMIT=$(
        echo "Side-by-side layout for 2d_maps_PW figure" |
            "$GIT" commit-tree "$TREE" -p "$PARENT"
    )
    "$GIT" update-ref refs/heads/$("$GIT" symbolic-ref --short HEAD) "$COMMIT"
    echo "Commit: $COMMIT"
    env -i HOME="$HOME" PATH=/software/git-2.10-el7-x86_64/bin:/usr/bin:/bin \
        SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
        "$GIT" -C "$REPO" push origin HEAD || echo "WARNING: git push failed; push manually from a node with GitHub access."
fi

echo "=== Done at $(date) ==="
