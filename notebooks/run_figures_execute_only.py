#!/usr/bin/env python3
"""
Execute final_figures.ipynb and save figures to disk without re-saving the notebook.
This avoids nbconvert validation errors on notebook save. Figures are written by
plt.savefig() in the notebook cells to the current directory.
"""
import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK = "final_figures.ipynb"

def fix_outputs(nb):
    """Ensure all outputs have required fields for execution."""
    for c in nb.get("cells", []):
        for out in c.get("outputs", []):
            if out.get("output_type") == "stream" and "name" not in out:
                out["name"] = "stdout"
            if out.get("output_type") in ("display_data", "execute_result") and "metadata" not in out:
                out["metadata"] = {}
    return nb

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isfile(NOTEBOOK):
        print(f"Not found: {NOTEBOOK}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading {NOTEBOOK}...")
    with open(NOTEBOOK) as f:
        nb = nbformat.read(f, as_version=4)
    fix_outputs(nb)
    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=7200)
    try:
        ep.preprocess(nb, {"metadata": {"path": os.getcwd()}})
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("Notebook executed successfully. Figures saved to current directory.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
