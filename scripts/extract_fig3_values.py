"""Print the Fig. 3c-f bar values (winter - summer, PW) for a given sampled-flux file.

Executes the data cells of notebooks/final_figures.ipynb in a private namespace
with NC_FLUX pointed at the requested WITH_INT file, then reproduces the bar
arithmetic of the Fig. 3 plotting cell. Used to quantify how the corrected
storage term (v3) changes the SHF / storage / residual bars relative to the
published file.

usage: python extract_fig3_values.py [--nc PATH] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE.parent / "notebooks" / "final_figures.ipynb"
DATA_CELLS = (1, 2, 4, 6, 8, 9)  # constants, helpers, stormtrack, area means, D_I


def run_notebook_data(nc_flux: str | None):
    nb = json.load(open(NB))
    ns: dict = {"__name__": "__nb__"}
    for i in DATA_CELLS:
        src = "\n".join(
            ln for ln in "".join(nb["cells"][i]["source"]).splitlines() if not ln.lstrip().startswith("%")
        )
        if i == 1 and nc_flux is not None:
            src += f"\nNC_FLUX = pathlib.Path({nc_flux!r})\n"
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    # Fig. 3a,b (total Eulerian by population): run cell 21 up to the pop_bars loop only
    src21 = "".join(nb["cells"][21]["source"])
    cut = src21.find("def _te_track_series_12")
    exec(compile(src21[:cut], "<cell 21 head>", "exec"), ns)
    return ns


def bar_values(ns, hemi, icut):
    d = {}
    for k, v in ns["DI_weak"].items():
        if not k.startswith("stormtrack"):
            d[k] = v
    for k, v in ns["DI_strong"].items():
        if not k.startswith("stormtrack"):
            d[k] = v
    pw = 1e15 * ns["PW_FACTOR"]
    w, s = [11, 0, 1], [5, 6, 7]

    def ws(x):
        return float(np.mean(x[w]) - np.mean(x[s]))

    te = d[f"D_I_{hemi}_F_TE{icut}"] * pw
    rad = (d[f"D_I_{hemi}_F_Swabs{icut}"] + d[f"D_I_{hemi}_F_Olr{icut}"]) * pw
    shf = d[f"D_I_{hemi}_F_Shf{icut}"] * pw
    dhdt = -d[f"D_I_{hemi}_F_Dhdt{icut}"] * pw
    umz = -d[f"D_I_{hemi}_F_UM_z{icut}"] * pw
    resid = (
        d[f"D_I_{hemi}_F_TE{icut}"]
        - (d[f"D_I_{hemi}_tot_energy{icut}"] - d[f"D_I_{hemi}_F_Dhdt{icut}"])
        + d[f"D_I_{hemi}_F_UM_z{icut}"]
    ) * pw
    return {
        "bars_W-S_PW": {
            "TE": ws(te), "RAD": ws(rad), "SHF": ws(shf),
            "-dhdt": ws(dhdt), "-ZA": ws(umz), "-R": ws(resid),
        },
        "monthly_PW": {"SHF": (shf).tolist(), "-dhdt": dhdt.tolist(), "-R": resid.tolist()},
    }


def land_ocean_values(ns, nc_flux):
    """Fig. 4 (NH land/ocean) bars: run every code cell up to the Fig. 4 cell with saving disabled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nb = json.load(open(NB))
    plt.Figure.savefig = lambda *a, **k: None
    plt.savefig = lambda *a, **k: None
    plt.show = lambda *a, **k: None
    for i, c in enumerate(nb["cells"][:26]):
        if c["cell_type"] != "code" or i in DATA_CELLS:
            continue
        src = "\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith("%"))
        exec(compile(src, f"<cell {i}>", "exec"), ns)
        plt.close("all")
    return {k: [float(v) for v in vals] for k, vals in ns["results_land_ocean"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--landocean", action="store_true", help="also compute the Fig. 4 land/ocean bars")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE.parent))
    ns = run_notebook_data(a.nc)
    print("NC_FLUX:", ns["NC_FLUX"])
    out = {}
    names = ("TE", "RAD", "SHF", "-dhdt", "-ZA", "-R")
    for (h, pop), bars in ns["pop_bars"].items():
        key = f"Eulerian {h} {pop}"
        out[key] = {"bars_W-S_PW": dict(zip(names, map(float, bars)))}
        print(f"{key:24s} " + "  ".join(f"{k}={v:+.3f}" for k, v in out[key]["bars_W-S_PW"].items()))
    for hemi, icut, name in [("SH", 0, "SH weak"), ("NH", 0, "NH weak"), ("SH", 5, "SH intense"), ("NH", 5, "NH intense")]:
        out[name] = bar_values(ns, hemi, icut)
        b = out[name]["bars_W-S_PW"]
        print(f"{name:24s} " + "  ".join(f"{k}={v:+.3f}" for k, v in b.items()))
    if a.landocean:
        lo = land_ocean_values(ns, a.nc)
        for key, vals in lo.items():
            out[f"Fig4 {key}"] = {"bars_W-S_PW": dict(zip(names, vals))}
            print(f"{'Fig4 ' + key:24s} " + "  ".join(f"{k}={v:+.3f}" for k, v in zip(names, vals)))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
