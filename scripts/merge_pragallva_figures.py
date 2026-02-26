#!/usr/bin/env python3
"""Add Pragallva figures (A–E) from plots_for_pragallva.ipynb into final_figures.ipynb."""
import json
import re

PRAGALLAVA_NB = "/project2/tas1/gmsarro/cyclone_centered/plots_for_pragallva.ipynb"
FINAL_NB = "/project2/tas1/gmsarro/mse_cyclone-energetics/notebooks/final_figures.ipynb"
OUT_NB = "/project2/tas1/gmsarro/mse_cyclone-energetics/notebooks/final_figures.ipynb"


def adapt_paths(code: str) -> str:
    """Replace OUT_DIR and BASE paths so figures save to cwd and paths use BASE (pathlib)."""
    code = re.sub(
        r"os\.path\.join\s*\(\s*OUT_DIR\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
        r"'\1'",
        code,
    )
    code = re.sub(
        r"os\.path\.join\s*\(\s*OUT_DIR\s*,\s*f['\"]([^'\"]+)['\"]\s*\)",
        r"'\1'",
        code,
    )
    code = re.sub(
        r'os\.path\.join\s*\(\s*BASE\s*,\s*"([^"]+)"\s*\)',
        lambda m: 'str(BASE / ' + ' / '.join(f'"{p}"' for p in m.group(1).split('/')) + ')',
        code,
    )
    return code


def main():
    with open(PRAGALLAVA_NB) as f:
        prag = json.load(f)
    with open(FINAL_NB) as f:
        final = json.load(f)

    # Setup code: load condensed composite, storm-track, helpers (from cells 2, 4, 6)
    # Use BASE from notebook; add NC = netCDF4.Dataset, paths as str(BASE / ...)
    setup_src = '''# --- Pragallva figures setup: condensed composites, storm-track, helpers ---
import os
from netCDF4 import Dataset as NC_P

# Paths (BASE is pathlib from first cell)
NC_FLUX = str(BASE / "cyclone_centered" / "WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc")
NC_CYC_INT = str(BASE / "track" / "final" / "cyclonic_intensity.nc")
TRACK_SH = str(BASE / "cyclone_centered" / "TRACK_VO_anom_T42_ERA5_1979_2018_allSH.nc")
TRACK_NH = str(BASE / "cyclone_centered" / "TRACK_VO_anom_T42_ERA5_1979_2018_allNH.nc")
OUT_DIR = "."

# Condensed composites (SH, 2 categories)
xr = xarray
ds = xarray.open_dataset(str(BASE / "cyclone_centered" / "Composites_monthly_condensed.nc"))

# Storm-track latitudes from flux file
with NC_P(NC_FLUX) as f:
    lat_flux = np.asarray(f["lat"][:])
    fte_all = np.asarray(f["F_TE_final"][:])
fte_total = np.sum(fte_all, axis=0)
fte_zonal = np.mean(fte_total, axis=2)
nlat = len(lat_flux)
x_d = np.linspace(0, nlat - 1, nlat)
y_d = np.linspace(0, 12, 12)
x3_d = np.linspace(0, nlat - 1, 25600)
Lat = scipy.interpolate.interp1d(x_d, lat_flux)(x3_d)
fte_interp = scipy.interpolate.RectBivariateSpline(y_d, x_d, fte_zonal)(y_d, x3_d)
stormtrack_NH = np.argmax(fte_interp, axis=1)
stormtrack_SH = np.argmin(fte_interp, axis=1)
stormtrack_lat_nh = Lat[stormtrack_NH]
stormtrack_lat_sh = Lat[stormtrack_SH]
HALF_WIN = int(25600 / 2 / 9)

R = 6.371e6
PW_factor = -2 * np.pi * R / 1e15
SEASON_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
YEAR1, YEAR2 = 2000, 2015
STEPS_PER_YEAR = 365 * 4
MLEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MCUM = np.concatenate([[0], np.cumsum(MLEN)])

colors = ["#4B0082", "#3F51B5", "#2196F3", "dodgerblue", "skyblue", "#FFFFFF", "#FFFFFF",
          "lightpink", "#FF9800", "#F44336", "#B71C1C", "maroon"]
custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("PurpleBlue_White_OrangeRed", colors, N=256)

def look_upside_down(y):
    return y[::-1, :]
def _identity(y):
    return y
def annual_mean_weighted_flux(variable, C=0, factor=1, func=_identity):
    y_ann = func(factor * ds[variable].sel(category=C)).mean(dim="month")
    return y_ann
def changing_flux_and_changing_weight(variable, C=0, DJF=[12, 1, 2], JJA=[6, 7, 8], factor=1, func=_identity):
    y_djf = func(factor * ds[variable].sel(category=C).sel(month=DJF)).mean(dim="month")
    y_jja = func(factor * ds[variable].sel(category=C).sel(month=JJA)).mean(dim="month")
    w_djf = ds.count_track.sel(category=C).sel(month=DJF).mean(dim="month")
    w_jja = ds.count_track.sel(category=C).sel(month=JJA).mean(dim="month")
    w_ann = ds.count_track.sel(category=C).mean(dim="month")
    return (y_jja * w_jja - y_djf * w_djf) / w_ann
def constant_weight(variable, C=0, DJF=[12, 1, 2], JJA=[6, 7, 8], factor=1, func=_identity):
    y_djf = func(factor * ds[variable].sel(category=C).sel(month=DJF)).mean(dim="month")
    y_jja = func(factor * ds[variable].sel(category=C).sel(month=JJA)).mean(dim="month")
    w_ann = ds.count_track.sel(category=C).mean(dim="month")
    return w_ann * (y_jja - y_djf) / w_ann
def constant_flux(variable, C=0, DJF=[12, 1, 2], JJA=[6, 7, 8], factor=1, func=_identity):
    y_ann = func(factor * ds[variable].sel(category=C)).mean(dim="month")
    w_djf = ds.count_track.sel(category=C).sel(month=DJF).mean(dim="month")
    w_jja = ds.count_track.sel(category=C).sel(month=JJA).mean(dim="month")
    w_ann = ds.count_track.sel(category=C).mean(dim="month")
    return (w_jja - w_djf) * y_ann / w_ann
def mean_around_track(field_12x_nfine, idx_12, half_win=HALF_WIN):
    out = np.zeros(12)
    for n in range(12):
        lo = max(0, idx_12[n] - half_win)
        hi = min(len(Lat), idx_12[n] + half_win)
        out[n] = np.mean(field_12x_nfine[n, lo:hi])
    return out
def interp_lat_2d(field_12x_nlat, x_d, y_d, x3_d, y3_d):
    spl = scipy.interpolate.RectBivariateSpline(y_d, x_d, field_12x_nlat)
    return spl(y3_d, x3_d)

# Alias for figure code that uses NC()
NC = NC_P
print("Pragallva setup done: ds, storm-track, helpers.")
'''

    new_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Pragallva figures (A–E)\n",
            "\n",
            "Figures from the co-author review: 2D composite of cyclone seasonality, "
            "3×3 decomposition (strong/weak), seasonal track density and f(φ), "
            "intensity histograms (summer vs winter), and per-cyclone MSE efficiency vs intensity. "
            "Outputs are saved in the current directory (e.g. `2D_composite_of_cyclone_seasonality.pdf`)."
        ]},
        {"cell_type": "code", "metadata": {}, "source": [line if line.endswith("\n") else line + "\n" for line in setup_src.split("\n")], "outputs": [], "execution_count": None},
    ]

    # Add figure cells 8, 10, 12, 14, 16 from pragallva (with path adaptations)
    for idx, label in [(8, "A"), (10, "B"), (12, "C"), (14, "D"), (16, "E")]:
        src = "".join(prag["cells"][idx]["source"])
        src = adapt_paths(src)
        # Fix BASE path: os.path.join(BASE, "a/b") already done; str(BASE / "a/b") won't work for "a/b" - need "a", "b"
        src = re.sub(r'str\(BASE / "cyclone_centered/([^"]+)"\)', r'str(BASE / "cyclone_centered" / "\1")', src)
        # Actually BASE / "cyclone_centered/Composites_Weak_SH_noleap.nc" is one path - pathlib accepts that
        new_cells.append({"cell_type": "code", "metadata": {}, "source": [line + "\n" for line in src.split("\n")], "outputs": [], "execution_count": None})

    # Find last cell index (before empty markdown if any)
    cells = final["cells"]
    insert_at = len(cells)
    for i in range(len(cells) - 1, -1, -1):
        if cells[i]["cell_type"] == "markdown" and not "".join(cells[i].get("source", [])).strip():
            insert_at = i
            break

    final["cells"] = cells[:insert_at] + new_cells + cells[insert_at:]
    with open(OUT_NB, "w") as f:
        json.dump(final, f, indent=1)
    print(f"Inserted {len(new_cells)} cells at index {insert_at}. Wrote {OUT_NB}")


if __name__ == "__main__":
    main()
