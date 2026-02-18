# A Localized Moist Static Energy Budget for Cyclone Seasonality

> **What Controls the Seasonality of Intense Cyclones?**
> Giorgio M. Sarro, Pragallva Barpanda, Tiffany A. Shaw
> *Manuscript submitted to Geophysical Research Letters*

We derive a **new localized Moist Static Energy (MSE) budget** to
quantify what drives the seasonality of intense cyclones.  This
repository contains the complete analysis pipeline — from raw ERA5
reanalysis to final publication figures.  Starting from 6-hourly ERA5
fields and TRACK algorithm output, the code computes every term of the
budget (transient-eddy flux divergence, surface heat fluxes, radiation,
MSE storage, zonal MSE advection), applies cyclone masks, performs
poleward integrations, bins by cyclone intensity, builds cyclone-centred
composites, and generates all seven figures presented in the paper.

---

## Repository layout

```
cyclone_energetics/
├── pyproject.toml              # Package metadata and dependencies
├── README.md
├── LICENSE
├── ncl/
│   └── hoskins_filter.ncl     # NCL Hoskins spectral filter (driven by smoothing.py)
├── notebooks/
│   └── final_figures.ipynb     # Notebook that generates every paper figure
└── src/cyclone_energetics/
    ├── __init__.py
    ├── cli.py                  # Typer CLI (one command per pipeline step)
    ├── constants.py            # Physical constants
    ├── geometry.py             # Spherical-coordinate transforms
    ├── flux_computation.py     # Step 1a  — transient-eddy flux divergence
    ├── storage_computation.py  # Step 1b  — MSE storage term (dh/dt)
    ├── zonal_advection.py      # Step 1c  — zonal MSE advection divergence
    ├── smoothing.py            # Step 2   — CDO regridding & NCL Hoskins spectral filter
    ├── track_processing.py     # Step 3   — TRACK algorithm post-processing
    ├── masking.py              # Step 4   — cyclone/anticyclone masks
    ├── integration.py          # Step 5   — poleward flux integration
    ├── flux_assignment.py      # Step 6   — assign fluxes by intensity bin
    ├── composites.py           # Step 7a  — cyclone-centred composites (integrated fluxes)
    ├── composites_wm.py        # Step 7b  — cyclone-centred composites (W/m² budget terms)
    └── condensed_composites.py # Step 8   — condensed monthly composites
```

---

## Installation

```bash
git clone https://github.com/gmsarro/mse_cyclone-energetics.git
cd mse_cyclone-energetics
pip install -e ".[dev]"
```

Requires **Python ≥ 3.10**, [CDO](https://code.mpimet.mpg.de/projects/cdo)
(for regridding), and [NCL](https://www.ncl.ucar.edu/) (for the Hoskins
spectral filter).

---

## Data requirements

The pipeline requires:

| Dataset | Description |
|---------|-------------|
| **ERA5 6-hourly fields** | `t`, `q`, `ps`, `z`, `v`, `u`, `tsr`, `ssr`, `ttr`, `vint` (vertically integrated fluxes), `t2m` — one NetCDF per month |
| **TRACK output** | Cyclone and anticyclone track files (`TRACK_VO_anom_T42_ERA5_*.nc`, `ANTIC_VO_anom_T42_ERA5_*.nc`) |
| **Filtered vorticity** | T42 vorticity fields produced by the TRACK algorithm (`VO.anom.T42.*.nc`) |

Filenames follow the ERA5 convention used by ECMWF/CDS:
`era5_{variable}_{year}_{month}.6hrly.nc`.

---

## Processing workflow

The pipeline must be run **in order**.  Each step has a corresponding CLI
command (`cyclone-energetics <command>`) whose options are fully
documented via `--help`.

```
 ┌──────────────┐
 │   ERA5 data  │
 └──────┬───────┘
        │
        ├─── Step 1a: compute-te ──────────► TE divergence files
        ├─── Step 1b: compute-dhdt ────────► dh/dt files
        ├─── Step 1c: compute-zonal-mse ──► Zonal MSE advection files
        │
        ▼
 ┌─────────────────────────────────────────────────┐
 │  Step 2a: smooth-cdo (CDO remapbil)             │──► Regridded fields
 │  Step 2b: smooth-hoskins (Hoskins spectral)     │──► Spectrally filtered TE/vint/dh/dt
 └──────┬──────────────────────────────────────────┘
        │
        │  TRACK output
        │       │
        │       ▼
        │  Step 3: process-tracks ──────► .npz arrays
        │       │
        │       ▼
        │  Step 4: create-masks ────────► Cyclone/anticyclone masks
        │       │
        ▼       ▼
 ┌─────────────────────────────────┐
 │  Step 5: integrate-fluxes       │──► Poleward-integrated flux NetCDFs
 └──────┬──────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────┐
 │  Step 6: assign-fluxes          │──► Intensity-binned flux file
 └──────┬──────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────┐
 │  Step 7a: build-composites      │──► Composites from integrated fluxes
 └──────┬──────────────────────────┘
        │
 ┌─────────────────────────────────┐
 │  Step 7b: build-wm-composites   │──► Composites of local (W/m²) budget terms
 └──────┬──────────────────────────┘    (SHF residual, Z, VO, etc.)
        │
        ▼
 ┌─────────────────────────────────┐
 │  Step 8: condense-composites    │──► Condensed monthly file
 └──────┬──────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────┐
 │  notebooks/final_figures.ipynb  │──► All paper figures
 └─────────────────────────────────┘
```

### Running individual steps

```bash
# Example: compute transient-eddy fluxes for 2000–2022
cyclone-energetics compute-te \
    --era5-base-directory /path/to/ERA5 \
    --output-directory    /path/to/output/TE \
    --year-start 2000 \
    --year-end   2023

# Example: compute dh/dt storage term
cyclone-energetics compute-dhdt \
    --era5-base-directory /path/to/ERA5 \
    --output-directory    /path/to/output/dhdt \
    --year-start 2000 \
    --year-end   2023

# Example: compute zonal MSE advection divergence
cyclone-energetics compute-zonal-mse \
    --era5-base-directory /path/to/ERA5 \
    --output-directory    /path/to/output/zonal_mse \
    --year-start 2000 \
    --year-end   2023

# Full help for any command
cyclone-energetics integrate-fluxes --help
```

---

## Generating the paper figures

After all processing steps have been completed, open the Jupyter notebook:

```bash
jupyter lab notebooks/final_figures.ipynb
```

The notebook loads the final output files and generates every figure in
the paper.  Update the `BASE` path variable at the top of the notebook
to point to the directory containing the pipeline output.

| Figure | Description |
|--------|-------------|
| 1 | 3-panel seasonality (total transport, footprint, efficiency) |
| 2 | Area-percent and track-density seasonality |
| 3 | Time and zonal-mean transient-eddy transport |
| 4 | Weak vs strong cyclones energy budget bar chart |
| 5 | Ocean vs land (NH, C+A and 6+ CVU) bar chart |
| 6 | 2-D global maps of TE for weak and strong cyclones |
| 7 | Cyclone-centred SH composites (SHF & TE, DJF−JJA) |

---

## Key design decisions

* **Spherical geometry** — All meridional integrations and divergences
  use spherical coordinates (cos-latitude weighting, `d/d(lat_rad)`
  scaled by `a·cos(lat)`).  Zonal advection divergence uses periodic
  longitude boundaries.
* **Below-ground masking** — Vertical integration uses a pressure-based
  beta mask that smoothly transitions to zero for levels below the
  surface.  Level 36 on the ERA5 37-level grid is always recalculated to
  match the reference implementation.
* **Vectorised integration** — Poleward integration operates on full
  `(time, lat, lon)` arrays in a single pass (no Python loops over
  timesteps or longitudes).
* **Two-stage smoothing** — Raw ERA5 fields are regridded with CDO
  (`cdo remapbil`); derived fields (TE, vint, dh/dt) are spectrally
  smoothed with a Hoskins filter (n₀ = 60, r = 1, truncation T100)
  using NCL's built-in spherical-harmonic routines (`shaeC` / `shseC`).
  Python drives the NCL script (`ncl/hoskins_filter.ncl`) by passing
  all parameters via environment variables.
* **Complete energy budget** — The pipeline computes every term of the
  cyclone-centred MSE energy budget: transient eddy (TE), surface heat
  flux (SHF), radiation (Swabs, OLR), MSE storage (dh/dt), and zonal
  MSE advection.
* **SHF as a residual** — The surface heat flux (SHF) in W m⁻² is
  calculated as a residual of the MSE budget:
  `SHF = (vigd + vimdf·Lv + vithed) − (TSR − SSR) − TTR + dh/dt`.
  This is computed in the W/m² composites (Step 7b) and does not require
  separate ERA5 surface flux downloads.

---

## Cyclone tracking

Cyclone and anticyclone tracks are generated with the **TRACK** algorithm
(Hoskins, B., & Hodges, K. I. (2019). The annual cycle of Northern Hemisphere
storm tracks. Part I: Seasons. *J. Climate*, **32**, 1743–1760.
[doi:10.1175/JCLI-D-17-0870](https://doi.org/10.1175/JCLI-D-17-0870)).

TRACK also produces the filtered T42 vorticity fields used in the masking step.

---

## Code standards

| Rule | Convention |
|------|-----------|
| **Imports** | PEP 8 order (stdlib → third-party → local).  Always `import x`, never `from x import y`. |
| **Type hints** | Throughout; validated with `mypy`. |
| **Logging** | `_LOG.info('message: %s', value)` format — no f-strings. |
| **Functions** | Keyword-only arguments after the first positional argument. |
| **CLI** | Built with `typer`; file/directory arguments use `pathlib.Path`. |
| **Formatting** | `ruff check --fix && ruff format`. |

---

## Citation

If you use this code, please cite:

```bibtex
@article{sarro_barpanda_shaw_2025,
  title   = {What Controls the Seasonality of Intense Cyclones?},
  author  = {Sarro, Giorgio M. and Barpanda, Pragallva and Shaw, Tiffany A.},
  journal = {Geophysical Research Letters},
  year    = {2025},
  note    = {In review}
}
```

---

## License

MIT
