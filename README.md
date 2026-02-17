# Cyclone Energetics

> **What Controls the Seasonality of Intense Cyclones?**
> Giorgio M. Sarro, Pragallva Barpanda, Tiffany A. Shaw
> *Manuscript submitted to Geophysical Research Letters*

Analysis pipeline for **cyclone energetics and the seasonal cycle of
poleward energy transport**.  The package takes ERA5 reanalysis data
through a multi-stage workflow — flux computation, smoothing, cyclone
tracking, masking, integration, assignment, and composite construction —
and produces the final figures presented in the paper.

---

## Repository layout

```
cyclone_energetics/
├── pyproject.toml                    # Package metadata and dependencies
├── README.md
├── LICENSE
├── notebooks/
│   └── final_figures.ipynb           # Notebook that generates every paper figure
├── slurm/                            # SLURM batch scripts (broadwl partition)
│   ├── compute_te.sbatch
│   ├── smooth.sbatch
│   ├── process_tracks.sbatch
│   ├── create_masks.sbatch
│   ├── integrate_fluxes.sbatch
│   ├── assign_fluxes.sbatch
│   └── build_composites.sbatch
├── tests/
│   └── test_validate.py              # Regression tests against reference data
└── src/cyclone_energetics/
    ├── __init__.py
    ├── cli.py                        # Typer CLI entry points
    ├── constants.py                  # Physical / path constants
    ├── geometry.py                   # Coordinate transforms, running mean
    ├── flux_computation.py           # Transient-eddy flux from ERA5
    ├── smoothing.py                  # CDO-based field smoothing
    ├── track_processing.py           # Read and process TRACK algorithm output
    ├── masking.py                    # Cyclone / anticyclone vorticity masks
    ├── integration.py                # Poleward flux integration
    ├── flux_assignment.py            # Assign fluxes to cyclone categories
    ├── composites.py                 # Cyclone-centred composites
    └── condensed_composites.py       # Condense monthly composite files
```

---

## Installation

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode together with development
dependencies (mypy, pytest, ruff).

---

## Processing workflow

The pipeline runs in the order below.  Each step has a corresponding CLI
command (`cyclone-energetics <command>`) and a SLURM submission script under
`slurm/`.

| Step | Command | Module | Description |
|------|---------|--------|-------------|
| 1 | `compute-te` | `flux_computation` | Compute transient-eddy (TE) divergence from raw ERA5 fields.  Vertical integration accounts for below-ground data via a pressure-based beta mask. |
| 2 | `smooth` | `smoothing` | Regrid / smooth ERA5 fields using CDO bilinear interpolation (`cdo remapbil`). |
| 3 | `process-tracks` | `track_processing` | Read TRACK algorithm output (NetCDF) and write processed `.npz` arrays. |
| 4 | `create-masks` | `masking` | Build cyclone and anticyclone masks from vorticity contours and track data. |
| 5 | `integrate-fluxes` | `integration` | Integrate energy fluxes (TE, SHF, Swabs, OLR, dh/dt) poleward in spherical coordinates. |
| 6 | `assign-fluxes` | `flux_assignment` | Assign integrated fluxes to cyclones/anticyclones by intensity bin. |
| 7 | `build-composites` | `composites` | Create cyclone-centred composites per hemisphere and intensity. |
| 8 | `condense-composites` | `condensed_composites` | Produce a single condensed monthly composite file. |

### Example: running a single step

```bash
cyclone-energetics compute-te \
    --era5-base-directory /project2/tas1/gmsarro/ERA5 \
    --output-directory    /project2/tas1/gmsarro/TE_output \
    --year-start 1979 \
    --year-end   2016
```

### Example: submitting to SLURM

```bash
sbatch slurm/compute_te.sbatch
```

All SLURM scripts target the `broadwl` partition under account `pi-tas1`.
Array-job scripts are provided where parallelisation across years is
beneficial.

---

## Generating the paper figures

After all processing stages have run, open the Jupyter notebook:

```bash
jupyter lab notebooks/final_figures.ipynb
```

The notebook reads the final output files and produces every figure in the
paper:

| Figure | Description | Output file(s) |
|--------|-------------|----------------|
| 1 | 3-panel seasonality (total, footprint, efficiency) | `Seasonality_3panel_PW.png / .pdf` |
| 2 | Area-percent and track-density seasonality | `AreaPercent_and_TrackDensity_Seasonality.png / .pdf` |
| 3 | Time and zonal-mean transient-eddy transport | (inline display) |
| 4 | Weak vs strong cyclones bar chart | `weak_vs_strong_PW_winter_minus_summer_textonly.png` |
| 5 | Ocean vs land (NH, C+A and 6+ CVU) bar chart | `CA_land_ocean_plus_6CVU_land_ocean_pw_FIXED.png` |
| 6 | 2-D global maps of TE for weak and strong cyclones | `2d_maps_PW.png` |
| 7 | Cyclone-centred SH composites (SHF & TE, DJF−JJA) | `Composites.png` |

---

## Cyclone tracking

Cyclone and anticyclone tracks are computed using the **TRACK** algorithm
of Hodges (1994, 1995).  The configuration follows the setup described in:

> Hoskins, B., & Hodges, K. I. (2019). The annual cycle of Northern
> Hemisphere storm tracks. Part I: Seasons. *Journal of Climate*, **32**,
> 1743–1760. [doi:10.1175/JCLI-D-17-0870](https://doi.org/10.1175/JCLI-D-17-0870)

TRACK also produces the filtered T42 vorticity fields that are used by the
area-assignment step (masking).

---

## Key design decisions

* **Spherical geometry** — All meridional integrations and divergences are
  performed in spherical coordinates (cos-latitude weighting,
  `d/d(lat_rad)` scaling by `a·cos(lat)`).
* **Below-ground masking** — The TE vertical integration uses a
  pressure-based beta mask so that levels below the surface are
  weighted to zero.
* **Vectorised integration** — Poleward integration operates on full
  `(time, lat, lon)` arrays in a single pass (no Python loops over
  time steps or longitudes).
* **CDO smoothing** — Field smoothing/regridding delegates to CDO
  (`cdo remapbil`) for efficiency.

---

## Data paths

The pipeline expects ERA5 and track data to reside under
`/project2/tas1/gmsarro/`.  Key intermediate files:

| File | Contents |
|------|----------|
| `cyclone_centered/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc` | Integrated and intensity-binned poleward fluxes |
| `track/final/cyclonic_intensity.nc` | Cyclonic intensity fields per bin |
| `track/final/anticyclonic_intensity.nc` | Anticyclonic intensity fields per bin |
| `track/final/<Month>_count.csv` / `_6CVU.csv` | Monthly cyclone count grids |
| `cyclone_centered/Intense_Composites_SH_only_noleap.nc` | 6+ CVU SH composites |
| `cyclone_centered/Composites_SH_only_noleap.nc` | 1–5 CVU SH composites |

---

## Code standards

* **Imports** follow PEP 8 order: standard library → third-party → local.
  Always `import x` (never `from x import y`).
* **Type hints** are used throughout; check with `mypy`.
* **Logging** uses `_LOG.info('message: %s', value)` format (no f-strings in
  log calls).
* **Functions** use keyword-only arguments after the first positional argument.
* **CLI** is built with `typer`; file/directory arguments use `pathlib.Path`.
* **Formatting**: run `ruff check --fix` (or `./grind check --fix` on the
  cluster).
* **Testing**: run `pytest` (or `./grind devtest` on the cluster).

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
