# Cyclone Energetics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)](http://mypy-lang.org/)

Compute a **localized Moist Static Energy (MSE) budget** for cyclone
seasonality analysis.  Associated with the methods described in
Sarro, Barpanda & Shaw (2025).

## Overview

This package decomposes the poleward energy transport into contributions
from cyclones and anticyclones of different intensities. The pipeline
reads 6-hourly ERA5 (or compatible reanalysis / GCM) fields, computes
vertically integrated energy fluxes, assigns them to cyclone/anticyclone
areas identified by the TRACK algorithm, and produces the composites and
figures used in the associated publication.

## Repository layout

```
cyclone_energetics/     Python package (installed as cyclone-energetics)
  cli.py                Typer CLI entry point (cyclone-energetics command)
  constants.py          Physical constants and ERA5 calendar arrays
  geometry.py           Lon/lat ↔ polar stereographic, spherical distance
  gridded_data.py       Dimension-safe NetCDF reader (normalises to canonical ordering)
  flux_computation.py   Transient-eddy MSE flux divergence (v'h')
  storage_computation.py  MSE storage term dh/dt
  zonal_advection.py    Zonal + meridional MSE advection
  smoothing.py          Hoskins spectral filter (drives NCL)
  integration.py        Poleward integration of energy flux fields
  track_processing.py   TRACK algorithm output → .npz arrays
  masking.py            Cyclone / anticyclone area masks from vorticity contours
  flux_assignment.py    Assign integrated fluxes to cyclone categories by intensity
  composites.py         Cyclone-centred composites (PW and W/m² fields)
  condensed_composites.py  Condensed monthly composite file
  variability.py        Interannual variability for confidence bands
ncl/
  hoskins_filter.ncl    NCL Hoskins spectral filter script
notebooks/
  final_figures.ipynb   Generates all publication figures
pyproject.toml          Package metadata and dependencies
```

## Installation

```bash
pip install -e .
```

For development (includes mypy, pytest, ruff):

```bash
pip install -e ".[dev]"
```

### External dependency

The Hoskins spectral filter step requires
[NCL](https://www.ncl.ucar.edu/) to be available on `PATH`.

## Pipeline steps

The full pipeline is exposed through the `cyclone-energetics` CLI.
Each step reads from and writes to user-specified directories.

| Step | Command | Description |
|------|---------|-------------|
| 1a | `compute-te` | Transient-eddy MSE flux divergence |
| 1b | `compute-dhdt` | MSE storage term dh/dt |
| 1c | `compute-zonal-mse` | Zonal + meridional MSE advection |
| 2 | `smooth-hoskins` | Hoskins spectral filter on dh/dt, vint, adv-MSE |
| 3 | `process-tracks` | Raw TRACK output → processed .npz arrays |
| 4 | `create-masks` | Cyclone/anticyclone area masks |
| 5 | `integrate-fluxes` | Poleward integration of all energy terms |
| 6 | `assign-fluxes` | Assign fluxes to cyclone/anticyclone by intensity |
| 7 | `build-composites` | Cyclone-centred composites |
| 8 | `condense-composites` | Condensed monthly composite file |
| 9 | `compute-variability` | Interannual variability for confidence bands |

Run `cyclone-energetics --help` for full CLI documentation, or
`cyclone-energetics <command> --help` for per-step options.

### Example

```bash
cyclone-energetics compute-te \
    --data-directory /path/to/era5 \
    --output-directory /path/to/output/te
```

All directory arguments use `pathlib.Path` and accept both relative and
absolute paths.

## Input data

The pipeline expects 6-hourly ERA5 fields organised by variable
subdirectory.  The default filename pattern is
`era5_{variable}_{year}_{month}.6hrly.nc`, configurable via
`--filename-pattern`.

Required variables: temperature (`t`), specific humidity (`q`),
surface pressure (`ps`), geopotential (`z`), meridional wind (`v`),
zonal wind (`u`), radiation fields (`tsr`, `ssr`, `ttr`), and
vertically integrated energy fluxes (`vigd`, `vimdf`, `vithed`).

The dimension-safe reader (`gridded_data.py`) normalises any axis
ordering to the canonical `(time, level, latitude, longitude)` form,
so input files with non-standard dimension names or orderings are
handled automatically.

## Formatting

```bash
./grind check --fix
```

## Testing

```bash
./grind devtest
```

## License

MIT
