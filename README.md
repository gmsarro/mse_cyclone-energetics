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
cyclone_energetics/
  cli.py                  Typer CLI entry point
  constants.py            Physical constants and ERA5 calendar arrays
  geometry.py             Lon/lat ↔ polar stereographic, spherical distance
  gridded_data.py         Dimension-safe NetCDF reader and coordinate resolver
  computation/
    flux.py               Transient-eddy MSE flux divergence (v'h')
    storage.py            MSE storage term dh/dt
    advection.py          Zonal + meridional MSE advection
  smoothing/
    hoskins.py            Hoskins spectral filter (drives NCL)
  tracks/
    processing.py         TRACK algorithm output → .npz arrays
  masking/
    masks.py              Cyclone / anticyclone area masks from vorticity contours
  integration/
    poleward.py           Poleward integration of energy flux fields
  assignment/
    flux_assignment.py    Assign integrated fluxes to cyclone categories
  composites/
    builder.py            Cyclone-centred composites (PW and W/m² fields)
    condensed.py          Condensed monthly composite file
  variability/
    interannual.py        Interannual variability for confidence bands
ncl/
  hoskins_filter.ncl      NCL Hoskins spectral filter script
notebooks/
  final_figures.ipynb     Generates all publication figures
pyproject.toml            Package metadata and dependencies
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

## Pipeline

The full pipeline is exposed through the `cyclone-energetics` CLI.
Each step reads from and writes to user-specified directories.

| Phase       | Step | Command               | Module                           |
|-------------|------|-----------------------|----------------------------------|
| Computation | 1a   | `compute-te`          | `computation.flux`               |
| Computation | 1b   | `compute-dhdt`        | `computation.storage`            |
| Computation | 1c   | `compute-zonal-mse`   | `computation.advection`          |
| Smoothing   | 2    | `smooth-hoskins`      | `smoothing.hoskins`              |
| Tracks      | 3    | `process-tracks`      | `tracks.processing`              |
| Masking     | 4    | `create-masks`        | `masking.masks`                  |
| Integration | 5    | `integrate-fluxes`    | `integration.poleward`           |
| Assignment  | 6    | `assign-fluxes`       | `assignment.flux_assignment`     |
| Composites  | 7    | `build-composites`    | `composites.builder`             |
| Composites  | 8    | `condense-composites` | `composites.condensed`           |
| Variability | 9    | `compute-variability` | `variability.interannual`        |

Run `cyclone-energetics --help` for full CLI documentation, or
`cyclone-energetics <command> --help` for per-step options.

### Example

```bash
cyclone-energetics compute-te \
    --data-directory /path/to/era5 \
    --output-directory /path/to/output/te
```

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
ordering to the canonical `(time, level, latitude, longitude)` form
and provides `resolve_dimension_name()` for flexible coordinate lookup
across datasets with different naming conventions.

## License

MIT
