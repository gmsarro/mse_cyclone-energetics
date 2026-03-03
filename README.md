# Cyclone Energetics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)](http://mypy-lang.org/)

Compute a **localized Moist Static Energy (MSE) budget** for cyclone seasonality analysis. Associated with the methods described in Sarro, Barpanda & Shaw (2025).

## Installation

```bash
pip install git+https://github.com/gmsarro/mse_cyclone-energetics.git
```

Or from a local clone:

```bash
git clone https://github.com/gmsarro/mse_cyclone-energetics.git
cd mse_cyclone-energetics
pip install -e .
```

### What you get

One importable Python package and one CLI with multiple commands:

| Package | Import | CLI Commands |
|---------|--------|--------------|
| **cyclone_energetics** | `import cyclone_energetics` | `cyclone-energetics compute-te`, `compute-dhdt`, `compute-zonal-mse`, `smooth-hoskins`, `process-tracks`, `create-masks`, `integrate-fluxes`, `assign-fluxes`, `build-composites`, `condense-composites` |

## Quick Start

The pipeline processes ERA5 reanalysis and TRACK algorithm output to compute the complete MSE budget for cyclones.

### Step 1: Compute budget terms

```bash
# Transient-eddy flux divergence
cyclone-energetics compute-te \
    --era5-base-directory /path/to/ERA5 \
    --output-directory /path/to/output/TE \
    --year-start 2000 --year-end 2020

# MSE storage term (dh/dt)
cyclone-energetics compute-dhdt \
    --era5-base-directory /path/to/ERA5 \
    --output-directory /path/to/output/dhdt \
    --year-start 2000 --year-end 2020

# Zonal MSE advection
cyclone-energetics compute-zonal-mse \
    --era5-base-directory /path/to/ERA5 \
    --output-directory /path/to/output/adv_mse \
    --year-start 2000 --year-end 2020
```

### Step 2: Apply Hoskins spectral filter

```bash
cyclone-energetics smooth-hoskins \
    --dhdt-directory /path/to/output/dhdt \
    --vint-directory /path/to/ERA5/vint \
    --adv-mse-directory /path/to/output/adv_mse \
    --output-dhdt-directory /path/to/output/smoothed_dhdt \
    --output-vint-directory /path/to/output/smoothed_vint \
    --output-adv-mse-directory /path/to/output/smoothed_adv_mse \
    --year-start 2000 --year-end 2020
```

### Step 3: Process TRACK output and create masks

```bash
cyclone-energetics process-tracks \
    --track-directory /path/to/TRACK \
    --output-directory /path/to/output/tracks

cyclone-energetics create-masks \
    --track-directory /path/to/output/tracks \
    --output-directory /path/to/output/masks \
    --year-start 2000 --year-end 2020
```

### Step 4: Integrate fluxes poleward

```bash
cyclone-energetics integrate-fluxes \
    --te-directory /path/to/output/TE \
    --dhdt-directory /path/to/output/smoothed_dhdt \
    --vint-directory /path/to/output/smoothed_vint \
    --radiation-directory /path/to/ERA5/rad \
    --adv-mse-directory /path/to/output/smoothed_adv_mse \
    --output-directory /path/to/output/integrated \
    --year-start 2000 --year-end 2020
```

### Step 5: Assign fluxes by intensity and build composites

```bash
cyclone-energetics assign-fluxes \
    --integrated-flux-directory /path/to/output/integrated \
    --mask-directory /path/to/output/masks \
    --track-directory /path/to/output/tracks \
    --output-directory /path/to/output/assigned

cyclone-energetics build-composites \
    --track-path /path/to/output/tracks/TRACK_NH.nc \
    --integrated-flux-directory /path/to/output/assigned \
    --output-directory /path/to/output/composites \
    --hemisphere NH --year-start 2000 --year-end 2020
```

Full help for any command:

```bash
cyclone-energetics --help
cyclone-energetics compute-te --help
```

## Python API

The package exposes functions for direct use in scripts and notebooks:

```python
import cyclone_energetics.flux_computation
import cyclone_energetics.integration
import cyclone_energetics.composites

# Compute transient-eddy flux for a specific year range
cyclone_energetics.flux_computation.compute_transient_eddy_flux(
    year_start=2015,
    year_end=2016,
    era5_base_directory=pathlib.Path("/path/to/ERA5"),
    output_directory=pathlib.Path("/path/to/output"),
)

# Build cyclone-centred composites
cyclone_energetics.composites.build_cyclone_composites(
    year_start=2000,
    year_end=2020,
    hemisphere="SH",
    intensity_min=6,
    intensity_max=99,
    track_path=pathlib.Path("/path/to/tracks.nc"),
    # ... additional parameters
)
```

## Configurable Parameters

All physical and numerical parameters are exposed as CLI options:

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Year range | `--year-start`, `--year-end` | *required* | Processing period |
| Hemisphere | `--hemisphere` | *required* | `NH` or `SH` |
| Intensity range | `--intensity-min`, `--intensity-max` | 1, 99 | CVU intensity filter |
| Area cut | `--area-cut` | 0.225 | Storm-track area threshold |
| Hoskins filter n₀ | `--n0` | 60 | Spectral filter cutoff |

## Repository Structure

```
mse_cyclone-energetics/
├── pyproject.toml           # Package configuration (pip install .)
├── cyclone_energetics/      # Main package
│   ├── __init__.py
│   ├── cli.py               #   Typer CLI entry point
│   ├── constants.py         #   Physical constants
│   ├── geometry.py          #   Spherical coordinate utilities
│   ├── flux_computation.py  #   Transient-eddy flux divergence
│   ├── storage_computation.py  # MSE storage term (dh/dt)
│   ├── zonal_advection.py   #   Zonal MSE advection
│   ├── smoothing.py         #   Hoskins spectral filter (NCL-driven)
│   ├── track_processing.py  #   TRACK algorithm post-processing
│   ├── masking.py           #   Cyclone/anticyclone masks
│   ├── integration.py       #   Poleward flux integration
│   ├── flux_assignment.py   #   Intensity-binned flux assignment
│   ├── composites.py        #   Cyclone-centred composites
│   └── condensed_composites.py  # Monthly condensed output
├── ncl/
│   └── hoskins_filter.ncl   # NCL Hoskins spectral filter script
├── notebooks/
│   └── final_figures.ipynb  # Reproduce all paper figures
├── README.md
├── LICENSE
└── .gitignore
```

## Requirements

### Python

- Python >= 3.10
- Dependencies are installed automatically via `pip install .`

### External

- [NCL](https://www.ncl.ucar.edu/) (for Hoskins spectral filter only)

### Data

| Dataset | Description |
|---------|-------------|
| **ERA5 6-hourly** | `t`, `q`, `ps`, `z`, `v`, `u`, `tsr`, `ssr`, `ttr`, `vint`, `t2m` |
| **TRACK output** | Cyclone/anticyclone tracks and T42 filtered vorticity |

## Generating Paper Figures

After running the full pipeline, open the example notebook:

```bash
jupyter lab notebooks/final_figures.ipynb
```

Set `OUTPUT_ROOT` at the top of the notebook to point to your output directory.

| Figure | Description |
|--------|-------------|
| 1 | 3-panel seasonality (transport, footprint, efficiency) |
| 2 | Area-percent and track-density seasonality |
| 3 | Time and zonal-mean transient-eddy transport |
| 4 | Weak vs strong cyclones energy budget |
| 5 | Ocean vs land energy budget (NH) |
| 6 | 2-D global maps of TE |
| 7 | Cyclone-centred composites (SHF & TE) |

## Citation

If you use this code, please cite:

> Sarro, G. M., P. Barpanda, and T. A. Shaw, 2025: What Controls the Seasonality of Intense Cyclones? *Geophysical Research Letters*. In review.

## License

MIT

## Contact

Giorgio M. Sarro — gmsarro@uchicago.edu
