from __future__ import annotations

"""Smooth / filter ERA5 fields using CDO and NCL.

Two smoothing strategies are provided:

1. **CDO bilinear regridding** (``smooth_era5_fields``)
   Uses ``cdo remapbil`` for fast regridding of raw ERA5 fields to a
   coarser grid (default ``r360x180``).

2. **NCL Hoskins spectral filter** (``hoskins_spectral_smooth``)
   Drives the NCL script ``ncl/hoskins_filter.ncl`` which reproduces the
   original NCL-based Hoskins filter used in the analysis pipeline
   (``smooth_vint.ncl``, ``smooth_dh_dt_ERA5.ncl``, ``smooth_VM_ERA5.ncl``).

   The field is decomposed into spherical harmonics (``shaeC``), the
   spectral coefficients are truncated and multiplied by the Hoskins
   damping coefficient

       w(n) = exp[ −(n(n+1) / n₀(n₀+1))^r ]

   with n₀ = 60 and r = 1, then synthesised back to grid space
   (``shseC``).  NCL's built-in spherical-harmonic routines are
   extremely efficient and well-tested.

CDO performs bilinear interpolation on the native grid with high
throughput and minimal memory overhead.  The Hoskins spectral filter is
applied to derived fields (TE, vint, dh/dt) that must be smoothed in
spectral space before meridional integration.
"""

import logging
import os
import pathlib
import subprocess

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

# Locate the NCL script shipped with this package
_NCL_SCRIPT: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "ncl" / "hoskins_filter.ncl"
)


# ---------------------------------------------------------------------------
# 1. CDO bilinear regridding
# ---------------------------------------------------------------------------
def smooth_era5_fields(
    *,
    year_start: int,
    year_end: int,
    variables: list,
    input_directory: pathlib.Path,
    output_directory: pathlib.Path,
    target_grid: str = "r360x180",
) -> None:
    """Regrid raw ERA5 fields with CDO ``remapbil``.

    For each ``(variable, year, month)`` combination the file
    ``era5_{var}_{year}_{month}.6hrly.nc`` is interpolated onto
    *target_grid* and written to *output_directory*.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    for variable in variables:
        for year in range(year_start, year_end):
            for month in constants.MONTH_STRINGS:
                _LOG.info(
                    "CDO regrid: variable=%s year=%s month=%s",
                    variable,
                    year,
                    month,
                )
                infile = (
                    input_directory
                    / variable
                    / ("era5_%s_%d_%s.6hrly.nc" % (variable, year, month))
                )
                outfile = (
                    output_directory
                    / variable
                    / (
                        "smoothed_era5_%s_%d_%s.6hrly.nc"
                        % (variable, year, month)
                    )
                )
                outfile.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "cdo",
                    "remapbil,%s" % target_grid,
                    str(infile),
                    str(outfile),
                ]
                _LOG.info("Running: %s", " ".join(cmd))
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )
                if result.returncode != 0:
                    _LOG.error(
                        "CDO smoothing failed for %s: %s",
                        infile,
                        result.stderr,
                    )
                    raise RuntimeError(
                        "CDO smoothing failed for %s" % infile
                    )


# ---------------------------------------------------------------------------
# 2. NCL Hoskins spectral filter
# ---------------------------------------------------------------------------
def hoskins_spectral_smooth(
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    variable_names: list,
    output_variable_names: list = None,
    ntrunc: int = constants.HOSKINS_NTRUNC,
    n0: int = constants.HOSKINS_N0,
    r: int = constants.HOSKINS_R,
    ncl_script: pathlib.Path = None,
) -> None:
    """Apply the Hoskins spectral filter by calling the NCL script.

    Parameters
    ----------
    input_path : pathlib.Path
        NetCDF file to filter.
    output_path : pathlib.Path
        Destination for the filtered NetCDF file.
    variable_names : list[str]
        Variables to filter (e.g. ``["TE"]`` or ``["vigd", "vimdf", "vithed"]``).
    output_variable_names : list[str] or None
        Output variable names.  If *None*, each input name gets a
        ``_filtered`` suffix.
    ntrunc : int
        Maximum total wavenumber (T-number) to retain.
    n0 : int
        Hoskins filter damping wavenumber.
    r : int
        Hoskins filter exponent.
    ncl_script : pathlib.Path or None
        Override path to the NCL script.  Defaults to
        ``<repo>/ncl/hoskins_filter.ncl``.
    """
    if ncl_script is None:
        ncl_script = _NCL_SCRIPT

    if not ncl_script.exists():
        raise FileNotFoundError(
            "NCL script not found: %s" % ncl_script
        )

    if output_variable_names is None:
        output_variable_names = ["%s_filtered" % v for v in variable_names]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["SMOOTH_INPUT_FILE"] = str(input_path)
    env["SMOOTH_OUTPUT_FILE"] = str(output_path)
    env["SMOOTH_VARNAMES"] = ",".join(variable_names)
    env["SMOOTH_SUFFIXED"] = ",".join(output_variable_names)
    env["SMOOTH_TLOW"] = "0"
    env["SMOOTH_THIGH"] = str(ntrunc)
    env["SMOOTH_N0"] = str(n0)
    env["SMOOTH_R"] = str(r)

    cmd = ["ncl", str(ncl_script)]
    _LOG.info(
        "Running NCL Hoskins filter: %s → %s  vars=%s  T=%s  n0=%s  r=%s",
        input_path.name,
        output_path.name,
        variable_names,
        ntrunc,
        n0,
        r,
    )

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env
    )

    if result.returncode != 0:
        _LOG.error(
            "NCL Hoskins filter failed for %s:\nSTDOUT:\n%s\nSTDERR:\n%s",
            input_path,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(
            "NCL Hoskins filter failed for %s (exit %d)"
            % (input_path, result.returncode)
        )

    _LOG.info("Saved Hoskins-filtered file: %s", output_path)


def smooth_all_pipeline_fields(
    *,
    year_start: int,
    year_end: int,
    te_directory: pathlib.Path,
    dhdt_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    output_te_directory: pathlib.Path,
    output_dhdt_directory: pathlib.Path,
    output_vint_directory: pathlib.Path,
    ntrunc: int = constants.HOSKINS_NTRUNC,
) -> None:
    """Apply the Hoskins spectral filter to all pipeline fields.

    This replaces the NCL scripts ``smooth_vint.ncl``,
    ``smooth_dh_dt_ERA5.ncl``, and ``smooth_VM_ERA5.ncl``.

    Smoothed files are written with a ``_filtered`` suffix in the output
    directories.
    """
    output_te_directory.mkdir(parents=True, exist_ok=True)
    output_dhdt_directory.mkdir(parents=True, exist_ok=True)
    output_vint_directory.mkdir(parents=True, exist_ok=True)

    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            # TE files
            te_in = te_directory / ("TE_%d_%s.nc" % (year, month))
            te_out = output_te_directory / ("TE_%d_%s_filtered.nc" % (year, month))
            if te_in.exists():
                hoskins_spectral_smooth(
                    input_path=te_in,
                    output_path=te_out,
                    variable_names=["TE"],
                    ntrunc=ntrunc,
                )

            # dh/dt files
            dhdt_in = dhdt_directory / ("tend_%d_%s_2.nc" % (year, month))
            dhdt_out = output_dhdt_directory / ("tend_%d_%s_filtered.nc" % (year, month))
            if dhdt_in.exists():
                hoskins_spectral_smooth(
                    input_path=dhdt_in,
                    output_path=dhdt_out,
                    variable_names=["tend"],
                    output_variable_names=["tend_filtered"],
                    ntrunc=ntrunc,
                )

            # Vint files (three variables)
            vint_in = vint_directory / ("era5_vint_%d_%s.6hrly.nc" % (year, month))
            vint_out = output_vint_directory / (
                "era5_vint_%d_%s_filtered.nc" % (year, month)
            )
            if vint_in.exists():
                hoskins_spectral_smooth(
                    input_path=vint_in,
                    output_path=vint_out,
                    variable_names=["vigd", "vimdf", "vithed"],
                    output_variable_names=[
                        "vigd_filtered",
                        "vimdf_filtered",
                        "vithed_filtered",
                    ],
                    ntrunc=ntrunc,
                )
