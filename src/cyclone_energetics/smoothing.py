"""Smooth / filter ERA5 fields using CDO and a Hoskins spectral filter.

Two smoothing strategies are provided:

1. **CDO bilinear regridding** (``smooth_era5_fields``)
   Uses ``cdo remapbil`` for fast regridding of raw ERA5 fields to a
   coarser grid (default ``r360x180``).

2. **Hoskins spectral filter** (``hoskins_spectral_smooth``)
   Reproduces the NCL-based Hoskins filter used in the original pipeline
   (``smooth_vint.ncl``, ``smooth_dh_dt_ERA5.ncl``, etc.).  The field is
   decomposed into spherical harmonics, multiplied by the Hoskins damping
   coefficient

       w(n) = exp[ −(n(n+1) / n₀(n₀+1))^r ]

   with n₀ = 60 and r = 1, and then synthesised back to grid space.
   This requires the ``windspharm``/``spharm`` or ``pyshtools`` library.
   A pure-NumPy fallback using CDO ``sp2gp``/``gp2sp`` is **not** available
   because CDO cannot apply arbitrary per-wavenumber damping.

CDO performs bilinear interpolation on the native grid with high
throughput and minimal memory overhead, making it the most efficient
choice for regridding large 6-hourly ERA5 datasets.  The Hoskins
spectral filter is applied to derived fields (TE, vint, dh/dt) that
must be smoothed in spectral space before meridional integration.
"""

import logging
import pathlib
import subprocess

import netCDF4
import numpy as np
import numpy.typing as npt

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)


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
# 2. Hoskins spectral filter (replaces NCL smooth_vint.ncl etc.)
# ---------------------------------------------------------------------------
def _hoskins_coefficients(
    *,
    ntrunc: int,
    n0: int = constants.HOSKINS_N0,
    r: int = constants.HOSKINS_R,
) -> npt.NDArray[np.float64]:
    """Hoskins damping coefficients for total wavenumbers 0 … ntrunc."""
    n = np.arange(ntrunc + 1, dtype=np.float64)
    return np.exp(-((n * (n + 1)) / (n0 * (n0 + 1))) ** r)


def _apply_hoskins_filter_spharm(
    field_2d: npt.NDArray[np.float64],
    *,
    ntrunc: int,
    filter_coeff: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply the Hoskins spectral filter to a single 2-D (lat, lon) field.

    Uses the ``spharm`` package (SPHEREPACK Python wrapper) for spherical
    harmonic analysis and synthesis.

    The input field must be on an equally-spaced latitude–longitude grid
    with latitude **ascending** (south → north).
    """
    import spharm  # type: ignore[import-untyped]

    nlat, nlon = field_2d.shape
    s = spharm.Spharmt(nlon, nlat, gridtype="regular", rsphere=constants.EARTH_RADIUS)

    spec = s.grdtospec(field_2d, ntrunc=ntrunc)

    # ``spec`` is a 1-D complex array of length (ntrunc+1)*(ntrunc+2)/2
    # ordered by total wavenumber n then order m:
    #   index = n*(n+1)/2 + m   for  n=0..ntrunc, m=0..n
    idx = 0
    for n in range(ntrunc + 1):
        n_orders = n + 1
        spec[idx : idx + n_orders] *= filter_coeff[n]
        idx += n_orders

    return s.spectogrd(spec)


def _apply_hoskins_filter_pyshtools(
    field_2d: npt.NDArray[np.float64],
    *,
    ntrunc: int,
    filter_coeff: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fallback using ``pyshtools`` if ``spharm`` is not available."""
    import pyshtools  # type: ignore[import-untyped]

    grid = pyshtools.SHGrid.from_array(field_2d, grid="DH2")
    coeffs = grid.expand(lmax=ntrunc)

    for n in range(min(ntrunc + 1, coeffs.lmax + 1)):
        coeffs.coeffs[:, n, : n + 1] *= filter_coeff[n]

    return coeffs.expand(grid="DH2").data


def _get_filter_backend() -> str:
    """Detect which spectral-harmonic library is available."""
    try:
        import spharm  # noqa: F401
        return "spharm"
    except ImportError:
        pass
    try:
        import pyshtools  # noqa: F401
        return "pyshtools"
    except ImportError:
        pass
    raise ImportError(
        "Neither 'spharm' nor 'pyshtools' is installed. "
        "Install one of them to use the Hoskins spectral filter. "
        "pip install pyspharm   OR   pip install pyshtools"
    )


def hoskins_spectral_smooth(
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    variable_names: list,
    ntrunc: int = constants.HOSKINS_NTRUNC,
    n0: int = constants.HOSKINS_N0,
    r: int = constants.HOSKINS_R,
) -> None:
    """Apply the Hoskins spectral filter to selected variables in a NetCDF file.

    The file is read, filtered, and written to *output_path*.
    Latitude must be monotonic; if descending it is flipped internally
    (matching the NCL ``shaeC`` requirement).

    Parameters
    ----------
    variable_names : list[str]
        Variables to filter (e.g. ``["vigd", "vimdf", "vithed"]`` or ``["tend"]``
        or ``["TE"]``).
    ntrunc : int
        Maximum total wavenumber to retain (default 100, matching NCL scripts).
    """
    backend = _get_filter_backend()
    filter_coeff = _hoskins_coefficients(ntrunc=ntrunc, n0=n0, r=r)

    _LOG.info(
        "Hoskins filter: input=%s  variables=%s  backend=%s  ntrunc=%s  n0=%s",
        input_path,
        variable_names,
        backend,
        ntrunc,
        n0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(str(input_path)) as ds_in:
        # Copy structure
        with netCDF4.Dataset(
            str(output_path), "w", format="NETCDF4_CLASSIC"
        ) as ds_out:
            # Copy dimensions
            for dim_name, dim in ds_in.dimensions.items():
                ds_out.createDimension(
                    dim_name,
                    len(dim) if not dim.isunlimited() else None,
                )

            # Copy global attributes
            ds_out.setncatts(ds_in.__dict__)

            # Copy all variables
            for var_name, var_in in ds_in.variables.items():
                var_out = ds_out.createVariable(
                    var_name, var_in.datatype, var_in.dimensions
                )
                var_out.setncatts(var_in.__dict__)

                if var_name in variable_names:
                    # Apply Hoskins filter
                    data = np.array(var_in[:], dtype=np.float64)

                    # Detect latitude axis and ensure ascending
                    lat_dim_name = None
                    for dname in var_in.dimensions:
                        if "lat" in dname.lower():
                            lat_dim_name = dname
                            break

                    flipped = False
                    if lat_dim_name is not None:
                        lat_vals = ds_in[lat_dim_name][:]
                        if lat_vals[0] > lat_vals[-1]:
                            # Latitude descending — flip to ascending for SHT
                            lat_axis = list(var_in.dimensions).index(lat_dim_name)
                            data = np.flip(data, axis=lat_axis)
                            flipped = True

                    if data.ndim == 2:
                        if backend == "spharm":
                            filtered = _apply_hoskins_filter_spharm(
                                data, ntrunc=ntrunc, filter_coeff=filter_coeff
                            )
                        else:
                            filtered = _apply_hoskins_filter_pyshtools(
                                data, ntrunc=ntrunc, filter_coeff=filter_coeff
                            )
                    elif data.ndim == 3:
                        filtered = np.empty_like(data)
                        for t in range(data.shape[0]):
                            if backend == "spharm":
                                filtered[t] = _apply_hoskins_filter_spharm(
                                    data[t], ntrunc=ntrunc, filter_coeff=filter_coeff
                                )
                            else:
                                filtered[t] = _apply_hoskins_filter_pyshtools(
                                    data[t], ntrunc=ntrunc, filter_coeff=filter_coeff
                                )
                    else:
                        _LOG.info(
                            "Skipping Hoskins filter for %s: ndim=%s (expected 2 or 3)",
                            var_name,
                            data.ndim,
                        )
                        var_out[:] = var_in[:]
                        continue

                    if flipped:
                        filtered = np.flip(filtered, axis=lat_axis)

                    var_out[:] = filtered

                    # Add suffix to variable name for output tracking
                    _LOG.info("  Filtered variable: %s", var_name)
                else:
                    var_out[:] = var_in[:]

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
            dhdt_out = output_dhdt_directory / ("tend_%d_%s_filtered_2.nc" % (year, month))
            if dhdt_in.exists():
                hoskins_spectral_smooth(
                    input_path=dhdt_in,
                    output_path=dhdt_out,
                    variable_names=["tend"],
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
                    ntrunc=ntrunc,
                )
