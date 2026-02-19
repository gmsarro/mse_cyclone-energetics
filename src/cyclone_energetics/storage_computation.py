from __future__ import annotations

"""Compute the vertically integrated MSE storage term (dh/dt).

The storage term is the time tendency of the vertically integrated moist
static energy (MSE).  For each 6-hourly timestep the 3-D MSE is
computed as

    h = (c_p * T  +  L_v * q) * beta

where *beta* is the below-ground weighting factor computed from the
**time-mean** surface pressure (see
:mod:`cyclone_energetics.flux_computation`).  The centred-difference
time derivative is then vertically integrated and written to a NetCDF
file.

The computation is performed in latitude chunks to keep memory usage
well below 8 GB per chunk, matching the strategy used in
:mod:`cyclone_energetics.flux_computation`.
"""

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

# np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

_LATITUDE_CHUNK_SIZE: int = 72
_DT_CENTERED: float = 43200.0  # 12 h in seconds (centred difference)
_DT_FORWARD: float = 21600.0   # 6 h in seconds (forward / backward diff)


def compute_storage_term(
    *,
    year_start: int,
    year_end: int,
    era5_base_directory: pathlib.Path,
    output_directory: pathlib.Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Computing dh/dt: year=%s month=%s", year, month)
            _process_single_month_dhdt(
                year=year,
                month=month,
                era5_base_directory=era5_base_directory,
                output_directory=output_directory,
            )


def _compute_beta_mask_3d(
    *,
    pressure_levels_3d: npt.NDArray[np.floating],
    surface_pressure_3d: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute below-ground beta mask for a single-timestep 3-D field.

    Input shape: ``(n_plev, n_lat, n_lon)``.
    """
    pa3d = pressure_levels_3d
    ps3d = surface_pressure_3d

    p_j_minus_1 = np.copy(pa3d)
    p_j_plus_1 = np.copy(pa3d)
    p_j_plus_1[1:, :, :] = pa3d[:-1, :, :]
    p_j_minus_1[1:, :, :] = pa3d[1:, :, :]

    idx_below = p_j_plus_1 > ps3d
    idx_above = p_j_minus_1 < ps3d

    beta = (ps3d - p_j_plus_1) / (p_j_minus_1 - p_j_plus_1)
    beta[idx_above] = 1.0
    beta[36, :, :] = (
        (ps3d[36, :, :] - p_j_plus_1[36, :, :])
        / (p_j_minus_1[36, :, :] - p_j_plus_1[36, :, :])
    )
    beta[idx_below] = 0.0

    return beta


def _process_single_month_dhdt(
    *,
    year: int,
    month: str,
    era5_base_directory: pathlib.Path,
    output_directory: pathlib.Path,
) -> None:
    t_path = era5_base_directory / "t" / ("era5_t_%d_%s.6hrly.nc" % (year, month))
    q_path = era5_base_directory / "q" / ("era5_q_%d_%s.6hrly.nc" % (year, month))
    ps_path = era5_base_directory / "ps" / ("era5_ps_%d_%s.6hrly.nc" % (year, month))
    z_path = era5_base_directory / "z" / ("era5_z_%d_%s.6hrly.nc" % (year, month))

    with netCDF4.Dataset(str(t_path)) as ds_t:
        n_time = len(ds_t["time"][:])
        latitude_now = np.array(ds_t["latitude"][:])
        longitude_now = np.array(ds_t["longitude"][:])

    with netCDF4.Dataset(str(q_path)) as ds_q:
        plev = np.array(ds_q["level"][:]) * 100.0

    n_lat = len(latitude_now)
    n_lon = len(longitude_now)
    n_plev = plev.size
    chunk = _LATITUDE_CHUNK_SIZE

    # The original code uses the time-mean surface pressure for the beta mask.
    with netCDF4.Dataset(str(ps_path)) as ds_ps:
        ps_mean = np.mean(
            np.array(ds_ps["sp"][:, :, :], dtype=np.float64), axis=0
        )

    dvmsedt = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    for lat_block in range(n_lat // chunk):
        lat_start = lat_block * chunk
        lat_end = (lat_block + 1) * chunk
        n_chunk = lat_end - lat_start
        _LOG.info("  Latitude block: %s to %s", lat_start, lat_end)

        # Beta mask for this latitude chunk (time-invariant)
        ps_chunk = ps_mean[lat_start:lat_end, :]
        pa3d_single = np.broadcast_to(
            plev[:, np.newaxis, np.newaxis], (n_plev, n_chunk, n_lon)
        ).copy()
        ps3d_chunk = np.broadcast_to(
            ps_chunk[np.newaxis, :, :], (n_plev, n_chunk, n_lon)
        ).copy()

        beta = _compute_beta_mask_3d(
            pressure_levels_3d=pa3d_single,
            surface_pressure_3d=ps3d_chunk,
        )
        del ps3d_chunk

        # Read T and Q for all timesteps in this chunk
        s = np.s_[:, :, lat_start:lat_end, :]

        with netCDF4.Dataset(str(t_path)) as ds_t:
            ta = np.array(
                ds_t["t"][s].filled(fill_value=np.nan), dtype=np.float64
            )

        with netCDF4.Dataset(str(q_path)) as ds_q:
            hus = np.array(
                ds_q["q"][s].filled(fill_value=np.nan), dtype=np.float64
            )

        # MSE = (c_p * T + L_v * q) * beta  (no geopotential for storage)
        beta_4d = beta[np.newaxis, :, :, :]
        mse = (
            constants.CPD * ta + constants.LATENT_HEAT_VAPORIZATION * hus
        ) * beta_4d
        del ta, hus, beta, beta_4d

        # Centred-difference time tendency
        dmsedt = np.empty_like(mse)
        dmsedt[1:-1] = (mse[2:] - mse[:-2]) / _DT_CENTERED
        dmsedt[0] = (mse[1] - mse[0]) / _DT_FORWARD
        dmsedt[-1] = (mse[-1] - mse[-2]) / _DT_FORWARD
        del mse

        # Replace NaN with zero before integration (safety measure;
        # ERA5 pressure-level data should not have missing values)
        np.nan_to_num(dmsedt, copy=False, nan=0.0)

        # Vertically integrate — fully vectorised over time and longitude
        pa3d = np.broadcast_to(
            plev[np.newaxis, :, np.newaxis, np.newaxis],
            (n_time, n_plev, n_chunk, n_lon),
        )

        sign = 1.0 if plev[1] - plev[0] > 0 else -1.0
        dvmsedt[:, lat_start:lat_end, :] = (
            sign / constants.GRAVITY * _trapz(dmsedt, pa3d, axis=1)
        )
        del dmsedt
        _LOG.info("  Block %s complete", lat_block)

    _LOG.info("Vertical integration complete for year=%s month=%s", year, month)

    # Save output
    out_path = output_directory / ("tend_%d_%s_2.nc" % (year, month))
    with netCDF4.Dataset(str(z_path)) as ds_z:
        with netCDF4.Dataset(
            str(out_path), "w", format="NETCDF4_CLASSIC"
        ) as ds_out:
            for name, dimension in ds_z.dimensions.items():
                if "level" in name:
                    continue
                ds_out.createDimension(
                    name,
                    len(dimension) if not dimension.isunlimited() else None,
                )
            for name, variable in ds_z.variables.items():
                if name in ("z", "level"):
                    continue
                x = ds_out.createVariable(
                    name, variable.datatype, variable.dimensions
                )
                x.setncatts(ds_z[name].__dict__)
                x[:] = ds_z[name][:]

            tend_var = ds_out.createVariable(
                "tend", "f4", ("time", "latitude", "longitude")
            )
            tend_var.units = "W/m^2"
            tend_var.long_name = (
                "time tendency of vertically integrated moist static energy"
            )
            tend_var[:, :, :] = dvmsedt

    _LOG.info("Saved dh/dt file: %s", out_path)
