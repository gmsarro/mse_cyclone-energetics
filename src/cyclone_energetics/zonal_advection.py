from __future__ import annotations

"""Compute the vertically integrated zonal MSE advection divergence.

The zonal MSE advection is the zonal divergence of the vertically
integrated product of zonal wind and moist static energy:

    1/(a cos lat) * d/dlon [ (1/g) int_0^ps  u * h * beta^2  dp ]

where h = c_p*T + g*Z + L_v*q is the full MSE (including geopotential).
The below-ground weighting *beta* and the vertical integration follow the
same conventions as :mod:`cyclone_energetics.flux_computation`.
"""

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt

import cyclone_energetics.constants as constants
import cyclone_energetics.flux_computation as flux_computation

_LOG = logging.getLogger(__name__)

# np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

_LATITUDE_CHUNK_SIZE: int = 72


def compute_zonal_advection(
    *,
    year_start: int,
    year_end: int,
    era5_base_directory: pathlib.Path,
    output_directory: pathlib.Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Computing zonal advection: year=%s month=%s", year, month)
            _process_single_month_zonal(
                year=year,
                month=month,
                era5_base_directory=era5_base_directory,
                output_directory=output_directory,
            )


def _process_single_month_zonal(
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
    u_path = era5_base_directory / "u" / ("era5_u_%d_%s.6hrly.nc" % (year, month))

    with netCDF4.Dataset(str(t_path)) as ds_t:
        n_time = len(ds_t["time"][:])
        latitude_now = np.array(ds_t["latitude"][:])
        longitude_now = np.array(ds_t["longitude"][:])

    n_lat = len(latitude_now)
    n_lon = len(longitude_now)
    chunk = _LATITUDE_CHUNK_SIZE

    with netCDF4.Dataset(str(q_path)) as ds_q:
        plev = np.array(ds_q["level"][:]) * 100.0

    dvmsedt = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    for lat_block in range(n_lat // chunk):
        lat_start = lat_block * chunk
        lat_end = (lat_block + 1) * chunk
        _LOG.info("  Latitude block: %s to %s", lat_start, lat_end)

        s = np.s_[:, :, lat_start:lat_end, :]

        with netCDF4.Dataset(str(t_path)) as ds_t:
            ta = np.array(ds_t["t"][s].filled(fill_value=np.nan), dtype=np.float64)

        with netCDF4.Dataset(str(q_path)) as ds_q:
            hus = np.array(ds_q["q"][s].filled(fill_value=np.nan), dtype=np.float64)

        with netCDF4.Dataset(str(ps_path)) as ds_ps:
            ps = np.array(ds_ps["sp"][:, lat_start:lat_end, :], dtype=np.float64)

        with netCDF4.Dataset(str(z_path)) as ds_z:
            zg = np.array(ds_z["z"][s], dtype=np.float64) / constants.GRAVITY

        n_plev = plev.size
        n_chunk = lat_end - lat_start

        ps3d = np.broadcast_to(
            ps[:, np.newaxis, :, :], (n_time, n_plev, n_chunk, n_lon)
        ).copy()
        pa3d = np.broadcast_to(
            plev[np.newaxis, :, np.newaxis, np.newaxis],
            (n_time, n_plev, n_chunk, n_lon),
        ).copy()

        beta = flux_computation._compute_beta_mask(
            pressure_levels_3d=pa3d, surface_pressure_3d=ps3d,
        )
        del ps3d

        # Full MSE: c_p*T + g*Z + L_v*q
        mse = (
            constants.CPD * ta
            + constants.GRAVITY * zg
            + constants.LATENT_HEAT_VAPORIZATION * hus
        )
        del ta, hus, zg

        # Zonal wind
        with netCDF4.Dataset(str(u_path)) as ds_u:
            u = np.array(ds_u["u"][s].filled(fill_value=np.nan), dtype=np.float64)

        # Vertically integrate u * MSE with beta^2 weighting
        # NaN from .filled() below ground would poison trapz (0 * NaN = NaN
        # in plain numpy); nan_to_num ensures beta=0 zeros dominate.
        te_flux = np.nan_to_num(u * mse * beta * beta, nan=0.0)
        del u, mse, beta

        sign = 1.0 if plev[1] - plev[0] > 0 else -1.0
        dvmsedt[:, lat_start:lat_end, :] = (
            sign / constants.GRAVITY * _trapz(te_flux, pa3d, axis=1)
        )
        del te_flux, pa3d
        _LOG.info("  Vertical integration completed for block %s", lat_block)

    # Zonal divergence: 1/(a*cos(lat)) * d/dlon(field)
    # Pad with periodic boundary conditions
    te_div = _compute_zonal_divergence(
        field=dvmsedt,
        latitude=latitude_now,
        longitude=longitude_now,
    )

    out_path = output_directory / ("TE_%d_%s.nc" % (year, month))
    with netCDF4.Dataset(str(z_path)) as ds_z:
        with netCDF4.Dataset(str(out_path), "w", format="NETCDF4_CLASSIC") as ds_out:
            for name, dimension in ds_z.dimensions.items():
                if "level" in name:
                    continue
                ds_out.createDimension(
                    name, len(dimension) if not dimension.isunlimited() else None,
                )
            for name, variable in ds_z.variables.items():
                if name in ("z", "level"):
                    continue
                x = ds_out.createVariable(name, variable.datatype, variable.dimensions)
                x.setncatts(ds_z[name].__dict__)
                x[:] = ds_z[name][:]

            te_var = ds_out.createVariable("TE", "f4", ("time", "latitude", "longitude"))
            te_var.units = "W/m2"
            te_var.long_name = "vertically integrated zonal MSE advection divergence"
            te_var[:, :, :] = te_div

    _LOG.info("Saved zonal advection file: %s", out_path)


def _compute_zonal_divergence(
    *,
    field: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    longitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Zonal divergence on the sphere with periodic longitude boundaries.

    (1 / (a * cos(lat))) * d/dlon [field]

    Fully vectorised over time (no Python loops).
    Input shape: ``(n_time, n_lat, n_lon)``
    """
    n_lon = longitude.shape[0]

    lon_mod = np.zeros(n_lon + 2)
    lon_mod[1:-1] = np.copy(longitude)
    lon_mod[0] = longitude[0] - (longitude[1] - longitude[0])
    lon_mod[-1] = longitude[-1] + (longitude[1] - longitude[0])
    lon_rad = np.deg2rad(lon_mod)

    lat_rad = np.deg2rad(latitude)
    cos_lat = np.cos(lat_rad)

    field_padded = np.empty(
        (field.shape[0], len(latitude), n_lon + 2), dtype=field.dtype
    )
    field_padded[:, :, 1:-1] = field
    field_padded[:, :, 0] = field[:, :, -1]
    field_padded[:, :, -1] = field[:, :, 0]

    grad_lon = np.gradient(field_padded, lon_rad, axis=2)
    cos_3d = cos_lat[np.newaxis, :, np.newaxis]
    # Avoid division by zero at poles (cos(90°) ≈ 0); set divergence to 0 there
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (grad_lon / (constants.EARTH_RADIUS * cos_3d))[:, :, 1:-1]
    # At the poles the zonal divergence is identically zero by symmetry
    pole_mask = np.abs(cos_lat) < 1e-10
    result[:, pole_mask, :] = 0.0
    return result
