"""Compute the vertically integrated meridional MSE flux.

Produces files ``TE_YYYY_MM.nc`` containing the vertically integrated
product of meridional wind and moist static energy:

    (1/g) ∫₀ᵖˢ  v · h · β²  dp

where h = c_p T + g Z + L_v q is the moist static energy (including
geopotential) and β is the below-ground weighting factor.

The divergence / poleward integration is performed in a subsequent
pipeline step (:mod:`cyclone_energetics.integration`).
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


def _compute_beta_mask(
    *,
    pressure_levels_3d: npt.NDArray[np.floating],
    surface_pressure_3d: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the below-ground weighting factor (beta).

    For each pressure level j the neighbours are defined as:
      p_j_plus_1  = pa3d[:, j-1, :, :]   (the level above, lower pressure)
      p_j_minus_1 = pa3d[:, j+1, :, :]   (the level below, higher pressure)

    beta = (ps - p_j_plus_1) / (p_j_minus_1 - p_j_plus_1)
    Points fully above the surface (p_j_minus_1 < ps) get beta = 1.
    Points fully below the surface (p_j_plus_1 > ps)  get beta = 0.

    Level index 36 (typically ~925 hPa on the ERA5 37-level grid) is
    always recomputed to ensure the transition level is handled correctly,
    matching the original implementation exactly.
    """
    pa3d = pressure_levels_3d
    ps3d = surface_pressure_3d

    p_j_minus_1 = np.copy(pa3d)
    p_j_plus_1 = np.copy(pa3d)
    p_j_plus_1[:, 1:, :, :] = pa3d[:, :-1, :, :]
    p_j_minus_1[:, 1:, :, :] = pa3d[:, 1:, :, :]

    idx_below = p_j_plus_1 > ps3d
    idx_above = p_j_minus_1 < ps3d

    beta = (ps3d - p_j_plus_1) / (p_j_minus_1 - p_j_plus_1)
    beta[idx_above] = 1.0
    beta[:, 36, :, :] = (
        (ps3d[:, 36, :, :] - p_j_plus_1[:, 36, :, :])
        / (p_j_minus_1[:, 36, :, :] - p_j_plus_1[:, 36, :, :])
    )
    beta[idx_below] = 0.0

    return beta


def _compute_mse(
    *,
    temperature: npt.NDArray[np.floating],
    specific_humidity: npt.NDArray[np.floating],
    geopotential_height: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Moist static energy: c_p*T + g*Z + L_v*q."""
    return (
        constants.CPD * temperature
        + constants.GRAVITY * geopotential_height
        + constants.LATENT_HEAT_VAPORIZATION * specific_humidity
    )


def _compute_divergence(
    *,
    field: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Meridional divergence on the sphere: (1/(a*cos(lat))) d/dlat [F*cos(lat)].

    Fully vectorised over time and longitude (no Python loops).
    """
    lat_rad = np.deg2rad(latitude)
    cos_lat = np.cos(lat_rad)

    cos_2d = cos_lat[np.newaxis, :, np.newaxis]
    field_cos = field * cos_2d

    divergence = np.gradient(field_cos, lat_rad, axis=1) / (
        constants.EARTH_RADIUS * cos_2d
    )
    return divergence


def compute_transient_eddy_flux(
    *,
    year_start: int,
    year_end: int,
    era5_base_directory: pathlib.Path,
    output_directory: pathlib.Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Processing TE flux: year=%s month=%s", year, month)
            _process_single_month_te(
                year=year,
                month=month,
                era5_base_directory=era5_base_directory,
                output_directory=output_directory,
            )


def _process_single_month_te(
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
    v_path = era5_base_directory / "v" / ("era5_v_%d_%s.6hrly.nc" % (year, month))

    with netCDF4.Dataset(str(t_path)) as ds_t:
        n_time = len(ds_t["time"][:])
        latitude_now = np.array(ds_t["latitude"][:])
        longitude_now = np.array(ds_t["longitude"][:])

    n_lat = len(latitude_now)
    n_lon = len(longitude_now)
    dvmsedt = np.zeros((n_time, n_lat, n_lon))
    chunk = _LATITUDE_CHUNK_SIZE

    with netCDF4.Dataset(str(q_path)) as ds_q:
        plev = np.array(ds_q["level"][:]) * 100.0

    for lat_block in range(n_lat // chunk):
        lat_start = lat_block * chunk
        lat_end = (lat_block + 1) * chunk
        _LOG.info("Latitude block: %s to %s", lat_start, lat_end)

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

        beta = _compute_beta_mask(
            pressure_levels_3d=pa3d, surface_pressure_3d=ps3d
        )
        del ps3d

        mse = _compute_mse(
            temperature=ta,
            specific_humidity=hus,
            geopotential_height=zg,
        )
        del ta, hus, zg

        with netCDF4.Dataset(str(v_path)) as ds_v:
            v = np.array(ds_v["v"][s].filled(fill_value=np.nan), dtype=np.float64)

        # Full v * MSE product (matching original make_TE_ERA5.py).
        # NaN from .filled() below ground would poison trapz (0 * NaN = NaN
        # in plain numpy); nan_to_num ensures beta=0 zeros dominate.
        te_flux = np.nan_to_num(v * mse * beta * beta, nan=0.0)
        del v, mse, beta

        sign = 1.0 if plev[1] - plev[0] > 0 else -1.0
        dvmsedt[:, lat_start:lat_end, :] = (
            sign / constants.GRAVITY * _trapz(te_flux, pa3d, axis=1)
        )
        del te_flux, pa3d
        _LOG.info("Vertical integration completed for block %s", lat_block)

    # Save the pre-divergence vertically integrated v*MSE flux.
    # The divergence / poleward integration is done in integration.py.
    out_path = output_directory / ("TE_%d_%s.nc" % (year, month))
    with netCDF4.Dataset(str(z_path)) as ds_z:
        with netCDF4.Dataset(str(out_path), "w", format="NETCDF4_CLASSIC") as ds_out:
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

            te_var = ds_out.createVariable(
                "TE", "f4", ("time", "latitude", "longitude")
            )
            te_var.units = "W"
            te_var.long_name = (
                "vertically integrated meridional MSE flux (v * MSE)"
            )
            te_var[:, :, :] = dvmsedt
    _LOG.info("Saved TE file: %s", out_path)
