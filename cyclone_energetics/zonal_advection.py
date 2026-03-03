from __future__ import annotations

"""Compute the vertically integrated MSE advection (zonal + meridional).

Produces files ``Adv_YYYY_MM.nc`` containing:

  u_mse:  (1/g) int  u * beta * [d(MSE*beta)/dlon / (a cos lat)]  dp
  v_mse:  (1/g) int  v * beta * [d(MSE*beta)/dlat / a          ]  dp

where MSE = c_p*T + g*Z + L_v*q is the full moist static energy,
beta is the below-ground weighting factor computed from the
**time-mean** surface pressure, and the derivatives are computed
level-by-level with gradient clipping to suppress numerical artefacts.

The zonal term uses periodic longitude boundary padding.
The meridional term uses the standard (non-periodic) latitude gradient.

Both terms are then vertically integrated and written to a single
NetCDF file per month.
"""

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

_LATITUDE_CHUNK_SIZE: int = 72
_GRADIENT_CLIP_ZONAL: float = 0.5
_GRADIENT_CLIP_MERIDIONAL_FACTOR: float = 0.5


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
            _LOG.info("Computing MSE advection: year=%s month=%s", year, month)
            _process_single_month_advection(
                year=year,
                month=month,
                era5_base_directory=era5_base_directory,
                output_directory=output_directory,
            )


def _compute_beta_mask_timemean(
    *,
    plev: npt.NDArray[np.floating],
    ps_mean: npt.NDArray[np.floating],
    n_time: int,
) -> npt.NDArray[np.floating]:
    """Beta mask using time-mean surface pressure, matching the original."""
    n_plev = plev.size
    n_lat, n_lon = ps_mean.shape

    ps3d = np.broadcast_to(
        ps_mean[np.newaxis, np.newaxis, :, :],
        (n_time, n_plev, n_lat, n_lon),
    ).copy()

    pa3d = np.broadcast_to(
        plev[np.newaxis, :, np.newaxis, np.newaxis],
        (n_time, n_plev, n_lat, n_lon),
    ).copy()

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


def _process_single_month_advection(
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
    v_path = era5_base_directory / "v" / ("era5_v_%d_%s.6hrly.nc" % (year, month))

    with netCDF4.Dataset(str(t_path)) as ds_t:
        n_time = len(ds_t["time"][:])
        latitude_now = np.array(ds_t["latitude"][:], dtype=np.float64)
        longitude_now = np.array(ds_t["longitude"][:], dtype=np.float64)

    with netCDF4.Dataset(str(q_path)) as ds_q:
        plev = np.array(ds_q["level"][:], dtype=np.float64) * 100.0

    n_lat = len(latitude_now)
    n_lon = len(longitude_now)
    n_plev = plev.size
    chunk = _LATITUDE_CHUNK_SIZE
    a = constants.EARTH_RADIUS
    g = constants.GRAVITY

    # ------------------------------------------------------------------
    # Term 2 (u_mse): zonal advection — chunked along LATITUDE
    # u * beta * d(MSE*beta)/dlon / (a cos lat)
    # ------------------------------------------------------------------
    term_two_final = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    lon_mod = np.zeros(n_lon + 2, dtype=np.float64)
    lon_mod[1:-1] = longitude_now
    lon_mod[0] = longitude_now[0] - (longitude_now[1] - longitude_now[0])
    lon_mod[-1] = longitude_now[-1] + (longitude_now[1] - longitude_now[0])
    lon_rad_mod = np.deg2rad(lon_mod)

    for lat_block in range(n_lat // chunk):
        lat_s = lat_block * chunk
        lat_e = (lat_block + 1) * chunk
        _LOG.info("  u_mse lat block: %s to %s", lat_s, lat_e)

        s = np.s_[:, :, lat_s:lat_e, :]
        n_chunk = lat_e - lat_s

        with netCDF4.Dataset(str(t_path)) as ds_t:
            ta = np.array(ds_t["t"][s], dtype=np.float64)
        with netCDF4.Dataset(str(q_path)) as ds_q:
            hus = np.array(ds_q["q"][s], dtype=np.float64)
        with netCDF4.Dataset(str(ps_path)) as ds_ps:
            ps = np.array(ds_ps["sp"][:, lat_s:lat_e, :], dtype=np.float64)
        with netCDF4.Dataset(str(z_path)) as ds_z:
            zg = np.array(ds_z["z"][s], dtype=np.float64) / g

        ps_mean = np.mean(ps, axis=0)
        beta = _compute_beta_mask_timemean(
            plev=plev, ps_mean=ps_mean, n_time=n_time,
        )

        mse = constants.CPD * ta + g * zg + constants.LATENT_HEAT_VAPORIZATION * hus
        del ta, hus, zg

        with netCDF4.Dataset(str(u_path)) as ds_u:
            u_wind = np.array(ds_u["u"][s], dtype=np.float64)

        lat_rad_chunk = np.deg2rad(latitude_now[lat_s:lat_e])
        cos_lat_2d = np.cos(lat_rad_chunk)[:, np.newaxis]

        beta_for_div = np.copy(beta)
        beta_for_div[beta == 0] = np.nan

        pa3d = np.broadcast_to(
            plev[np.newaxis, :, np.newaxis, np.newaxis],
            (n_time, n_plev, n_chunk, n_lon),
        )

        term_two_div = np.zeros((n_time, n_plev, n_chunk, n_lon), dtype=np.float64)

        for t_idx in range(n_time):
            for lev_idx in range(n_plev):
                field_padded = np.zeros((n_chunk, n_lon + 2), dtype=np.float64)
                field_padded[:, 1:-1] = (
                    mse[t_idx, lev_idx, :, :]
                    * beta_for_div[t_idx, lev_idx, :, :]
                )
                field_padded[:, 0] = (
                    mse[t_idx, lev_idx, :, -1]
                    * beta_for_div[t_idx, lev_idx, :, -1]
                )
                field_padded[:, -1] = (
                    mse[t_idx, lev_idx, :, 0]
                    * beta_for_div[t_idx, lev_idx, :, 0]
                )

                cos_lat_padded = np.broadcast_to(
                    cos_lat_2d, (n_chunk, n_lon + 2)
                )

                grad_lon = np.gradient(field_padded, lon_rad_mod, axis=1)
                mse_div = (grad_lon / (a * cos_lat_padded))[:, 1:-1]

                mse_div_clipped = np.copy(mse_div)
                mse_div_clipped[mse_div > _GRADIENT_CLIP_ZONAL] = 0.0
                mse_div_clipped[mse_div < -_GRADIENT_CLIP_ZONAL] = 0.0

                term_two_div[t_idx, lev_idx, :, :] = (
                    beta[t_idx, lev_idx, :, :]
                    * u_wind[t_idx, lev_idx, :, :]
                    * mse_div_clipped
                )

        term_two_div = np.nan_to_num(term_two_div, nan=0.0)
        sign = 1.0 if plev[1] - plev[0] > 0 else -1.0
        term_two_final[:, lat_s:lat_e, :] = (
            sign / g * _trapz(term_two_div, pa3d, axis=1)
        )
        del term_two_div, mse, beta, beta_for_div, u_wind, ps
        _LOG.info("  u_mse block %s complete", lat_block)

    # ------------------------------------------------------------------
    # Term 1 (v_mse): meridional advection — chunked along LONGITUDE
    # v * beta * d(MSE*beta)/dlat / a
    # ------------------------------------------------------------------
    term_one_final = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)
    lat_rad_full = np.deg2rad(latitude_now)

    for lon_block in range(n_lon // chunk):
        lon_s = lon_block * chunk
        lon_e = (lon_block + 1) * chunk
        _LOG.info("  v_mse lon block: %s to %s", lon_s, lon_e)

        n_chunk = lon_e - lon_s
        s = np.s_[:, :, :, lon_s:lon_e]

        with netCDF4.Dataset(str(t_path)) as ds_t:
            ta = np.array(ds_t["t"][s], dtype=np.float64)
        with netCDF4.Dataset(str(q_path)) as ds_q:
            hus = np.array(ds_q["q"][s], dtype=np.float64)
        with netCDF4.Dataset(str(ps_path)) as ds_ps:
            ps = np.array(ds_ps["sp"][:, :, lon_s:lon_e], dtype=np.float64)
        with netCDF4.Dataset(str(z_path)) as ds_z:
            zg = np.array(ds_z["z"][s], dtype=np.float64) / g

        ps_mean = np.mean(ps, axis=0)
        beta = _compute_beta_mask_timemean(
            plev=plev, ps_mean=ps_mean, n_time=n_time,
        )

        mse = constants.CPD * ta + g * zg + constants.LATENT_HEAT_VAPORIZATION * hus
        del ta, hus, zg

        with netCDF4.Dataset(str(v_path)) as ds_v:
            v_wind = np.array(ds_v["v"][s], dtype=np.float64)

        beta_for_div = np.copy(beta)
        beta_for_div[beta == 0] = np.nan

        pa3d = np.broadcast_to(
            plev[np.newaxis, :, np.newaxis, np.newaxis],
            (n_time, n_plev, n_lat, n_chunk),
        )

        term_one_div = np.zeros(
            (n_time, n_plev, n_lat, n_chunk), dtype=np.float64
        )

        clip_merid = _GRADIENT_CLIP_MERIDIONAL_FACTOR * a

        for t_idx in range(n_time):
            for lev_idx in range(n_plev):
                field = (
                    mse[t_idx, lev_idx, :, :]
                    * beta_for_div[t_idx, lev_idx, :, :]
                )

                grad_lat = np.gradient(field, lat_rad_full, axis=0)
                dmse_dl = np.copy(grad_lat)
                dmse_dl[grad_lat > clip_merid] = 0.0
                dmse_dl[grad_lat < -clip_merid] = 0.0

                term_one_div[t_idx, lev_idx, :, :] = (
                    beta[t_idx, lev_idx, :, :]
                    * v_wind[t_idx, lev_idx, :, :]
                    * dmse_dl
                ) / a

        term_one_div = np.nan_to_num(term_one_div, nan=0.0)
        sign = 1.0 if plev[1] - plev[0] > 0 else -1.0
        term_one_final[:, :, lon_s:lon_e] = (
            sign / g * _trapz(term_one_div, pa3d, axis=1)
        )
        del term_one_div, mse, beta, beta_for_div, v_wind, ps
        _LOG.info("  v_mse block %s complete", lon_block)

    # ------------------------------------------------------------------
    # Save output: Adv_YYYY_MM.nc  with variables v_mse and u_mse
    # ------------------------------------------------------------------
    out_path = output_directory / ("Adv_%d_%s.nc" % (year, month))
    with netCDF4.Dataset(str(z_path)) as ds_z:
        with netCDF4.Dataset(
            str(out_path), "w", format="NETCDF4_CLASSIC"
        ) as ds_out:
            for name, dimension in ds_z.dimensions.items():
                ds_out.createDimension(
                    name,
                    len(dimension) if not dimension.isunlimited() else None,
                )
            for name, variable in ds_z.variables.items():
                if name == "z":
                    continue
                x = ds_out.createVariable(
                    name, variable.datatype, variable.dimensions
                )
                x.setncatts(ds_z[name].__dict__)
                x[:] = ds_z[name][:]

            v_mse_var = ds_out.createVariable(
                "v_mse", "f4", ("time", "latitude", "longitude")
            )
            v_mse_var.units = "J"
            v_mse_var.long_name = "V*MSE"
            v_mse_var[:, :, :] = term_one_final

            u_mse_var = ds_out.createVariable(
                "u_mse", "f4", ("time", "latitude", "longitude")
            )
            u_mse_var.units = "J"
            u_mse_var.long_name = "U*MSE"
            u_mse_var[:, :, :] = term_two_final

    _LOG.info("Saved advection file: %s", out_path)
