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

import gc
import logging
import pathlib
import typing

import numpy as np
import xarray

import cyclone_energetics.constants as constants
import cyclone_energetics.gridded_data as gridded_data

_LOG = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE: int = 36
_GRADIENT_CLIP_ZONAL: float = 0.5
_GRADIENT_CLIP_MERIDIONAL_FACTOR: float = 0.5


def compute_zonal_advection(
    *,
    year_start: int,
    year_end: int,
    data_directory: pathlib.Path,
    output_directory: pathlib.Path,
    filename_pattern: str = gridded_data.DEFAULT_FILENAME_PATTERN,
    variable_names: typing.Optional[typing.Dict[str, str]] = None,
    subdirectories: typing.Optional[typing.Dict[str, str]] = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    vnames = variable_names or gridded_data.DEFAULT_VARIABLE_NAMES
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Computing MSE advection: year=%s month=%s", year, month)
            _process_single_month_advection(
                year=year,
                month=month,
                data_directory=data_directory,
                output_directory=output_directory,
                filename_pattern=filename_pattern,
                variable_names=vnames,
                subdirectories=subdirectories,
            )


def _resolve(
    *,
    data_directory: pathlib.Path,
    field: str,
    year: int,
    month: str,
    filename_pattern: str,
    subdirectories: typing.Optional[typing.Dict[str, str]],
) -> pathlib.Path:
    return gridded_data.resolve_path(
        data_directory=data_directory,
        field=field,
        year=year,
        month=month,
        filename_pattern=filename_pattern,
        subdirectories=subdirectories,
    )


def _process_single_month_advection(
    *,
    year: int,
    month: str,
    data_directory: pathlib.Path,
    output_directory: pathlib.Path,
    filename_pattern: str,
    variable_names: typing.Dict[str, str],
    subdirectories: typing.Optional[typing.Dict[str, str]],
) -> None:
    kw = dict(
        data_directory=data_directory, year=year, month=month,
        filename_pattern=filename_pattern, subdirectories=subdirectories,
    )
    t_path = _resolve(field="temperature", **kw)
    q_path = _resolve(field="specific_humidity", **kw)
    ps_path = _resolve(field="surface_pressure", **kw)
    z_path = _resolve(field="geopotential", **kw)
    u_path = _resolve(field="zonal_wind", **kw)
    v_path = _resolve(field="meridional_wind", **kw)

    vn = variable_names
    latitude, longitude = gridded_data.read_coordinates(t_path)
    n_time = gridded_data.read_n_time(t_path)
    plev_pa = gridded_data.read_pressure_levels(q_path)
    pressure_levels = xarray.DataArray(
        plev_pa, dims=["level"], coords={"level": plev_pa},
    )

    n_lat = len(latitude)
    n_lon = len(longitude)
    chunk = min(_DEFAULT_CHUNK_SIZE, n_lat)
    n_lat_blocks = (n_lat + chunk - 1) // chunk
    n_lon_blocks = (n_lon + chunk - 1) // chunk
    a = constants.EARTH_RADIUS
    g = constants.GRAVITY

    # ------------------------------------------------------------------
    # Term 2 (u_mse): zonal advection — chunked along LATITUDE
    # u * beta * d(MSE*beta)/dlon / (a cos lat)
    # ------------------------------------------------------------------
    term_two_final = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    lon_mod = np.zeros(n_lon + 2, dtype=np.float64)
    lon_mod[1:-1] = longitude
    lon_mod[0] = longitude[0] - (longitude[1] - longitude[0])
    lon_mod[-1] = longitude[-1] + (longitude[1] - longitude[0])
    lon_rad_mod = np.deg2rad(lon_mod)

    for lat_block in range(n_lat_blocks):
        lat_s = lat_block * chunk
        lat_e = min((lat_block + 1) * chunk, n_lat)
        _LOG.info("  u_mse lat block: %s to %s", lat_s, lat_e)

        n_chunk = lat_e - lat_s
        lat_sl = slice(lat_s, lat_e)

        ta = gridded_data.open_field(
            t_path, vn["temperature"], latitude_slice=lat_sl,
        ).assign_coords(level=plev_pa)
        hus = gridded_data.open_field(
            q_path, vn["specific_humidity"], latitude_slice=lat_sl,
        ).assign_coords(level=plev_pa)
        ps = gridded_data.open_field(
            ps_path, vn["surface_pressure"], latitude_slice=lat_sl,
        )
        zg = (
            gridded_data.open_field(
                z_path, vn["geopotential"], latitude_slice=lat_sl,
            ) / g
        ).assign_coords(level=plev_pa)

        ps_mean = ps.mean(dim="time")
        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels, surface_pressure=ps_mean,
        )

        mse = constants.CPD * ta + g * zg + constants.LATENT_HEAT_VAPORIZATION * hus
        del ta, hus, zg

        u_wind = gridded_data.open_field(
            u_path, vn["zonal_wind"], latitude_slice=lat_sl,
        ).assign_coords(level=plev_pa)

        lat_rad_chunk = np.deg2rad(latitude[lat_s:lat_e])
        cos_lat = np.cos(lat_rad_chunk)

        beta_for_div = beta.where(beta != 0, np.nan)
        field_np = (mse * beta_for_div).values
        n_plev = plev_pa.size

        padded = np.empty(
            (n_time, n_plev, n_chunk, n_lon + 2), dtype=np.float64,
        )
        padded[:, :, :, 1:-1] = field_np
        padded[:, :, :, 0] = field_np[:, :, :, -1]
        padded[:, :, :, -1] = field_np[:, :, :, 0]

        cos_lat_4d = cos_lat[np.newaxis, np.newaxis, :, np.newaxis]
        cos_lat_padded = np.broadcast_to(
            cos_lat[np.newaxis, np.newaxis, :, np.newaxis],
            (n_time, n_plev, n_chunk, n_lon + 2),
        )

        lon_axis = 3
        grad_lon = np.gradient(padded, lon_rad_mod, axis=lon_axis)
        mse_div = (grad_lon / (a * cos_lat_padded))[:, :, :, 1:-1]

        mse_div_clipped = np.where(
            np.abs(mse_div) > _GRADIENT_CLIP_ZONAL, 0.0, mse_div,
        )

        beta_np = beta.values
        beta_4d = beta_np[np.newaxis, :, :, :] if beta_np.ndim == 3 else beta_np
        u_np = u_wind.values
        term_two_div = np.nan_to_num(
            beta_4d * u_np * mse_div_clipped, nan=0.0,
        )

        pa3d = np.broadcast_to(
            plev_pa[np.newaxis, :, np.newaxis, np.newaxis],
            (n_time, n_plev, n_chunk, n_lon),
        )
        sign = 1.0 if plev_pa[1] - plev_pa[0] > 0 else -1.0
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        term_two_final[:, lat_s:lat_e, :] = (
            sign / g * _trapz(term_two_div, pa3d, axis=1)
        )
        del term_two_div, mse, beta, beta_for_div, u_wind, ps
        gc.collect()
        _LOG.info("  u_mse block %s complete", lat_block)

    # ------------------------------------------------------------------
    # Term 1 (v_mse): meridional advection — chunked along LONGITUDE
    # v * beta * d(MSE*beta)/dlat / a
    # ------------------------------------------------------------------
    term_one_final = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)
    lat_rad_full = np.deg2rad(latitude)

    for lon_block in range(n_lon_blocks):
        lon_s = lon_block * chunk
        lon_e = min((lon_block + 1) * chunk, n_lon)
        _LOG.info("  v_mse lon block: %s to %s", lon_s, lon_e)

        n_chunk = lon_e - lon_s
        lon_sl = slice(lon_s, lon_e)

        ta = gridded_data.open_field(
            t_path, vn["temperature"], longitude_slice=lon_sl,
        ).assign_coords(level=plev_pa)
        hus = gridded_data.open_field(
            q_path, vn["specific_humidity"], longitude_slice=lon_sl,
        ).assign_coords(level=plev_pa)
        ps = gridded_data.open_field(
            ps_path, vn["surface_pressure"], longitude_slice=lon_sl,
        )
        zg = (
            gridded_data.open_field(
                z_path, vn["geopotential"], longitude_slice=lon_sl,
            ) / g
        ).assign_coords(level=plev_pa)

        ps_mean = ps.mean(dim="time")
        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels, surface_pressure=ps_mean,
        )

        mse = constants.CPD * ta + g * zg + constants.LATENT_HEAT_VAPORIZATION * hus
        del ta, hus, zg

        v_wind = gridded_data.open_field(
            v_path, vn["meridional_wind"], longitude_slice=lon_sl,
        ).assign_coords(level=plev_pa)

        beta_for_div = beta.where(beta != 0, np.nan)
        field_np = (mse * beta_for_div).values
        n_plev = plev_pa.size

        lat_axis = field_np.ndim - 2
        grad_lat = np.gradient(field_np, lat_rad_full, axis=lat_axis)

        clip_merid = _GRADIENT_CLIP_MERIDIONAL_FACTOR * a
        dmse_dl = np.where(
            np.abs(grad_lat) > clip_merid, 0.0, grad_lat,
        )

        beta_np = beta.values
        beta_4d = beta_np[np.newaxis, :, :, :] if beta_np.ndim == 3 else beta_np
        v_np = v_wind.values
        term_one_div = np.nan_to_num(
            beta_4d * v_np * dmse_dl / a, nan=0.0,
        )

        pa3d = np.broadcast_to(
            plev_pa[np.newaxis, :, np.newaxis, np.newaxis],
            (n_time, n_plev, n_lat, n_chunk),
        )
        sign = 1.0 if plev_pa[1] - plev_pa[0] > 0 else -1.0
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        term_one_final[:, :, lon_s:lon_e] = (
            sign / g * _trapz(term_one_div, pa3d, axis=1)
        )
        del term_one_div, mse, beta, beta_for_div, v_wind, ps
        gc.collect()
        _LOG.info("  v_mse block %s complete", lon_block)

    # ------------------------------------------------------------------
    # Save output: Adv_YYYY_MM.nc  with variables v_mse and u_mse
    # ------------------------------------------------------------------
    out_path = output_directory / ("Adv_%d_%s.nc" % (year, month))
    ds_out = xarray.Dataset(
        {
            "v_mse": (
                ("time", "latitude", "longitude"),
                term_one_final.astype(np.float32),
                {"units": "J", "long_name": "V*MSE"},
            ),
            "u_mse": (
                ("time", "latitude", "longitude"),
                term_two_final.astype(np.float32),
                {"units": "J", "long_name": "U*MSE"},
            ),
        },
        coords={"latitude": latitude, "longitude": longitude},
    )
    ds_out.to_netcdf(str(out_path))
    _LOG.info("Saved advection file: %s", out_path)
