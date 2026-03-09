from __future__ import annotations

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
    resolved_variable_names = variable_names or gridded_data.DEFAULT_VARIABLE_NAMES
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Computing MSE advection: year=%s month=%s", year, month)
            _process_single_month_advection(
                year=year,
                month=month,
                data_directory=data_directory,
                output_directory=output_directory,
                filename_pattern=filename_pattern,
                variable_names=resolved_variable_names,
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
    path_kw = dict(
        data_directory=data_directory, year=year, month=month,
        filename_pattern=filename_pattern, subdirectories=subdirectories,
    )
    temperature_path = gridded_data.resolve_path(field="temperature", **path_kw)
    humidity_path = gridded_data.resolve_path(field="specific_humidity", **path_kw)
    surface_pressure_path = gridded_data.resolve_path(field="surface_pressure", **path_kw)
    geopotential_path = gridded_data.resolve_path(field="geopotential", **path_kw)
    zonal_wind_path = gridded_data.resolve_path(field="zonal_wind", **path_kw)
    meridional_wind_path = gridded_data.resolve_path(field="meridional_wind", **path_kw)

    latitude, longitude = gridded_data.read_coordinates(temperature_path)
    n_timesteps = gridded_data.read_n_time(temperature_path)
    pressure_levels_pa = gridded_data.read_pressure_levels(humidity_path)
    pressure_levels = xarray.DataArray(
        pressure_levels_pa, dims=["level"], coords={"level": pressure_levels_pa},
    )

    n_latitude = len(latitude)
    n_longitude = len(longitude)
    chunk = min(_DEFAULT_CHUNK_SIZE, n_latitude)
    n_lat_blocks = (n_latitude + chunk - 1) // chunk
    n_lon_blocks = (n_longitude + chunk - 1) // chunk
    earth_radius = constants.EARTH_RADIUS
    gravity = constants.GRAVITY

    zonal_advection_integrated = np.zeros((n_timesteps, n_latitude, n_longitude), dtype=np.float64)

    longitude_padded = np.zeros(n_longitude + 2, dtype=np.float64)
    longitude_padded[1:-1] = longitude
    longitude_padded[0] = longitude[0] - (longitude[1] - longitude[0])
    longitude_padded[-1] = longitude[-1] + (longitude[1] - longitude[0])
    longitude_rad_padded = np.deg2rad(longitude_padded)

    for lat_block in range(n_lat_blocks):
        lat_start = lat_block * chunk
        lat_end = min((lat_block + 1) * chunk, n_latitude)
        _LOG.info("  u_mse lat block: %s to %s", lat_start, lat_end)

        n_chunk_lat = lat_end - lat_start
        lat_sl = slice(lat_start, lat_end)

        temperature = gridded_data.open_field(
            temperature_path, variable=variable_names["temperature"], latitude_slice=lat_sl,
        ).assign_coords(level=pressure_levels_pa)
        specific_humidity = gridded_data.open_field(
            humidity_path, variable=variable_names["specific_humidity"], latitude_slice=lat_sl,
        ).assign_coords(level=pressure_levels_pa)
        surface_pressure = gridded_data.open_field(
            surface_pressure_path, variable=variable_names["surface_pressure"], latitude_slice=lat_sl,
        )
        geopotential_height = (
            gridded_data.open_field(
                geopotential_path, variable=variable_names["geopotential"], latitude_slice=lat_sl,
            ) / gravity
        ).assign_coords(level=pressure_levels_pa)

        surface_pressure_mean = surface_pressure.mean(dim="time")
        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels, surface_pressure=surface_pressure_mean,
        )

        moist_static_energy = (
            constants.CPD * temperature
            + gravity * geopotential_height
            + constants.LATENT_HEAT_VAPORIZATION * specific_humidity
        )
        del temperature, specific_humidity, geopotential_height

        zonal_wind = gridded_data.open_field(
            zonal_wind_path, variable=variable_names["zonal_wind"], latitude_slice=lat_sl,
        ).assign_coords(level=pressure_levels_pa)

        latitude_rad_chunk = np.deg2rad(latitude[lat_start:lat_end])
        cos_latitude = np.cos(latitude_rad_chunk)

        beta_nonzero = beta.where(beta != 0, np.nan)
        mse_masked = (moist_static_energy * beta_nonzero).values
        n_pressure_levels = pressure_levels_pa.size

        padded = np.empty(
            (n_timesteps, n_pressure_levels, n_chunk_lat, n_longitude + 2), dtype=np.float64,
        )
        padded[:, :, :, 1:-1] = mse_masked
        padded[:, :, :, 0] = mse_masked[:, :, :, -1]
        padded[:, :, :, -1] = mse_masked[:, :, :, 0]

        cos_latitude_padded = np.broadcast_to(
            cos_latitude[np.newaxis, np.newaxis, :, np.newaxis],
            (n_timesteps, n_pressure_levels, n_chunk_lat, n_longitude + 2),
        )

        lon_axis = 3
        gradient_longitude = np.gradient(padded, longitude_rad_padded, axis=lon_axis)
        mse_zonal_gradient = (gradient_longitude / (earth_radius * cos_latitude_padded))[:, :, :, 1:-1]

        mse_zonal_gradient_clipped = np.where(
            np.abs(mse_zonal_gradient) > _GRADIENT_CLIP_ZONAL, 0.0, mse_zonal_gradient,
        )

        beta_values = beta.values
        beta_4d = beta_values[np.newaxis, :, :, :] if beta_values.ndim == 3 else beta_values
        zonal_wind_values = zonal_wind.values
        zonal_advection_divergence = np.nan_to_num(
            beta_4d * zonal_wind_values * mse_zonal_gradient_clipped, nan=0.0,
        )

        pressure_broadcast = np.broadcast_to(
            pressure_levels_pa[np.newaxis, :, np.newaxis, np.newaxis],
            (n_timesteps, n_pressure_levels, n_chunk_lat, n_longitude),
        )
        sign = 1.0 if pressure_levels_pa[1] - pressure_levels_pa[0] > 0 else -1.0
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        zonal_advection_integrated[:, lat_start:lat_end, :] = (
            sign / gravity * _trapz(zonal_advection_divergence, pressure_broadcast, axis=1)
        )
        del zonal_advection_divergence, moist_static_energy, beta, beta_nonzero, zonal_wind, surface_pressure
        gc.collect()
        _LOG.info("  u_mse block %s complete", lat_block)

    meridional_advection_integrated = np.zeros((n_timesteps, n_latitude, n_longitude), dtype=np.float64)
    latitude_rad_full = np.deg2rad(latitude)

    for lon_block in range(n_lon_blocks):
        lon_start = lon_block * chunk
        lon_end = min((lon_block + 1) * chunk, n_longitude)
        _LOG.info("  v_mse lon block: %s to %s", lon_start, lon_end)

        n_chunk_lon = lon_end - lon_start
        lon_sl = slice(lon_start, lon_end)

        temperature = gridded_data.open_field(
            temperature_path, variable=variable_names["temperature"], longitude_slice=lon_sl,
        ).assign_coords(level=pressure_levels_pa)
        specific_humidity = gridded_data.open_field(
            humidity_path, variable=variable_names["specific_humidity"], longitude_slice=lon_sl,
        ).assign_coords(level=pressure_levels_pa)
        surface_pressure = gridded_data.open_field(
            surface_pressure_path, variable=variable_names["surface_pressure"], longitude_slice=lon_sl,
        )
        geopotential_height = (
            gridded_data.open_field(
                geopotential_path, variable=variable_names["geopotential"], longitude_slice=lon_sl,
            ) / gravity
        ).assign_coords(level=pressure_levels_pa)

        surface_pressure_mean = surface_pressure.mean(dim="time")
        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels, surface_pressure=surface_pressure_mean,
        )

        moist_static_energy = (
            constants.CPD * temperature
            + gravity * geopotential_height
            + constants.LATENT_HEAT_VAPORIZATION * specific_humidity
        )
        del temperature, specific_humidity, geopotential_height

        meridional_wind = gridded_data.open_field(
            meridional_wind_path, variable=variable_names["meridional_wind"], longitude_slice=lon_sl,
        ).assign_coords(level=pressure_levels_pa)

        beta_nonzero = beta.where(beta != 0, np.nan)
        mse_masked = (moist_static_energy * beta_nonzero).values
        n_pressure_levels = pressure_levels_pa.size

        lat_axis = mse_masked.ndim - 2
        gradient_latitude = np.gradient(mse_masked, latitude_rad_full, axis=lat_axis)

        clip_meridional = _GRADIENT_CLIP_MERIDIONAL_FACTOR * earth_radius
        mse_meridional_gradient = np.where(
            np.abs(gradient_latitude) > clip_meridional, 0.0, gradient_latitude,
        )

        beta_values = beta.values
        beta_4d = beta_values[np.newaxis, :, :, :] if beta_values.ndim == 3 else beta_values
        meridional_wind_values = meridional_wind.values
        meridional_advection_divergence = np.nan_to_num(
            beta_4d * meridional_wind_values * mse_meridional_gradient / earth_radius, nan=0.0,
        )

        pressure_broadcast = np.broadcast_to(
            pressure_levels_pa[np.newaxis, :, np.newaxis, np.newaxis],
            (n_timesteps, n_pressure_levels, n_latitude, n_chunk_lon),
        )
        sign = 1.0 if pressure_levels_pa[1] - pressure_levels_pa[0] > 0 else -1.0
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        meridional_advection_integrated[:, :, lon_start:lon_end] = (
            sign / gravity * _trapz(meridional_advection_divergence, pressure_broadcast, axis=1)
        )
        del meridional_advection_divergence, moist_static_energy, beta, beta_nonzero, meridional_wind, surface_pressure
        gc.collect()
        _LOG.info("  v_mse block %s complete", lon_block)

    out_path = output_directory / ("Adv_%d_%s.nc" % (year, month))
    dataset_out = xarray.Dataset(
        {
            "v_mse": (
                ("time", "latitude", "longitude"),
                meridional_advection_integrated.astype(np.float32),
                {
                    "units": "W m-2",
                    "long_name": "vertically integrated meridional MSE advection",
                },
            ),
            "u_mse": (
                ("time", "latitude", "longitude"),
                zonal_advection_integrated.astype(np.float32),
                {
                    "units": "W m-2",
                    "long_name": "vertically integrated zonal MSE advection",
                },
            ),
        },
        coords={"latitude": latitude, "longitude": longitude},
    )
    dataset_out.to_netcdf(str(out_path))
    _LOG.info("Saved advection file: %s", out_path)
