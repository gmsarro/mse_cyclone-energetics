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


def _compute_divergence(
    *,
    field: xarray.DataArray,
) -> xarray.DataArray:
    lat_rad = np.deg2rad(field.latitude.values)
    cos_lat = np.cos(lat_rad)
    lat_axis = field.dims.index("latitude")

    shape = [1] * field.ndim
    shape[lat_axis] = len(cos_lat)
    cos_lat_nd = cos_lat.reshape(shape)

    field_cos = field.values * cos_lat_nd
    grad = np.gradient(field_cos, lat_rad, axis=lat_axis)
    divergence_values = grad / (constants.EARTH_RADIUS * cos_lat_nd)

    return xarray.DataArray(
        divergence_values,
        dims=field.dims,
        coords=field.coords,
    )


def compute_transient_eddy_flux(
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
            _LOG.info("Processing TE flux: year=%s month=%s", year, month)
            _process_single_month_te(
                year=year,
                month=month,
                data_directory=data_directory,
                output_directory=output_directory,
                filename_pattern=filename_pattern,
                variable_names=resolved_variable_names,
                subdirectories=subdirectories,
            )


def _process_single_month_te(
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
    n_blocks = (n_latitude + chunk - 1) // chunk
    te_flux_integrated = np.zeros((n_timesteps, n_latitude, n_longitude), dtype=np.float64)

    for lat_block in range(n_blocks):
        lat_start = lat_block * chunk
        lat_end = min((lat_block + 1) * chunk, n_latitude)
        _LOG.info("Latitude block: %s to %s", lat_start, lat_end)
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
            )
            / constants.GRAVITY
        ).assign_coords(level=pressure_levels_pa)

        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels, surface_pressure=surface_pressure,
        )

        moist_static_energy = (
            constants.CPD * temperature
            + constants.GRAVITY * geopotential_height
            + constants.LATENT_HEAT_VAPORIZATION * specific_humidity
        )
        del temperature, specific_humidity, geopotential_height

        meridional_wind = gridded_data.open_field(
            meridional_wind_path, variable=variable_names["meridional_wind"], latitude_slice=lat_sl,
        ).assign_coords(level=pressure_levels_pa)

        mse_anomaly = moist_static_energy - moist_static_energy.mean(dim="time")
        wind_anomaly = meridional_wind - meridional_wind.mean(dim="time")
        del moist_static_energy, meridional_wind

        transient_eddy_flux = (wind_anomaly * mse_anomaly * beta * beta).fillna(0.0)
        del wind_anomaly, mse_anomaly, beta

        sign = 1.0 if float(pressure_levels_pa[1] - pressure_levels_pa[0]) > 0 else -1.0
        integrated = sign * transient_eddy_flux.integrate("level") / constants.GRAVITY
        del transient_eddy_flux

        te_flux_integrated[:, lat_start:lat_end, :] = integrated.values
        del integrated
        gc.collect()
        _LOG.info("Vertical integration completed for block %s", lat_block)

    te_flux_field = xarray.DataArray(
        te_flux_integrated,
        dims=("time", "latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )

    te_divergence = _compute_divergence(field=te_flux_field)

    out_path = output_directory / ("TE_%d_%s.nc" % (year, month))
    dataset_out = te_divergence.astype(np.float32).to_dataset(name="TE")
    dataset_out["TE"].attrs = {
        "units": "W m-2",
        "long_name": "divergence of vertically integrated transient-eddy MSE flux",
    }
    dataset_out.to_netcdf(str(out_path))
    _LOG.info("Saved TE file: %s", out_path)
