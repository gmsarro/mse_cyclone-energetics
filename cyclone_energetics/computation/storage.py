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


def compute_storage_term(
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
            _LOG.info("Computing dh/dt: year=%s month=%s", year, month)
            _process_single_month_dhdt(
                year=year,
                month=month,
                data_directory=data_directory,
                output_directory=output_directory,
                filename_pattern=filename_pattern,
                variable_names=resolved_variable_names,
                subdirectories=subdirectories,
            )


def _process_single_month_dhdt(
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

    latitude, longitude = gridded_data.read_coordinates(temperature_path)
    n_timesteps = gridded_data.read_n_time(temperature_path)
    pressure_levels_pa = gridded_data.read_pressure_levels(humidity_path)
    pressure_levels = xarray.DataArray(
        pressure_levels_pa, dims=["level"], coords={"level": pressure_levels_pa},
    )

    time_coord = gridded_data.open_field(
        temperature_path, variable=variable_names["temperature"],
        latitude_slice=slice(0, 1),
    ).coords["time"]
    dt_step = gridded_data.infer_time_step_seconds(time_coord)
    dt_centered = 2.0 * dt_step
    dt_forward = dt_step

    n_latitude = len(latitude)
    n_longitude = len(longitude)
    chunk = min(_DEFAULT_CHUNK_SIZE, n_latitude)
    n_blocks = (n_latitude + chunk - 1) // chunk

    surface_pressure_all = gridded_data.open_field(
        surface_pressure_path, variable=variable_names["surface_pressure"],
    )
    surface_pressure_mean = surface_pressure_all.mean(dim="time")
    del surface_pressure_all

    storage_term = np.zeros((n_timesteps, n_latitude, n_longitude), dtype=np.float64)

    for lat_block in range(n_blocks):
        lat_start = lat_block * chunk
        lat_end = min((lat_block + 1) * chunk, n_latitude)
        _LOG.info("  Latitude block: %s to %s", lat_start, lat_end)

        lat_sl = slice(lat_start, lat_end)
        surface_pressure_chunk = surface_pressure_mean.isel(latitude=lat_sl)

        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels,
            surface_pressure=surface_pressure_chunk,
        )

        temperature = gridded_data.open_field(
            temperature_path, variable=variable_names["temperature"], latitude_slice=lat_sl,
        ).assign_coords(level=pressure_levels_pa)
        specific_humidity = gridded_data.open_field(
            humidity_path, variable=variable_names["specific_humidity"], latitude_slice=lat_sl,
        ).assign_coords(level=pressure_levels_pa)

        moist_static_energy = (
            constants.CPD * temperature + constants.LATENT_HEAT_VAPORIZATION * specific_humidity
        ) * beta
        del temperature, specific_humidity, beta

        dmse_dt = xarray.zeros_like(moist_static_energy)
        dmse_dt[dict(time=slice(1, -1))] = (
            (moist_static_energy.isel(time=slice(2, None)).values
             - moist_static_energy.isel(time=slice(None, -2)).values)
            / dt_centered
        )
        dmse_dt[dict(time=0)] = (
            (moist_static_energy.isel(time=1).values
             - moist_static_energy.isel(time=0).values) / dt_forward
        )
        dmse_dt[dict(time=-1)] = (
            (moist_static_energy.isel(time=-1).values
             - moist_static_energy.isel(time=-2).values) / dt_forward
        )
        del moist_static_energy

        dmse_dt = dmse_dt.fillna(0.0)

        sign = 1.0 if float(pressure_levels_pa[1] - pressure_levels_pa[0]) > 0 else -1.0
        integrated = sign * dmse_dt.integrate("level") / constants.GRAVITY
        del dmse_dt

        storage_term[:, lat_start:lat_end, :] = integrated.values
        del integrated
        gc.collect()
        _LOG.info("  Block %s complete", lat_block)

    _LOG.info("Vertical integration complete for year=%s month=%s", year, month)

    out_path = output_directory / ("tend_%d_%s_2.nc" % (year, month))
    result = xarray.DataArray(
        storage_term.astype(np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )
    dataset_out = result.to_dataset(name="tend")
    dataset_out["tend"].attrs = {
        "units": "W/m^2",
        "long_name": "time tendency of vertically integrated moist static energy",
    }
    dataset_out.to_netcdf(str(out_path))
    _LOG.info("Saved dh/dt file: %s", out_path)
