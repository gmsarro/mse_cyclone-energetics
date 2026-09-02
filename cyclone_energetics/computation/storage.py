"""MSE storage term: time tendency of the column enthalpy.

    S(t) = d/dt [ g^-1 int_{p_top}^{p_sfc(t)} (cp T + Lv q) dp ]

The column enthalpy E(t) is evaluated at every 6-hourly step with the
instantaneous surface pressure as the lower limit (exact partial bottom layer,
see gridded_data.integrate_column_to_surface) and then differenced in time
(centred; forward/backward at the month ends). Differencing h on fixed pressure
levels below a time-mean surface would omit the column-mass term
h_sfc dp_sfc/dt / g, which is O(100 W m-2) in the pressure falls and rises of
individual cyclones and would otherwise alias into the residual surface flux.
"""

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

_DEFAULT_TIME_BLOCK: int = 8


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


def column_enthalpy(
    *,
    temperature: np.ndarray,
    specific_humidity: np.ndarray,
    surface_pressure: np.ndarray,
    pressure_levels_pa: np.ndarray,
) -> np.ndarray:
    """E = g^-1 int_0^{p_sfc} (cp T + Lv q) dp  [J m-2] for a (time, level, lat, lon) block."""
    enthalpy = constants.CPD * temperature
    enthalpy += constants.LATENT_HEAT_VAPORIZATION * specific_humidity
    return gridded_data.integrate_column_to_surface(
        field=enthalpy,
        surface_pressure=surface_pressure,
        pressure_levels=pressure_levels_pa,
    ) / constants.GRAVITY


def _process_single_month_dhdt(
    *,
    year: int,
    month: str,
    data_directory: pathlib.Path,
    output_directory: pathlib.Path,
    filename_pattern: str,
    variable_names: typing.Dict[str, str],
    subdirectories: typing.Optional[typing.Dict[str, str]],
    time_block: int = _DEFAULT_TIME_BLOCK,
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

    time_coord = gridded_data.open_field(
        temperature_path, variable=variable_names["temperature"],
        latitude_slice=slice(0, 1),
    ).coords["time"]
    dt_step = gridded_data.infer_time_step_seconds(time_coord)

    surface_pressure_all = gridded_data.read_field(
        surface_pressure_path, variable=variable_names["surface_pressure"],
    )
    if surface_pressure_all.shape[0] != n_timesteps:
        raise ValueError(
            "surface pressure has %d time steps, temperature %d"
            % (surface_pressure_all.shape[0], n_timesteps)
        )

    # Files are typically [time, level, lat, lon]; reading contiguous time blocks
    # is much faster than latitude slabs.
    energy = np.zeros((n_timesteps, len(latitude), len(longitude)), dtype=np.float64)
    for block_start in range(0, n_timesteps, time_block):
        block = slice(block_start, min(block_start + time_block, n_timesteps))
        temperature = gridded_data.read_field(
            temperature_path, variable=variable_names["temperature"], time_slice=block,
        )
        specific_humidity = gridded_data.read_field(
            humidity_path, variable=variable_names["specific_humidity"], time_slice=block,
        )
        energy[block] = column_enthalpy(
            temperature=temperature,
            specific_humidity=specific_humidity,
            surface_pressure=surface_pressure_all[block],
            pressure_levels_pa=pressure_levels_pa,
        )
        del temperature, specific_humidity
        gc.collect()
    _LOG.info("Column enthalpy complete for year=%s month=%s", year, month)

    storage_term = np.empty_like(energy)
    storage_term[1:-1] = (energy[2:] - energy[:-2]) / (2.0 * dt_step)
    storage_term[0] = (energy[1] - energy[0]) / dt_step
    storage_term[-1] = (energy[-1] - energy[-2]) / dt_step
    del energy

    out_path = output_directory / ("tend_%d_%s_2.nc" % (year, month))
    result = xarray.DataArray(
        storage_term.astype(np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": time_coord.values, "latitude": latitude, "longitude": longitude},
    )
    dataset_out = result.to_dataset(name="tend")
    dataset_out["tend"].attrs = {
        "units": "W/m^2",
        "long_name": "time tendency of the column moist static energy (cp T + Lv q) "
                     "integrated to the instantaneous surface pressure",
        "method": "E(t) = g^-1 int_0^{p_sfc(t)} h dp with an exact partial bottom layer; "
                  "centred time difference; includes the column-mass term h_sfc dp_sfc/dt / g",
    }
    dataset_out.to_netcdf(str(out_path))
    _LOG.info("Saved dh/dt file: %s", out_path)

