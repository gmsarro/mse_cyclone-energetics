from __future__ import annotations

"""Compute the vertically integrated MSE storage term (dh/dt).

The storage term is the time tendency of the vertically integrated moist
static energy (MSE).  For each 6-hourly timestep the 3-D MSE is
computed as

    h = (c_p * T  +  L_v * q) * beta

where *beta* is the below-ground weighting factor computed from the
**time-mean** surface pressure.  The centred-difference time derivative
is then vertically integrated and written to a NetCDF file.

The computation is performed in latitude chunks to keep memory usage
within reasonable bounds.
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
_DT_CENTERED: float = 43200.0   # 12 h in seconds (centred difference)
_DT_FORWARD: float = 21600.0    # 6 h in seconds (forward / backward diff)


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
    vnames = variable_names or gridded_data.DEFAULT_VARIABLE_NAMES
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Computing dh/dt: year=%s month=%s", year, month)
            _process_single_month_dhdt(
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
    kw = dict(
        data_directory=data_directory, year=year, month=month,
        filename_pattern=filename_pattern, subdirectories=subdirectories,
    )
    t_path = _resolve(field="temperature", **kw)
    q_path = _resolve(field="specific_humidity", **kw)
    ps_path = _resolve(field="surface_pressure", **kw)

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
    n_blocks = (n_lat + chunk - 1) // chunk

    ps_all = gridded_data.open_field(ps_path, vn["surface_pressure"])
    ps_mean = ps_all.mean(dim="time")
    del ps_all

    dvmsedt = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    for lat_block in range(n_blocks):
        lat_start = lat_block * chunk
        lat_end = min((lat_block + 1) * chunk, n_lat)
        n_chunk = lat_end - lat_start
        _LOG.info("  Latitude block: %s to %s", lat_start, lat_end)

        lat_sl = slice(lat_start, lat_end)
        ps_chunk = ps_mean.isel(latitude=lat_sl)

        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels,
            surface_pressure=ps_chunk,
        )

        ta = gridded_data.open_field(
            t_path, vn["temperature"], latitude_slice=lat_sl,
        ).assign_coords(level=plev_pa)
        hus = gridded_data.open_field(
            q_path, vn["specific_humidity"], latitude_slice=lat_sl,
        ).assign_coords(level=plev_pa)

        mse = (
            constants.CPD * ta + constants.LATENT_HEAT_VAPORIZATION * hus
        ) * beta
        del ta, hus, beta

        dmsedt = xarray.zeros_like(mse)
        dmsedt[dict(time=slice(1, -1))] = (
            (mse.isel(time=slice(2, None)).values
             - mse.isel(time=slice(None, -2)).values)
            / _DT_CENTERED
        )
        dmsedt[dict(time=0)] = (
            (mse.isel(time=1).values - mse.isel(time=0).values) / _DT_FORWARD
        )
        dmsedt[dict(time=-1)] = (
            (mse.isel(time=-1).values - mse.isel(time=-2).values) / _DT_FORWARD
        )
        del mse

        dmsedt = dmsedt.fillna(0.0)

        sign = 1.0 if float(plev_pa[1] - plev_pa[0]) > 0 else -1.0
        integrated = sign * dmsedt.integrate("level") / constants.GRAVITY
        del dmsedt

        dvmsedt[:, lat_start:lat_end, :] = integrated.values
        del integrated
        gc.collect()
        _LOG.info("  Block %s complete", lat_block)

    _LOG.info("Vertical integration complete for year=%s month=%s", year, month)

    out_path = output_directory / ("tend_%d_%s_2.nc" % (year, month))
    result = xarray.DataArray(
        dvmsedt.astype(np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )
    ds_out = result.to_dataset(name="tend")
    ds_out["tend"].attrs = {
        "units": "W/m^2",
        "long_name": "time tendency of vertically integrated moist static energy",
    }
    ds_out.to_netcdf(str(out_path))
    _LOG.info("Saved dh/dt file: %s", out_path)
