from __future__ import annotations

"""Compute the transient-eddy (TE) divergence.

Produces files ``TE_YYYY_MM.nc`` containing the meridional divergence of
the vertically integrated transient-eddy MSE flux:

    ∇_y · (1/g) ∫₀ᵖˢ  v' · h' · β²  dp

where v' = v − ⟨v⟩ and h' = h − ⟨h⟩ are anomalies from the monthly
mean, h = c_p T + g Z + L_v q is the moist static energy, and β is the
below-ground weighting factor.

The transient-eddy flux divergence captures the contribution of
sub-monthly (synoptic-scale) variability to the poleward energy
transport.  It is subsequently smoothed and poleward-integrated in
downstream pipeline steps.
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


def _compute_divergence(
    *,
    field: xarray.DataArray,
) -> xarray.DataArray:
    """Meridional divergence on the sphere: (1/(a cos φ)) ∂/∂φ [F cos φ]."""
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
    vnames = variable_names or gridded_data.DEFAULT_VARIABLE_NAMES
    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            _LOG.info("Processing TE flux: year=%s month=%s", year, month)
            _process_single_month_te(
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
    kw = dict(
        data_directory=data_directory, year=year, month=month,
        filename_pattern=filename_pattern, subdirectories=subdirectories,
    )
    t_path = _resolve(field="temperature", **kw)
    q_path = _resolve(field="specific_humidity", **kw)
    ps_path = _resolve(field="surface_pressure", **kw)
    z_path = _resolve(field="geopotential", **kw)
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
    n_blocks = (n_lat + chunk - 1) // chunk
    dvmsedt_values = np.zeros((n_time, n_lat, n_lon), dtype=np.float64)

    for lat_block in range(n_blocks):
        lat_start = lat_block * chunk
        lat_end = min((lat_block + 1) * chunk, n_lat)
        _LOG.info("Latitude block: %s to %s", lat_start, lat_end)
        lat_sl = slice(lat_start, lat_end)

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
            )
            / constants.GRAVITY
        ).assign_coords(level=plev_pa)

        beta = gridded_data.compute_beta_mask(
            pressure_levels=pressure_levels, surface_pressure=ps,
        )

        mse = (
            constants.CPD * ta
            + constants.GRAVITY * zg
            + constants.LATENT_HEAT_VAPORIZATION * hus
        )
        del ta, hus, zg

        v = gridded_data.open_field(
            v_path, vn["meridional_wind"], latitude_slice=lat_sl,
        ).assign_coords(level=plev_pa)

        mse_prime = mse - mse.mean(dim="time")
        v_prime = v - v.mean(dim="time")
        del mse, v

        te_flux = (v_prime * mse_prime * beta * beta).fillna(0.0)
        del v_prime, mse_prime, beta

        sign = 1.0 if float(plev_pa[1] - plev_pa[0]) > 0 else -1.0
        integrated = sign * te_flux.integrate("level") / constants.GRAVITY
        del te_flux

        dvmsedt_values[:, lat_start:lat_end, :] = integrated.values
        del integrated
        gc.collect()
        _LOG.info("Vertical integration completed for block %s", lat_block)

    dvmsedt = xarray.DataArray(
        dvmsedt_values,
        dims=("time", "latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )

    te_divergence = _compute_divergence(field=dvmsedt)

    out_path = output_directory / ("TE_%d_%s.nc" % (year, month))
    ds_out = te_divergence.astype(np.float32).to_dataset(name="TE")
    ds_out["TE"].attrs = {
        "units": "W m-2",
        "long_name": "divergence of vertically integrated transient-eddy MSE flux",
    }
    ds_out.to_netcdf(str(out_path))
    _LOG.info("Saved TE file: %s", out_path)
