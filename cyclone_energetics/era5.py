from __future__ import annotations

"""Dimension-safe ERA5 data reader.

Normalises dimension names to canonical form and returns numpy arrays
in a guaranteed order, so downstream computation code does not depend
on the dimension layout of the source NetCDF file.

Canonical ordering:
    4-D fields: (time, level, latitude, longitude)
    3-D fields: (time, latitude, longitude)
"""

import pathlib
import typing

import numpy as np
import numpy.typing as npt
import xarray

_DIM_ALIASES: typing.Dict[str, str] = {
    "valid_time": "time",
    "pressure_level": "level",
    "lat": "latitude",
    "lon": "longitude",
}

_4D_ORDER: typing.Tuple[str, ...] = ("time", "level", "latitude", "longitude")
_3D_ORDER: typing.Tuple[str, ...] = ("time", "latitude", "longitude")


def _normalise(ds: xarray.Dataset) -> xarray.Dataset:
    """Rename known dimension variants to canonical names."""
    rename = {k: v for k, v in _DIM_ALIASES.items() if k in ds.dims}
    return ds.rename(rename) if rename else ds


def read_field(
    path: pathlib.Path,
    variable: str,
    *,
    latitude_slice: typing.Optional[slice] = None,
    longitude_slice: typing.Optional[slice] = None,
    time_slice: typing.Optional[slice] = None,
    dtype: type = np.float64,
) -> npt.NDArray:
    """Read a variable and return in canonical dimension order as numpy.

    4-D fields are returned as ``(time, level, latitude, longitude)``.
    3-D fields are returned as ``(time, latitude, longitude)``.
    Dimension names in the source file are normalised automatically so the
    caller never needs to know the on-disk layout.
    """
    with xarray.open_dataset(str(path), decode_times=False) as ds:
        ds = _normalise(ds)
        da = ds[variable]
        if time_slice is not None:
            da = da.isel(time=time_slice)
        if latitude_slice is not None:
            da = da.isel(latitude=latitude_slice)
        if longitude_slice is not None:
            da = da.isel(longitude=longitude_slice)
        canonical = _4D_ORDER if "level" in da.dims else _3D_ORDER
        present = tuple(d for d in canonical if d in da.dims)
        da = da.transpose(*present)
        return np.asarray(da.values, dtype=dtype)


def read_coordinates(
    path: pathlib.Path,
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(latitude, longitude)`` coordinate arrays."""
    with xarray.open_dataset(str(path), decode_times=False) as ds:
        ds = _normalise(ds)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return lat, lon


def read_pressure_levels(
    path: pathlib.Path,
    *,
    scale_to_pa: float = 100.0,
) -> npt.NDArray[np.float64]:
    """Return pressure levels in Pa (input assumed hPa by default)."""
    with xarray.open_dataset(str(path), decode_times=False) as ds:
        ds = _normalise(ds)
        return np.asarray(ds["level"].values, dtype=np.float64) * scale_to_pa


def read_n_time(path: pathlib.Path) -> int:
    """Return the number of timesteps in a file."""
    with xarray.open_dataset(str(path), decode_times=False) as ds:
        ds = _normalise(ds)
        return int(ds.sizes["time"])
