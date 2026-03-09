from __future__ import annotations

import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing as npt
import xarray

_DIM_ALIASES: typing.Dict[str, str] = {
    "valid_time": "time",
    "pressure_level": "level",
    "lat": "latitude",
    "lon": "longitude",
}

_CANONICAL_CANDIDATES: typing.Dict[str, typing.List[str]] = {
    "time": ["time", "valid_time"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon"],
    "level": ["level", "pressure_level", "plev"],
}

_4D_ORDER: typing.Tuple[str, ...] = ("time", "level", "latitude", "longitude")
_3D_ORDER: typing.Tuple[str, ...] = ("time", "latitude", "longitude")


def resolve_dimension_name(
    dataset: netCDF4.Dataset | xarray.Dataset,
    *,
    standard_name: str,
) -> str:
    candidates = _CANONICAL_CANDIDATES.get(standard_name, [standard_name])
    if isinstance(dataset, xarray.Dataset):
        pool = set(dataset.dims) | set(dataset.coords)
    else:
        pool = set(dataset.dimensions.keys()) | set(dataset.variables.keys())
    for name in candidates:
        if name in pool:
            return name
    raise KeyError(
        "No dimension matching '%s' found; available: %s" % (standard_name, sorted(pool))
    )


def _normalise(ds: xarray.Dataset) -> xarray.Dataset:
    rename = {k: v for k, v in _DIM_ALIASES.items() if k in ds.dims}
    return ds.rename(rename) if rename else ds


def open_field(
    path: pathlib.Path,
    *,
    variable: str,
    latitude_slice: typing.Optional[slice] = None,
    longitude_slice: typing.Optional[slice] = None,
    time_slice: typing.Optional[slice] = None,
    dtype: type = np.float64,
) -> xarray.DataArray:
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
        da = da.transpose(*present).astype(dtype)
        da.load()
    return da


def read_field(
    path: pathlib.Path,
    *,
    variable: str,
    latitude_slice: typing.Optional[slice] = None,
    longitude_slice: typing.Optional[slice] = None,
    time_slice: typing.Optional[slice] = None,
    dtype: type = np.float64,
) -> npt.NDArray[typing.Any]:
    da = open_field(
        path,
        variable=variable,
        latitude_slice=latitude_slice,
        longitude_slice=longitude_slice,
        time_slice=time_slice,
        dtype=dtype,
    )
    return da.values


def read_coordinates(
    path: pathlib.Path,
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    with xarray.open_dataset(str(path), decode_times=False) as ds:
        ds = _normalise(ds)
        return np.asarray(ds["level"].values, dtype=np.float64) * scale_to_pa


def read_n_time(path: pathlib.Path) -> int:
    with xarray.open_dataset(str(path), decode_times=False) as ds:
        ds = _normalise(ds)
        return int(ds.sizes["time"])


DEFAULT_FILENAME_PATTERN: str = "era5_{variable}_{year}_{month}.6hrly.nc"

DEFAULT_SUBDIRECTORIES: typing.Dict[str, str] = {
    "temperature": "t",
    "specific_humidity": "q",
    "surface_pressure": "ps",
    "geopotential": "z",
    "meridional_wind": "v",
    "zonal_wind": "u",
}

DEFAULT_VARIABLE_NAMES: typing.Dict[str, str] = {
    "temperature": "t",
    "specific_humidity": "q",
    "surface_pressure": "sp",
    "geopotential": "z",
    "meridional_wind": "v",
    "zonal_wind": "u",
}


def resolve_path(
    *,
    data_directory: pathlib.Path,
    field: str,
    year: int,
    month: str,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    subdirectories: typing.Optional[typing.Dict[str, str]] = None,
) -> pathlib.Path:
    subs = subdirectories or DEFAULT_SUBDIRECTORIES
    subdir = subs[field]
    filename = filename_pattern.format(variable=subdir, year=year, month=month)
    return data_directory / subdir / filename


def compute_beta_mask(
    *,
    pressure_levels: xarray.DataArray,
    surface_pressure: xarray.DataArray,
) -> xarray.DataArray:
    p = pressure_levels
    ps = surface_pressure
    p_above = p.shift(level=1).fillna(p.isel(level=0))

    beta = (ps - p_above) / (p - p_above)
    beta = beta.where(p >= ps, 1.0)

    beta_surface = (
        (ps - p_above.isel(level=-1))
        / (p.isel(level=-1) - p_above.isel(level=-1))
    )
    beta = xarray.where(
        p.level == p.level.values[-1], beta_surface, beta,
    )

    beta = beta.where(p_above <= ps, 0.0)
    return beta


def infer_time_step_seconds(
    time_coord: xarray.DataArray,
) -> float:
    delta = time_coord.values[1] - time_coord.values[0]
    if np.issubdtype(time_coord.dtype, np.datetime64):
        return float(np.timedelta64(delta, "ns").astype(np.float64)) / 1e9
    return float(delta) * 3600.0
