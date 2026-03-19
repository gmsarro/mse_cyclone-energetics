from __future__ import annotations

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing as npt
import xarray

import cyclone_energetics.composites.builder as builder
import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_LAND_THRESHOLD: float = 0.5


def _load_land_sea_mask(
    *,
    lsm_path: pathlib.Path,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    _LOG.info("Loading land-sea mask from %s", lsm_path)
    with netCDF4.Dataset(str(lsm_path)) as ds:
        lat = np.asarray(ds["latitude"][:], dtype=np.float32)
        lon = np.asarray(ds["longitude"][:], dtype=np.float32)
        lsm = np.asarray(ds["lsm"][:])
    if lsm.ndim == 3:
        lsm = lsm[0]
    lsm = lsm.astype(np.float32)
    lon = builder._to_360(lon)
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    lsm = lsm[:, lon_order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        lsm = lsm[::-1, :]
    _LOG.info(
        "  LSM: lat [%s, %s], lon [%s, %s], shape %s",
        lat[0], lat[-1], lon[0], lon[-1], lsm.shape,
    )
    return lat, lon, lsm


def _is_center_over_land(
    *,
    lat0: float,
    lon0: float,
    lsm_lat: npt.NDArray,
    lsm_lon: npt.NDArray,
    lsm_data: npt.NDArray,
) -> bool:
    ilat = builder._nearest_idx(
        a_sorted=lsm_lat,
        q=np.array([lat0], dtype=np.float32),
    )[0]
    ilon = builder._nearest_idx(
        a_sorted=lsm_lon,
        q=np.array([lon0 % 360], dtype=np.float32),
    )[0]
    return bool(lsm_data[ilat, ilon] > _LAND_THRESHOLD)


def add_land_fraction_to_composites(
    *,
    composite_path: pathlib.Path,
    track_path: pathlib.Path,
    lsm_path: pathlib.Path,
    hemisphere: str,
    intensity_min: int,
    intensity_max: int,
    storm_lat: npt.NDArray,
    year_start: int = 2000,
    year_end: int = 2015,
    lat_band_half_width: float = constants.STORM_LAT_BAND_HALF_WIDTH,
) -> None:
    lsm_lat, lsm_lon, lsm_data = _load_land_sea_mask(lsm_path=lsm_path)

    lat_rel = np.linspace(
        -builder._R_WINDOW_DEG + builder._DRES / 2,
        builder._R_WINDOW_DEG - builder._DRES / 2,
        builder._NY,
    ).astype("float32")
    lon_rel = np.linspace(
        -builder._R_WINDOW_DEG + builder._DRES / 2,
        builder._R_WINDOW_DEG - builder._DRES / 2,
        builder._NX,
    ).astype("float32")

    with netCDF4.Dataset(str(track_path)) as ds:
        time_index = np.array(ds["time"][:]).astype(np.int64) - 1
        lat_trk = np.array(ds["latitude"][:]).astype(np.float32)
        lon_trk = builder._to_360(
            np.array(ds["longitude"][:]).astype(np.float32),
        )
        inten_trk = np.array(ds["intensity"][:]).astype(np.float32)

    steps_per_year = constants.TIMESTEPS_PER_YEAR
    year_from_idx = (
        constants.ERA5_BASE_YEAR + time_index // steps_per_year
    ).astype(np.int16)
    mask = (
        (year_from_idx >= year_start) & (year_from_idx < year_end)
        & (inten_trk >= intensity_min) & (inten_trk <= intensity_max)
    )
    idx_sel = np.where(mask)[0]
    lat_arr = lat_trk[idx_sel]
    lon_arr = lon_trk[idx_sel]
    step_arr = time_index[idx_sel]

    _LOG.info("  %d snapshots after year/intensity filter", len(idx_sel))

    doy_arr = (step_arr % steps_per_year) // 4
    mcum = constants.NOLEAP_MONTH_CUMULATIVE
    month_arr = np.searchsorted(mcum[1:], doy_arr, side="right").astype(np.int8) + 1
    midx_arr = month_arr - 1
    target_lat_arr = storm_lat[midx_arr]
    in_band = np.abs(lat_arr - target_lat_arr) <= lat_band_half_width
    band_idx = np.where(in_band)[0]

    _LOG.info("  %d snapshots pass lat-band filter", len(band_idx))

    land_frac_accum = np.zeros(
        (12, builder._NY, builder._NX), dtype=np.float64,
    )
    count_arr = np.zeros(12, dtype=np.int32)
    land_centers = np.zeros(12, dtype=np.int32)
    ocean_centers = np.zeros(12, dtype=np.int32)

    for bi in band_idx:
        lat0 = float(lat_arr[bi])
        lon0 = float(lon_arr[bi])
        midx = int(midx_arr[bi])

        lsm_patch = builder._extract_patch_np(
            arr2d=lsm_data,
            lat_vals=lsm_lat,
            lon_vals=lsm_lon,
            lat0=lat0,
            lon0=lon0,
            lat_offset=lat_rel,
            lon_offset=lon_rel,
        )
        land_frac_accum[midx] += lsm_patch
        count_arr[midx] += 1

        if _is_center_over_land(
            lat0=lat0,
            lon0=lon0,
            lsm_lat=lsm_lat,
            lsm_lon=lsm_lon,
            lsm_data=lsm_data,
        ):
            land_centers[midx] += 1
        else:
            ocean_centers[midx] += 1

    for m in range(12):
        if count_arr[m] > 0:
            land_frac_accum[m] /= count_arr[m]
        else:
            land_frac_accum[m] = np.nan

    _LOG.info(
        "  Total: %d accepted, %d land centers, %d ocean centers",
        int(count_arr.sum()),
        int(land_centers.sum()),
        int(ocean_centers.sum()),
    )

    ds_existing = xarray.open_dataset(str(composite_path))
    ds_existing["composite_land_frac"] = (
        ("month", "y", "x"),
        land_frac_accum.astype(np.float32),
    )
    ds_existing["composite_land_frac"].attrs["long_name"] = (
        "Fraction of composite area over land"
    )
    ds_existing["composite_land_frac"].attrs["units"] = "0-1"

    tmp_path = str(composite_path) + ".tmp"
    ds_existing.to_netcdf(tmp_path, format="NETCDF4")
    ds_existing.close()
    pathlib.Path(tmp_path).replace(composite_path)
    _LOG.info("Updated %s with composite_land_frac", composite_path)


def build_land_ocean_composites(
    *,
    year_start: int,
    year_end: int,
    intensity_min: int,
    intensity_max: int,
    track_path: pathlib.Path,
    lsm_path: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    dhdt_directory: pathlib.Path,
    radiation_directory: pathlib.Path,
    z_directory: pathlib.Path,
    t2m_directory: pathlib.Path,
    q_directory: pathlib.Path,
    output_directory: pathlib.Path,
    storm_lat: npt.NDArray,
    hemisphere: str = "NH",
    vorticity_directory: typing.Optional[pathlib.Path] = None,
    lat_band_half_width: float = constants.STORM_LAT_BAND_HALF_WIDTH,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    tag = "Weak" if intensity_max <= 5 else "Intense"
    _LOG.info(
        "Building land/ocean composites: hemisphere=%s %s intensity=[%s,%s]",
        hemisphere, tag, intensity_min, intensity_max,
    )

    lsm_lat, lsm_lon, lsm_data = _load_land_sea_mask(lsm_path=lsm_path)

    for surface_label, select_land in [("Land", True), ("Ocean", False)]:
        _LOG.info("  Building %s composites...", surface_label)

        def _make_filter(
            land: bool,
        ) -> typing.Callable[[float, float], bool]:
            def _fn(lat: float, lon: float) -> bool:
                over_land = _is_center_over_land(
                    lat0=lat,
                    lon0=lon,
                    lsm_lat=lsm_lat,
                    lsm_lon=lsm_lon,
                    lsm_data=lsm_data,
                )
                return over_land == land
            return _fn

        builder.build_cyclone_composites(
            year_start=year_start,
            year_end=year_end,
            hemisphere=hemisphere,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            track_path=track_path,
            integrated_flux_directory=integrated_flux_directory,
            vint_directory=vint_directory,
            dhdt_directory=dhdt_directory,
            radiation_directory=radiation_directory,
            z_directory=z_directory,
            t2m_directory=t2m_directory,
            q_directory=q_directory,
            output_directory=output_directory,
            storm_lat=storm_lat,
            vorticity_directory=vorticity_directory,
            lat_band_half_width=lat_band_half_width,
            snapshot_filter=_make_filter(select_land),
            output_suffix=surface_label,
        )
