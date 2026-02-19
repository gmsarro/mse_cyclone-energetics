from __future__ import annotations

"""Cyclone-centred monthly composites on a storm-relative 0.25° grid.

Based on the ``create_composites_AI.py`` / ``create_composites_intense_AI.py``
scripts.  The module composites **meridionally integrated** energy-budget
terms (TE, energy, SHF, Swabs, OLR, u_mse, v_mse, dh/dt) extracted from
the ``Integrated_Fluxes`` and ``New_Integrated_Fluxes`` NetCDF files
produced by step 5 of the pipeline.

Key design choices:

* **Storm-relative grid** — a 120×120 patch (±15° at 0.25° resolution)
  centred on the cyclone location, using nearest-neighbour lookup.
* **Monthly latitude-band filter** — only snapshots where the storm centre
  is within ±5° of the monthly-mean storm-track latitude are retained.
* **LRU dataset cache** — keeps the most-recent monthly flux files open,
  avoiding repeated I/O.

The module works for both hemispheres (SH / NH) and for different
intensity ranges (e.g. 1–5 CVU for non-intense, ≥6 CVU for intense).
"""

import collections
import csv
import logging
import os
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import xarray as xr

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_R_WINDOW_DEG: float = 15.0
_DRES: float = 0.25
_NY: int = int(round(2 * _R_WINDOW_DEG / _DRES))
_NX: int = _NY

_STEPS_PER_YEAR: int = 365 * 4


# ---------------------------------------------------------------------------
# LRU dataset cache (from the AI scripts)
# ---------------------------------------------------------------------------
class _DSCache:
    """Small LRU cache for open xarray datasets keyed by full path."""

    def __init__(self, *, capacity: int = 4) -> None:
        self._cap = capacity
        self._d: collections.OrderedDict[str, xr.Dataset] = collections.OrderedDict()

    def get(self, *, path: str, needed_vars: list) -> xr.Dataset:
        if path in self._d:
            ds = self._d.pop(path)
            self._d[path] = ds
            return ds
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
        if "lon" in ds:
            ds = ds.assign_coords(lon=_to_360(ds["lon"].values))
        keep = [v for v in needed_vars if v in ds.variables]
        ds = ds[keep]
        self._d[path] = ds
        if len(self._d) > self._cap:
            _, old = self._d.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return ds

    def close_all(self) -> None:
        for _, ds in self._d.items():
            try:
                ds.close()
            except Exception:
                pass
        self._d.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_360(lon: npt.NDArray) -> npt.NDArray:
    return (lon % 360 + 360) % 360


def _day_hour_from_track_index(
    *,
    step_6h: int,
) -> tuple:
    """Return ``(year, month_1based, day_in_month_0based, hour6_index)``."""
    yy = 1979 + step_6h // _STEPS_PER_YEAR
    doy = (step_6h % _STEPS_PER_YEAR) // 4
    hour6 = step_6h % 4
    mcum = constants.NOLEAP_MONTH_CUMULATIVE
    m = int(np.searchsorted(mcum[1:], doy, side="right") + 1)
    day_in_month = int(doy - mcum[m - 1])
    return int(yy), m, day_in_month, int(hour6)


def _extract_patch(
    da: xr.DataArray,
    lat0: float,
    lon0: float,
) -> npt.NDArray[np.float32]:
    """Return a storm-relative (NY × NX) patch via nearest-neighbour lookup."""
    lat_rel = np.linspace(
        -_R_WINDOW_DEG + _DRES / 2,
        _R_WINDOW_DEG - _DRES / 2,
        _NY,
    ).astype("float32")
    lon_rel = np.linspace(
        -_R_WINDOW_DEG + _DRES / 2,
        _R_WINDOW_DEG - _DRES / 2,
        _NX,
    ).astype("float32")

    lon_vals = np.asarray(da.lon.values)
    if not np.all(np.diff(lon_vals) > 0):
        order = np.argsort(lon_vals)
        da = da.isel(lon=order)
        lon_vals = np.asarray(da.lon.values)

    lat_vals = np.asarray(da.lat.values)
    if np.all(np.diff(lat_vals) < 0):
        da = da.isel(lat=slice(None, None, -1))
        lat_vals = lat_vals[::-1]
    elif not np.all(np.diff(lat_vals) > 0):
        order = np.argsort(lat_vals)
        da = da.isel(lat=order)
        lat_vals = np.asarray(da.lat.values)

    target_lat = (lat0 + lat_rel).astype("float32")
    target_lon = (lon0 + lon_rel) % 360.0

    def _nearest(a_sorted: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
        pos = np.searchsorted(a_sorted, q)
        pos = np.clip(pos, 1, a_sorted.size - 1)
        prev = a_sorted[pos - 1]
        nxt = a_sorted[pos]
        choose_prev = (q - prev) <= (nxt - q)
        idx = pos.copy()
        idx[choose_prev] = pos[choose_prev] - 1
        return idx.astype(np.int64)

    ilat = _nearest(lat_vals, target_lat)
    ilon = _nearest(lon_vals, target_lon)

    arr2d = np.asarray(da.values, dtype=np.float32)
    return arr2d[ilat[:, None], ilon[None, :]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_cyclone_composites(
    *,
    year_start: int,
    year_end: int,
    hemisphere: str,
    intensity_min: int,
    intensity_max: int,
    track_path: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    output_directory: pathlib.Path,
    storm_lat: npt.NDArray,
    lat_band_half_width: float = constants.STORM_LAT_BAND_HALF_WIDTH,
) -> None:
    """Build cyclone-centred composites of meridionally integrated fluxes.

    Parameters
    ----------
    hemisphere : str
        ``"SH"`` or ``"NH"``.
    intensity_min, intensity_max : int
        Inclusive CVU intensity range.  Use ``(1, 5)`` for non-intense
        and ``(6, 99)`` for intense cyclones.
    track_path : pathlib.Path
        Track NetCDF file (e.g. ``TRACK_VO_anom_T42_ERA5_…_allSH.nc``).
    integrated_flux_directory : pathlib.Path
        Directory containing ``Integrated_Fluxes_YYYY_MM_.nc`` and
        ``New_Integrated_Fluxes_YYYY_MM_.nc`` files.
    storm_lat : array (12,)
        Monthly-mean storm-track latitude for the latitude-band filter.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    _LOG.info(
        "Building composites: hemisphere=%s intensity=[%s,%s] years=[%s,%s)",
        hemisphere,
        intensity_min,
        intensity_max,
        year_start,
        year_end,
    )

    # Load track data
    with netCDF4.Dataset(str(track_path)) as ds:
        time_index = ds["time"][:].astype(np.int64) - 1
        lat_trk = ds["latitude"][:].astype(np.float32)
        lon_trk = _to_360(ds["longitude"][:].astype(np.float32))
        inten_trk = ds["intensity"][:].astype(np.float32)

    # Build xarray Dataset for efficient filtering
    trk = xr.Dataset(
        data_vars={
            "lat": ("row", lat_trk),
            "lon": ("row", lon_trk),
            "intensity": ("row", inten_trk),
            "time_index": ("row", time_index),
        },
        coords={"row": np.arange(time_index.size, dtype=np.int64)},
    )

    year_from_idx = 1979 + (trk["time_index"].to_numpy() // _STEPS_PER_YEAR)
    trk = trk.assign(year=("row", year_from_idx.astype(np.int16)))
    trk = trk.where(
        (trk["year"] >= year_start) & (trk["year"] < year_end), drop=True
    )
    trk = trk.where(
        (trk["intensity"] >= intensity_min) & (trk["intensity"] <= intensity_max),
        drop=True,
    )

    if trk.sizes.get("row", 0) == 0:
        _LOG.info("No storms matched filters — nothing to composite")
        return

    # Flux variables from the two file types
    flux_vars_1 = [
        "F_TE_final", "tot_energy_final", "F_Shf_final",
        "F_Swabs_final", "F_Olr_final", "lat", "lon", "time",
    ]
    flux_vars_2 = [
        "F_u_mse_final", "F_v_mse_final", "F_Dhdt_final",
        "lat", "lon", "time",
    ]

    field_map = {
        "TE": ("F_TE_final", 1),
        "energy": ("tot_energy_final", 1),
        "Shf": ("F_Shf_final", 1),
        "Swabs": ("F_Swabs_final", 1),
        "Olr": ("F_Olr_final", 1),
        "u_mse": ("F_u_mse_final", 2),
        "v_mse": ("F_v_mse_final", 2),
        "Dhdt": ("F_Dhdt_final", 2),
    }

    cache1 = _DSCache(capacity=4)
    cache2 = _DSCache(capacity=4)

    lat_rel = np.linspace(
        -_R_WINDOW_DEG + _DRES / 2,
        _R_WINDOW_DEG - _DRES / 2,
        _NY,
    ).astype("float32")
    lon_rel = np.linspace(
        -_R_WINDOW_DEG + _DRES / 2,
        _R_WINDOW_DEG - _DRES / 2,
        _NX,
    ).astype("float32")

    comps = {k: np.zeros((12, _NY, _NX), dtype=np.float32) for k in field_map}
    counts = np.zeros(12, dtype=np.int32)

    center_header = [
        "row", "year", "month", "day", "hour6", "lat", "lon", "intensity",
    ] + ["%s_center" % nm for nm in field_map]
    center_rows = [tuple(center_header)]

    flux_fmt_1 = str(integrated_flux_directory / "Integrated_Fluxes_{yyyy:04d}_{mm:02d}_.nc")
    flux_fmt_2 = str(integrated_flux_directory / "New_Integrated_Fluxes_{yyyy:04d}_{mm:02d}_.nc")

    n_storms = trk.sizes["row"]
    for i in range(n_storms):
        lat0 = float(trk["lat"].isel(row=i))
        lon0 = float(trk["lon"].isel(row=i))
        step = int(trk["time_index"].isel(row=i))

        yy, mm, day0, hour6 = _day_hour_from_track_index(step_6h=step)
        if yy < year_start or yy >= year_end:
            continue
        midx = mm - 1

        # Monthly latitude-band gate
        target_lat = storm_lat[midx]
        if not (target_lat - lat_band_half_width <= lat0 <= target_lat + lat_band_half_width):
            continue

        mlen = constants.NOLEAP_MONTH_LENGTHS
        local_t = day0 * 4 + hour6

        path1 = flux_fmt_1.format(yyyy=yy, mm=mm)
        path2 = flux_fmt_2.format(yyyy=yy, mm=mm)

        try:
            ds1 = cache1.get(path=path1, needed_vars=flux_vars_1)
            ds2 = cache2.get(path=path2, needed_vars=flux_vars_2)
        except FileNotFoundError:
            continue

        if "F_TE_final" not in ds1 or local_t >= ds1["F_TE_final"].sizes["time"]:
            continue

        row_center = {"%s_center" % nm: np.nan for nm in field_map}
        for out_name, (var_name, which) in field_map.items():
            ds = ds1 if which == 1 else ds2
            if var_name not in ds or local_t >= ds[var_name].sizes["time"]:
                continue
            da = ds[var_name].isel(time=local_t)
            patch = _extract_patch(da, lat0, lon0)
            comps[out_name][midx, :, :] += patch
            row_center["%s_center" % out_name] = float(patch[_NY // 2, _NX // 2])

        counts[midx] += 1

        center_rows.append((
            int(i), int(yy), int(mm), int(day0 + 1), int(hour6 * 6),
            float(lat0), float(lon0),
            float(trk["intensity"].isel(row=i)),
            *[row_center["%s_center" % nm] for nm in field_map],
        ))

        if i % 5000 == 0 and i > 0:
            _LOG.info("  Processed %s / %s track snapshots", i, n_storms)

    cache1.close_all()
    cache2.close_all()

    # Normalise sums to means
    for k in comps:
        for m in range(12):
            if counts[m] > 0:
                comps[k][m, :, :] /= counts[m]
            else:
                comps[k][m, :, :] = np.nan

    # Determine output filename
    if intensity_min >= 6:
        tag = "Intense"
    elif intensity_max <= 5:
        tag = "Weak"
    else:
        tag = "All"
    out_nc = output_directory / ("Composites_%s_%s_noleap.nc" % (tag, hemisphere))
    out_csv = output_directory / ("Composites_%s_%s_noleap_center_samples.csv" % (tag, hemisphere))

    # Save NetCDF
    months = np.arange(1, 13, dtype=np.int16)
    ds_out = xr.Dataset(
        data_vars={
            "composite_%s" % k: (("month", "y", "x"), v.astype("float32"))
            for k, v in comps.items()
        },
        coords={
            "month": ("month", months),
            "y": ("y", lat_rel, {
                "units": "degrees",
                "long_name": "latitude offset from storm center",
            }),
            "x": ("x", lon_rel, {
                "units": "degrees",
                "long_name": "longitude offset from storm center",
            }),
            "count": ("month", counts, {
                "long_name": "storms per month (%s, %s, intensity %s–%s CVU)"
                % (hemisphere, tag, intensity_min, intensity_max),
            }),
        },
        attrs={
            "title": "Cyclone-centered composites (%s, %s, no-leap, lat-band filtered)"
            % (hemisphere, tag),
            "intensity_filter": "[%s, %s] CVU inclusive" % (intensity_min, intensity_max),
            "lat_band": "+/-%s deg around monthly storm-track latitude" % lat_band_half_width,
        },
    )
    ds_out.to_netcdf(str(out_nc), format="NETCDF4")
    ds_out.close()
    _LOG.info("Saved composites: %s", out_nc)

    # Save CSV
    with open(str(out_csv), "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(center_rows)
    _LOG.info("Saved center samples: %s", out_csv)
