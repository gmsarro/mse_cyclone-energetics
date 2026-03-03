from __future__ import annotations

"""Cyclone-centred monthly composites on a storm-relative grid.

Produces composites of both PW (meridionally integrated) and W/m² (local
ERA5) energy-budget fields for weak and intense cyclones.

The SHF residual is computed as:
    SHF = column_MSE - Swabs - OLR + dh/dt
"""

import collections
import csv
import logging
import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray as xr

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_R_WINDOW_DEG: float = 15.0
_DRES: float = 0.25
_NY: int = int(round(2 * _R_WINDOW_DEG / _DRES))
_NX: int = _NY

_STEPS_PER_YEAR: int = constants.TIMESTEPS_PER_YEAR
_LAT_BAND_HALF_WIDTH: float = 5.0

_GRAVITY: float = 9.81
_LV: float = 2.501e6
_GEOPOTENTIAL_LEVEL: int = 30
_Q_LEVEL: int = 35
_VO_GRID_SIZE: int = 20

_VINT_NAME_MAPS: typing.List[typing.Dict[str, str]] = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]

_PW_FIELD_MAP: typing.Dict[str, typing.Tuple[str, int]] = {
    "TE": ("F_TE_final", 1),
    "energy": ("tot_energy_final", 1),
    "Shf": ("F_Shf_final", 1),
    "Swabs": ("F_Swabs_final", 1),
    "Olr": ("F_Olr_final", 1),
    "u_mse": ("F_u_mse_final", 2),
    "v_mse": ("F_v_mse_final", 2),
    "Dhdt": ("F_Dhdt_final", 2),
}

_WM2_FIELD_NAMES: typing.List[str] = [
    "energy_wm", "Dhdt_wm", "Swabs_wm", "Olr_wm", "Shf_wm",
    "Z", "T", "Q", "VO",
]


def _to_360(lon: npt.NDArray) -> npt.NDArray:
    return (lon % 360 + 360) % 360


def _nearest_idx(
    *,
    a_sorted: npt.NDArray,
    q: npt.NDArray,
) -> npt.NDArray:
    pos = np.searchsorted(a_sorted, q)
    pos = np.clip(pos, 1, a_sorted.size - 1)
    prev = a_sorted[pos - 1]
    nxt = a_sorted[pos]
    choose_prev = (q - prev) <= (nxt - q)
    idx = pos.copy()
    idx[choose_prev] = pos[choose_prev] - 1
    return idx.astype(np.int64)


def _sort_coords(
    *,
    lat: npt.NDArray,
    lon: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray, bool, npt.NDArray]:
    lat_flip = False
    if lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1]
        lat_flip = True
    lon = _to_360(lon)
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    return lat, lon, lat_flip, lon_order


def _resolve_vint_names(
    *,
    ds: netCDF4.Dataset,
) -> typing.Dict[str, str]:
    for mapping in _VINT_NAME_MAPS:
        if all(mapping[k] in ds.variables for k in ("vigd", "vimdf", "vithed")):
            return mapping
    raise KeyError("Cannot resolve vint variable names: %s" % list(ds.variables))


def _extract_patch_np(
    *,
    arr2d: npt.NDArray,
    lat_vals: npt.NDArray,
    lon_vals: npt.NDArray,
    lat0: float,
    lon0: float,
    lat_offset: npt.NDArray,
    lon_offset: npt.NDArray,
) -> npt.NDArray[np.float32]:
    target_lat = (lat0 + lat_offset).astype("float32")
    target_lon = (lon0 + lon_offset) % 360.0
    ilat = _nearest_idx(a_sorted=lat_vals, q=target_lat)
    ilon = _nearest_idx(a_sorted=lon_vals, q=target_lon)
    return arr2d[ilat[:, None], ilon[None, :]].astype(np.float32)


def compute_stormtrack_latitudes(
    *,
    flux_assignment_path: pathlib.Path,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    _LOG.info("Computing storm-track latitudes from %s", flux_assignment_path)
    with netCDF4.Dataset(str(flux_assignment_path)) as ds:
        lat = np.array(ds["lat"][:])
        fte_total = np.array(ds["F_TE_final"][0, :, :, :])

    fte_zonal = np.mean(fte_total, axis=2)
    nlat = len(lat)
    x_d = np.linspace(0, nlat - 1, nlat)
    y_d = np.linspace(0, 12, 12)
    x3_d = np.linspace(0, nlat - 1, 25600)

    lat_fine = scipy.interpolate.interp1d(x_d, lat)(x3_d)
    fte_interp = scipy.interpolate.RectBivariateSpline(y_d, x_d, fte_zonal)(y_d, x3_d)

    stormtrack_lat_nh = lat_fine[np.argmax(fte_interp, axis=1)]
    stormtrack_lat_sh = lat_fine[np.argmin(fte_interp, axis=1)]

    _LOG.info("  NH: %s", np.array2string(stormtrack_lat_nh, precision=4))
    _LOG.info("  SH: %s", np.array2string(stormtrack_lat_sh, precision=4))
    return stormtrack_lat_nh, stormtrack_lat_sh


class _MonthlyData:
    """Preload all needed 2D fields for a given (year, month) into numpy."""

    def __init__(
        self,
        *,
        yy: int,
        mm: int,
        integrated_flux_directory: pathlib.Path,
        vint_directory: pathlib.Path,
        dhdt_directory: pathlib.Path,
        radiation_directory: pathlib.Path,
        z_directory: pathlib.Path,
        t2m_directory: pathlib.Path,
        q_directory: pathlib.Path,
        vorticity_directory: typing.Optional[pathlib.Path],
    ) -> None:
        self.yy = yy
        self.mm = mm
        self.nt = constants.NOLEAP_MONTH_LENGTHS[mm - 1] * 4

        self.pw1: typing.Dict[str, npt.NDArray] = {}
        self.pw2: typing.Dict[str, npt.NDArray] = {}
        self.pw_lat: typing.Optional[npt.NDArray] = None
        self.pw_lon: typing.Optional[npt.NDArray] = None

        self.energy_wm: typing.Optional[npt.NDArray] = None
        self.dhdt_wm: typing.Optional[npt.NDArray] = None
        self.swabs_wm: typing.Optional[npt.NDArray] = None
        self.olr_wm: typing.Optional[npt.NDArray] = None
        self.z_wm: typing.Optional[npt.NDArray] = None
        self.t_wm: typing.Optional[npt.NDArray] = None
        self.q_wm: typing.Optional[npt.NDArray] = None
        self.wm_lat: typing.Optional[npt.NDArray] = None
        self.wm_lon: typing.Optional[npt.NDArray] = None

        self.vo: typing.Optional[npt.NDArray] = None
        self.vo_lat: typing.Optional[npt.NDArray] = None
        self.vo_lon: typing.Optional[npt.NDArray] = None

        self._integrated_flux_directory = integrated_flux_directory
        self._vint_directory = vint_directory
        self._dhdt_directory = dhdt_directory
        self._radiation_directory = radiation_directory
        self._z_directory = z_directory
        self._t2m_directory = t2m_directory
        self._q_directory = q_directory
        self._vorticity_directory = vorticity_directory

        self._load_pw()
        self._load_wm2()

    def _load_pw(self) -> None:
        path1 = self._integrated_flux_directory / (
            "Integrated_Fluxes_%d_%02d_.nc" % (self.yy, self.mm)
        )
        path2 = self._integrated_flux_directory / (
            "New_Integrated_Fluxes_%d_%02d_.nc" % (self.yy, self.mm)
        )

        if not path1.exists() or not path2.exists():
            _LOG.warning("  PW flux files missing for %d/%02d", self.yy, self.mm)
            return

        with netCDF4.Dataset(str(path1)) as ds:
            lat = np.asarray(ds["lat"][:], dtype=np.float32)
            lon = _to_360(np.asarray(ds["lon"][:], dtype=np.float32))
            lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
            self.pw_lat = lat
            self.pw_lon = lon
            for vname in ["F_TE_final", "tot_energy_final", "F_Shf_final",
                          "F_Swabs_final", "F_Olr_final"]:
                if vname in ds.variables:
                    arr = np.asarray(ds[vname][:self.nt], dtype=np.float32)
                    if lat_flip:
                        arr = arr[:, ::-1, :]
                    arr = arr[:, :, lon_order]
                    self.pw1[vname] = arr

        with netCDF4.Dataset(str(path2)) as ds:
            for vname in ["F_u_mse_final", "F_v_mse_final", "F_Dhdt_final"]:
                if vname in ds.variables:
                    arr = np.asarray(ds[vname][:self.nt], dtype=np.float32)
                    if lat_flip:
                        arr = arr[:, ::-1, :]
                    arr = arr[:, :, lon_order]
                    self.pw2[vname] = arr

        _LOG.info("    PW loaded (%d vars)", len(self.pw1) + len(self.pw2))

    def _load_wm2(self) -> None:
        loaded = []

        vint_path = self._vint_directory / (
            "era5_vint_%d_%02d_filtered.nc" % (self.yy, self.mm)
        )
        if vint_path.exists():
            try:
                with netCDF4.Dataset(str(vint_path)) as ds:
                    vn = _resolve_vint_names(ds=ds)
                    lat = np.asarray(ds["latitude"][:], dtype=np.float32)
                    lon = _to_360(np.asarray(ds["longitude"][:], dtype=np.float32))

                    vigd = np.asarray(ds[vn["vigd"]][:self.nt], dtype=np.float32)
                    vimdf = np.asarray(ds[vn["vimdf"]][:self.nt], dtype=np.float32)
                    vithed = np.asarray(ds[vn["vithed"]][:self.nt], dtype=np.float32)

                    vigd = vigd[:, ::-1, :]
                    vimdf = vimdf[:, ::-1, :]
                    vithed = vithed[:, ::-1, :]

                    lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
                    if lat_flip:
                        vigd = vigd[:, ::-1, :]
                        vimdf = vimdf[:, ::-1, :]
                        vithed = vithed[:, ::-1, :]
                    vigd = vigd[:, :, lon_order]
                    vimdf = vimdf[:, :, lon_order]
                    vithed = vithed[:, :, lon_order]
                    vigd = vigd[:, :, ::-1]
                    vimdf = vimdf[:, :, ::-1]
                    vithed = vithed[:, :, ::-1]

                    self.wm_lat = lat
                    self.wm_lon = lon
                    self.energy_wm = vigd + vimdf * _LV + vithed
                    loaded.append("energy_wm")
            except (OSError, KeyError) as e:
                _LOG.debug("  Skipping energy_wm for %d/%02d: %s", self.yy, self.mm, e)

        dhdt_path = self._dhdt_directory / (
            "tend_%d_%02d_filtered_2.nc" % (self.yy, self.mm)
        )
        if dhdt_path.exists():
            try:
                with netCDF4.Dataset(str(dhdt_path)) as ds:
                    arr = np.asarray(ds["tend_filtered"][:self.nt], dtype=np.float32)
                    lat = np.asarray(ds["latitude"][:], dtype=np.float32)
                    lon = _to_360(np.asarray(ds["longitude"][:], dtype=np.float32))

                    arr = arr[:, ::-1, :]

                    lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
                    if lat_flip:
                        arr = arr[:, ::-1, :]
                    arr = arr[:, :, lon_order]
                    arr = arr[:, :, ::-1]
                    self.dhdt_wm = arr
                    if self.wm_lat is None:
                        self.wm_lat = lat
                        self.wm_lon = lon
                    loaded.append("dhdt_wm")
            except OSError as e:
                _LOG.debug("  Skipping dhdt_wm for %d/%02d: %s", self.yy, self.mm, e)

        rad_path = self._radiation_directory / (
            "era5_rad_%d_%02d.6hrly.nc" % (self.yy, self.mm)
        )
        if rad_path.exists():
            try:
                with netCDF4.Dataset(str(rad_path)) as ds:
                    time_var = "valid_time" if "valid_time" in ds.variables else "time"
                    nt_rad = min(self.nt, ds[time_var].shape[0])
                    tsr = np.asarray(ds["tsr"][:nt_rad], dtype=np.float32)
                    ssr = np.asarray(ds["ssr"][:nt_rad], dtype=np.float32)
                    ttr = np.asarray(ds["ttr"][:nt_rad], dtype=np.float32)
                    lat = np.asarray(ds["latitude"][:], dtype=np.float32)
                    lon = _to_360(np.asarray(ds["longitude"][:], dtype=np.float32))
                    lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
                    if lat_flip:
                        tsr = tsr[:, ::-1, :]
                        ssr = ssr[:, ::-1, :]
                        ttr = ttr[:, ::-1, :]
                    tsr = tsr[:, :, lon_order]
                    ssr = ssr[:, :, lon_order]
                    ttr = ttr[:, :, lon_order]
                    self.swabs_wm = np.nan_to_num((tsr - ssr) / 3600.0, nan=0.0)
                    self.olr_wm = np.nan_to_num(ttr / 3600.0, nan=0.0)
                    loaded.append("swabs+olr")
            except OSError as e:
                _LOG.debug("  Skipping radiation for %d/%02d: %s", self.yy, self.mm, e)

        z_path = self._z_directory / ("era5_z_%d_%02d.6hrly.nc" % (self.yy, self.mm))
        if z_path.exists():
            try:
                with netCDF4.Dataset(str(z_path)) as ds:
                    lat = np.asarray(ds["latitude"][:], dtype=np.float32)
                    lon = _to_360(np.asarray(ds["longitude"][:], dtype=np.float32))
                    lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
                    arr = np.asarray(
                        ds["z"][:self.nt, _GEOPOTENTIAL_LEVEL, :, :],
                        dtype=np.float32
                    ) / _GRAVITY
                    if lat_flip:
                        arr = arr[:, ::-1, :]
                    arr = arr[:, :, lon_order]
                    self.z_wm = arr
                    if self.wm_lat is None:
                        self.wm_lat = lat
                        self.wm_lon = lon
                    loaded.append("Z")
            except OSError as e:
                _LOG.debug("  Skipping Z for %d/%02d: %s", self.yy, self.mm, e)

        t2m_path = self._t2m_directory / (
            "era5_t2m_%d_%02d.6hrly.nc" % (self.yy, self.mm)
        )
        if t2m_path.exists():
            try:
                with netCDF4.Dataset(str(t2m_path)) as ds:
                    lat = np.asarray(ds["latitude"][:], dtype=np.float32)
                    lon = _to_360(np.asarray(ds["longitude"][:], dtype=np.float32))
                    lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
                    arr = np.asarray(ds["t2m"][:self.nt], dtype=np.float32)
                    if lat_flip:
                        arr = arr[:, ::-1, :]
                    arr = arr[:, :, lon_order]
                    self.t_wm = arr
                    loaded.append("T")
            except OSError as e:
                _LOG.debug("  Skipping T for %d/%02d: %s", self.yy, self.mm, e)

        q_path = self._q_directory / ("era5_q_%d_%02d.6hrly.nc" % (self.yy, self.mm))
        if q_path.exists():
            try:
                with netCDF4.Dataset(str(q_path)) as ds:
                    lat = np.asarray(ds["latitude"][:], dtype=np.float32)
                    lon = _to_360(np.asarray(ds["longitude"][:], dtype=np.float32))
                    lat, lon, lat_flip, lon_order = _sort_coords(lat=lat, lon=lon)
                    arr = np.asarray(
                        ds["q"][:self.nt, _Q_LEVEL, :, :],
                        dtype=np.float32
                    )
                    if lat_flip:
                        arr = arr[:, ::-1, :]
                    arr = arr[:, :, lon_order]
                    self.q_wm = arr
                    loaded.append("Q")
            except OSError as e:
                _LOG.debug("  Skipping Q for %d/%02d: %s", self.yy, self.mm, e)

        if self._vorticity_directory is not None:
            vo_path = self._vorticity_directory / (
                "VO850_%d.nc" % self.yy
            )
            if vo_path.exists():
                try:
                    with netCDF4.Dataset(str(vo_path)) as ds:
                        self.vo_lat = np.asarray(ds["lat"][:], dtype=np.float32)
                        self.vo_lon = _to_360(
                            np.asarray(ds["lon"][:], dtype=np.float32)
                        )
                        if self.vo_lat[0] > self.vo_lat[-1]:
                            self.vo_lat = self.vo_lat[::-1]
                        vo_lon_order = np.argsort(self.vo_lon)
                        self.vo_lon = self.vo_lon[vo_lon_order]
                        month_start = constants.NOLEAP_MONTH_CUMULATIVE[self.mm - 1] * 4
                        month_end = month_start + self.nt
                        arr = np.asarray(
                            ds["VO"][month_start:month_end], dtype=np.float32
                        )
                        if ds["lat"][0] > ds["lat"][-1]:
                            arr = arr[:, ::-1, :]
                        arr = arr[:, :, vo_lon_order]
                        self.vo = arr
                        loaded.append("VO")
                except OSError as e:
                    _LOG.debug("  Skipping VO for %d/%02d: %s", self.yy, self.mm, e)

        _LOG.info("    W/m2 loaded [%s]", ", ".join(loaded) if loaded else "none")


def build_cyclone_composites(
    *,
    year_start: int,
    year_end: int,
    hemisphere: str,
    intensity_min: int,
    intensity_max: int,
    track_path: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    dhdt_directory: pathlib.Path,
    radiation_directory: pathlib.Path,
    z_directory: pathlib.Path,
    t2m_directory: pathlib.Path,
    q_directory: pathlib.Path,
    output_directory: pathlib.Path,
    storm_lat: npt.NDArray,
    vorticity_directory: typing.Optional[pathlib.Path] = None,
    lat_band_half_width: float = _LAT_BAND_HALF_WIDTH,
) -> None:
    """Build cyclone-centred composites of both PW and W/m² budget terms.

    Parameters
    ----------
    hemisphere : str
        ``"SH"`` or ``"NH"``.
    intensity_min, intensity_max : int
        Inclusive CVU intensity range. Use ``(1, 5)`` for weak and
        ``(6, 99)`` for intense cyclones.
    track_path : pathlib.Path
        Track NetCDF file.
    storm_lat : array (12,)
        Monthly-mean storm-track latitude for the latitude-band filter.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    tag = "Weak" if intensity_max <= 5 else "Intense"
    _LOG.info(
        "Building composites: hemisphere=%s %s intensity=[%s,%s] years=[%s,%s)",
        hemisphere,
        tag,
        intensity_min,
        intensity_max,
        year_start,
        year_end,
    )

    with netCDF4.Dataset(str(track_path)) as ds:
        time_index = np.array(ds["time"][:]).astype(np.int64) - 1
        lat_trk = np.array(ds["latitude"][:]).astype(np.float32)
        lon_trk = _to_360(np.array(ds["longitude"][:]).astype(np.float32))
        inten_trk = np.array(ds["intensity"][:]).astype(np.float32)

    year_from_idx = (constants.ERA5_BASE_YEAR + time_index // _STEPS_PER_YEAR).astype(np.int16)
    mask = (
        (year_from_idx >= year_start) & (year_from_idx < year_end)
        & (inten_trk >= intensity_min) & (inten_trk <= intensity_max)
    )
    idx_sel = np.where(mask)[0]
    lat_arr = lat_trk[idx_sel]
    lon_arr = lon_trk[idx_sel]
    inten_arr = inten_trk[idx_sel]
    step_arr = time_index[idx_sel]

    n_storms = len(idx_sel)
    if n_storms == 0:
        _LOG.warning("No storms matched -- skipping")
        return
    _LOG.info("  %d track snapshots after year/intensity filter", n_storms)

    year_arr = (constants.ERA5_BASE_YEAR + step_arr // _STEPS_PER_YEAR).astype(np.int16)
    doy_arr = (step_arr % _STEPS_PER_YEAR) // 4
    hour6_arr = (step_arr % 4).astype(np.int8)
    mcum = constants.NOLEAP_MONTH_CUMULATIVE
    month_arr = np.searchsorted(mcum[1:], doy_arr, side="right").astype(np.int8) + 1
    day_arr = (doy_arr - mcum[month_arr - 1]).astype(np.int16)
    local_t_arr = (day_arr * 4 + hour6_arr).astype(np.int16)

    midx_arr = month_arr - 1
    target_lat_arr = storm_lat[midx_arr]
    in_band = np.abs(lat_arr - target_lat_arr) <= lat_band_half_width
    band_idx = np.where(in_band)[0]
    _LOG.info(
        "  %d snapshots pass lat-band filter (%.1f%%)",
        len(band_idx),
        100.0 * len(band_idx) / n_storms if n_storms else 0
    )

    groups: typing.Dict[typing.Tuple[int, int], typing.List[int]] = (
        collections.defaultdict(list)
    )
    for bi in band_idx:
        groups[(int(year_arr[bi]), int(month_arr[bi]))].append(bi)

    sorted_keys = sorted(groups.keys())
    _LOG.info("  %d unique (year, month) groups", len(sorted_keys))

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

    pw_comps = {k: np.zeros((12, _NY, _NX), dtype=np.float64) for k in _PW_FIELD_MAP}
    wm2_comps = {
        k: np.zeros((12, _NY, _NX), dtype=np.float64) for k in _WM2_FIELD_NAMES if k != "VO"
    }
    wm2_comps["VO"] = np.zeros((12, _VO_GRID_SIZE, _VO_GRID_SIZE), dtype=np.float64)
    counts = np.zeros(12, dtype=np.int32)

    center_header = (
        ["row", "year", "month", "day", "hour6", "lat", "lon", "intensity"]
        + ["%s_center" % nm for nm in _PW_FIELD_MAP]
    )
    center_rows: typing.List[typing.Tuple[typing.Any, ...]] = [tuple(center_header)]

    vo_half = _VO_GRID_SIZE * 1.5 / 2
    vo_lat_rel = np.linspace(
        -vo_half + 0.75, vo_half - 0.75, _VO_GRID_SIZE
    ).astype("float32")
    vo_lon_rel = np.linspace(
        -vo_half + 0.75, vo_half - 0.75, _VO_GRID_SIZE
    ).astype("float32")

    total_accepted = 0

    for gi, (yy, mm) in enumerate(sorted_keys):
        midx = mm - 1
        snap_indices = groups[(yy, mm)]
        _LOG.info(
            "  [%d/%d] %d-%02d: %d snapshots",
            gi + 1, len(sorted_keys), yy, mm, len(snap_indices)
        )

        mdata = _MonthlyData(
            yy=yy,
            mm=mm,
            integrated_flux_directory=integrated_flux_directory,
            vint_directory=vint_directory,
            dhdt_directory=dhdt_directory,
            radiation_directory=radiation_directory,
            z_directory=z_directory,
            t2m_directory=t2m_directory,
            q_directory=q_directory,
            vorticity_directory=vorticity_directory,
        )

        for si in snap_indices:
            lat0 = float(lat_arr[si])
            lon0 = float(lon_arr[si])
            lt = int(local_t_arr[si])

            pw_ok = False
            row_center = {"%s_center" % nm: np.nan for nm in _PW_FIELD_MAP}
            for out_name, (var_name, which) in _PW_FIELD_MAP.items():
                src = mdata.pw1 if which == 1 else mdata.pw2
                if var_name not in src:
                    continue
                arr3d = src[var_name]
                if lt >= arr3d.shape[0]:
                    continue
                patch = _extract_patch_np(
                    arr2d=arr3d[lt],
                    lat_vals=mdata.pw_lat,
                    lon_vals=mdata.pw_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )
                pw_comps[out_name][midx] += patch
                row_center["%s_center" % out_name] = float(patch[_NY // 2, _NX // 2])
                pw_ok = True

            if not pw_ok:
                continue

            counts[midx] += 1
            total_accepted += 1

            center_rows.append((
                int(si), int(yy), int(mm), int(day_arr[si] + 1), int(hour6_arr[si] * 6),
                float(lat0), float(lon0), float(inten_arr[si]),
                *[row_center["%s_center" % nm] for nm in _PW_FIELD_MAP],
            ))

            energy_patch = None
            if mdata.energy_wm is not None and lt < mdata.energy_wm.shape[0]:
                energy_patch = _extract_patch_np(
                    arr2d=mdata.energy_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )
                wm2_comps["energy_wm"][midx] += energy_patch

            dhdt_patch = None
            if mdata.dhdt_wm is not None and lt < mdata.dhdt_wm.shape[0]:
                dhdt_patch = _extract_patch_np(
                    arr2d=mdata.dhdt_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )
                wm2_comps["Dhdt_wm"][midx] += dhdt_patch

            swabs_patch = None
            olr_patch = None
            if mdata.swabs_wm is not None and lt < mdata.swabs_wm.shape[0]:
                swabs_patch = _extract_patch_np(
                    arr2d=mdata.swabs_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )
                wm2_comps["Swabs_wm"][midx] += swabs_patch
            if mdata.olr_wm is not None and lt < mdata.olr_wm.shape[0]:
                olr_patch = _extract_patch_np(
                    arr2d=mdata.olr_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )
                wm2_comps["Olr_wm"][midx] += olr_patch

            if all(x is not None for x in [energy_patch, swabs_patch, olr_patch, dhdt_patch]):
                wm2_comps["Shf_wm"][midx] += (
                    energy_patch - swabs_patch - olr_patch + dhdt_patch
                )

            if mdata.z_wm is not None and lt < mdata.z_wm.shape[0]:
                wm2_comps["Z"][midx] += _extract_patch_np(
                    arr2d=mdata.z_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )

            if mdata.t_wm is not None and lt < mdata.t_wm.shape[0]:
                wm2_comps["T"][midx] += _extract_patch_np(
                    arr2d=mdata.t_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )

            if mdata.q_wm is not None and lt < mdata.q_wm.shape[0]:
                wm2_comps["Q"][midx] += _extract_patch_np(
                    arr2d=mdata.q_wm[lt],
                    lat_vals=mdata.wm_lat,
                    lon_vals=mdata.wm_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=lat_rel,
                    lon_offset=lon_rel,
                )

            if mdata.vo is not None and lt < mdata.vo.shape[0]:
                vo_patch = _extract_patch_np(
                    arr2d=mdata.vo[lt],
                    lat_vals=mdata.vo_lat,
                    lon_vals=mdata.vo_lon,
                    lat0=lat0,
                    lon0=lon0,
                    lat_offset=vo_lat_rel,
                    lon_offset=vo_lon_rel,
                )
                wm2_comps["VO"][midx] += vo_patch

        del mdata
        _LOG.info("    Done (total accepted so far: %d)", total_accepted)

    _LOG.info("  Total accepted: %d", total_accepted)

    for k in pw_comps:
        for m in range(12):
            if counts[m] > 0:
                pw_comps[k][m] /= counts[m]
            else:
                pw_comps[k][m] = np.nan

    for k in wm2_comps:
        for m in range(12):
            if counts[m] > 0:
                wm2_comps[k][m] /= counts[m]
            else:
                wm2_comps[k][m] = np.nan

    x_vo = np.linspace(0, _VO_GRID_SIZE, _VO_GRID_SIZE)
    y_interp = np.linspace(0, _VO_GRID_SIZE, _NY)
    vo_interp = np.full((12, _NY, _NX), np.nan, dtype=np.float64)
    for m in range(12):
        if counts[m] > 0:
            spl = scipy.interpolate.RectBivariateSpline(
                x_vo, x_vo, wm2_comps["VO"][m], kx=1, ky=1
            )
            vo_interp[m] = spl(y_interp, y_interp)
    wm2_comps["VO"] = vo_interp

    out_nc = output_directory / ("Composites_%s_%s_noleap.nc" % (tag, hemisphere))
    out_csv = output_directory / (
        "Composites_%s_%s_noleap_center_samples.csv" % (tag, hemisphere)
    )

    data_vars = {}
    for k, v in pw_comps.items():
        data_vars["composite_%s" % k] = (("month", "y", "x"), v.astype("float32"))
    for k, v in wm2_comps.items():
        data_vars["composite_%s" % k] = (("month", "y", "x"), v.astype("float32"))

    months = np.arange(1, 13, dtype=np.int16)
    ds_out = xr.Dataset(
        data_vars=data_vars,
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
                "long_name": "storms per month (%s, %s, intensity %s-%s CVU)"
                % (hemisphere, tag, intensity_min, intensity_max),
            }),
        },
        attrs={
            "title": "Cyclone-centered composites (%s, %s, no-leap, lat-band filtered)"
            % (hemisphere, tag),
            "hemisphere": hemisphere,
            "intensity_filter": "[%s, %s] CVU inclusive" % (intensity_min, intensity_max),
            "lat_band": "+/-%s deg around monthly storm-track latitude" % lat_band_half_width,
            "storm_track_latitudes": np.array2string(storm_lat, precision=6, separator=", "),
            "year_range": "[%s, %s)" % (year_start, year_end),
        },
    )
    ds_out.to_netcdf(str(out_nc), format="NETCDF4")
    ds_out.close()
    _LOG.info("Saved composites: %s", out_nc)

    with open(str(out_csv), "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(center_rows)
    _LOG.info("Saved center samples: %s", out_csv)
