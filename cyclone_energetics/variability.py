from __future__ import annotations

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_FINE_GRID_FACTOR: int = 36

_TRACK_HALF_WIDTH_DEG: float = 10.0

_PW_FACTOR: float = 2 * np.pi * constants.EARTH_RADIUS / 1e15

_SMOOTH_WINDOW: int = 3

_LAND_REGIONS: typing.List[typing.Tuple[float, float, float, float]] = [
    (32.5, 90.0, 0.0, 112.5),
    (57.5, 90.0, 112.5, 337.5),
    (32.5, 57.5, 112.5, 137.5),
    (42.5, 57.5, 112.5, 122.5),
    (32.5, 47.5, 122.5, 137.5),
    (37.5, 62.5, 237.5, 285.0),
    (47.5, 87.5, 237.5, 300.0),
]
_OCEAN_OVERRIDES: typing.List[typing.Tuple[float, float, float, float]] = [
    (80.0, 90.0, 0.0, 360.0),
    (70.0, 90.0, 55.0, 95.0),
    (50.0, 67.5, 122.5, 137.5),
    (45.0, 52.5, 0.0, 22.5),
]
_MASK_LAT_SOUTH: float = 0.0
_MASK_LAT_NORTH: float = 57.5


def _interp_lat_2d(
    field: npt.NDArray,
    *,
    n_fine: int,
) -> npt.NDArray:
    n_months, n_lat = field.shape
    x_d = np.linspace(0, n_lat - 1, n_lat)
    y_d = np.linspace(0, n_months, n_months)
    x_fine = np.linspace(0, n_lat - 1, n_fine)
    y_fine = np.linspace(0, n_months, n_months)
    spline = scipy.interpolate.RectBivariateSpline(y_d, x_d, field)
    return spline(y_fine, x_fine)


def _fine_lat(
    latitude: npt.NDArray,
    *,
    n_fine: int,
) -> npt.NDArray:
    n_lat = latitude.shape[0]
    x_d = np.linspace(0, n_lat - 1, n_lat)
    x_fine = np.linspace(0, n_lat - 1, n_fine)
    return scipy.interpolate.interp1d(x_d, latitude)(x_fine)


def _half_win_from_lat(
    lat_fine: npt.NDArray,
    *,
    half_width_deg: float = _TRACK_HALF_WIDTH_DEG,
) -> int:
    dlat = float(np.abs(lat_fine[1] - lat_fine[0]))
    return max(1, int(round(half_width_deg / dlat)))


def _interp_mask_field(
    field: npt.NDArray,
    *,
    target_shape: typing.Tuple[int, int],
) -> npt.NDArray:
    in_ny, in_nx = field.shape
    out_ny, out_nx = target_shape
    y_in = np.linspace(0, in_ny - 1, in_ny)
    x_in = np.linspace(0, in_nx - 1, in_nx)
    y_out = np.linspace(0, in_ny - 1, out_ny)
    x_out = np.linspace(0, in_nx - 1, out_nx)
    spline = scipy.interpolate.RectBivariateSpline(y_in, x_in, field, kx=1, ky=1)
    return spline(y_out, x_out)


def _stormtrack_from_total_fte(
    fte_zon_int: npt.NDArray,
    *,
    lat_fine: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    st_nh = np.argmax(fte_zon_int, axis=1)
    st_sh = np.argmin(fte_zon_int, axis=1)
    return st_nh, st_sh, lat_fine[st_nh], lat_fine[st_sh]


def _mean_around_track(
    field: npt.NDArray,
    *,
    idx: npt.NDArray,
    half_win: int,
) -> npt.NDArray:
    n_months = field.shape[0]
    n_fine = field.shape[1]
    out = np.zeros(n_months)
    for n in range(n_months):
        i0 = max(0, idx[n] - half_win)
        i1 = min(n_fine, idx[n] + half_win)
        out[n] = np.mean(field[n, i0:i1])
    return out


def _running_mean(
    data: npt.NDArray,
    *,
    window_size: int = _SMOOTH_WINDOW,
) -> npt.NDArray:
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def _seasonal_diff(x: npt.NDArray) -> float:
    x = np.asarray(x)
    djf_idx = [11, 0, 1]
    jja_idx = [5, 6, 7]
    return float(np.mean(x[djf_idx]) - np.mean(x[jja_idx]))


def _band_from_lines(
    arr: npt.NDArray,
) -> typing.Tuple[float, npt.NDArray, npt.NDArray]:
    std_per_month = np.std(arr, axis=0)
    mean_std = np.mean(std_per_month, axis=1)
    return float(np.max(mean_std)), mean_std, std_per_month


def _nearest_idx(
    arr: npt.NDArray,
    *,
    val: float,
) -> int:
    return int(np.argmin(np.abs(arr - val)))


def _build_land_ocean_masks(
    *,
    latitude: npt.NDArray,
    longitude: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    n_lat = latitude.shape[0]
    n_lon = longitude.shape[0]
    lon360 = longitude % 360

    lat_s_idx = _nearest_idx(latitude, val=_MASK_LAT_SOUTH)
    lat_n_idx = _nearest_idx(latitude, val=_MASK_LAT_NORTH)
    lat_lo = min(lat_s_idx, lat_n_idx)
    lat_hi = max(lat_s_idx, lat_n_idx) + 1

    land_mask = np.zeros((n_lat, n_lon))

    for (lat_south, lat_north, lon_west, lon_east) in _LAND_REGIONS:
        i_s = _nearest_idx(latitude, val=lat_south)
        i_n = _nearest_idx(latitude, val=lat_north)
        j_w = _nearest_idx(lon360, val=lon_west)
        j_e = _nearest_idx(lon360, val=lon_east)
        r_lo, r_hi = min(i_s, i_n), max(i_s, i_n) + 1
        c_lo, c_hi = min(j_w, j_e), max(j_w, j_e) + 1
        land_mask[r_lo:r_hi, c_lo:c_hi] = 1.0

    for (lat_south, lat_north, lon_west, lon_east) in _OCEAN_OVERRIDES:
        i_s = _nearest_idx(latitude, val=lat_south)
        i_n = _nearest_idx(latitude, val=lat_north)
        j_w = _nearest_idx(lon360, val=lon_west)
        j_e = _nearest_idx(lon360, val=lon_east)
        r_lo, r_hi = min(i_s, i_n), max(i_s, i_n) + 1
        c_lo, c_hi = min(j_w, j_e), max(j_w, j_e) + 1
        land_mask[r_lo:r_hi, c_lo:c_hi] = 0.0

    if latitude[0] > latitude[-1]:
        land_mask[:lat_lo, :] = 0.0
        land_mask[lat_hi:, :] = 0.0
    else:
        land_mask[:lat_lo, :] = 0.0
        land_mask[lat_hi:, :] = 0.0

    ocean_mask = 1.0 - land_mask
    if latitude[0] > latitude[-1]:
        ocean_mask[:lat_lo, :] = 0.0
        ocean_mask[lat_hi:, :] = 0.0
    else:
        ocean_mask[:lat_lo, :] = 0.0
        ocean_mask[lat_hi:, :] = 0.0

    return ocean_mask, land_mask


def _load_yearly_fluxes(
    *,
    yearly_files: typing.List[pathlib.Path],
    year_idx: int,
    years_per_file: int,
) -> typing.Dict[str, npt.NDArray]:
    stage = year_idx // years_per_file
    yr_in_file = year_idx % years_per_file

    if stage >= len(yearly_files):
        raise IndexError(
            "Year index %d requires file index %d, "
            "but only %d files available"
            % (year_idx, stage, len(yearly_files))
        )

    result: typing.Dict[str, npt.NDArray] = {}
    with netCDF4.Dataset(str(yearly_files[stage])) as ds:
        for icut in (0, 5):
            for suffix in ("", "_cycl", "_ant"):
                vn_te = "F_TE_final%s" % suffix
                if vn_te in ds.variables:
                    arr = ds[vn_te][icut, :, yr_in_file, :, :]
                    result["F_TE%s_%d" % (suffix, icut)] = np.mean(arr, axis=-1)

                for vn_base in ("F_Swabs_final", "F_Olr_final", "F_Dhdt_final",
                                "tot_energy_final", "F_TE_z_final", "F_UM_z_final"):
                    vn = "%s%s" % (vn_base, suffix)
                    if vn in ds.variables:
                        arr = ds[vn][icut, :, yr_in_file, :, :]
                        key = "%s%s_%d" % (vn_base.replace("_final", ""), suffix, icut)
                        result[key] = np.mean(arr, axis=-1)

            for suffix in ("_cycl", "_ant"):
                for vn_base in ("F_TE_final", "F_Swabs_final", "F_Olr_final",
                                "F_Dhdt_final", "tot_energy_final",
                                "F_TE_z_final", "F_UM_z_final"):
                    vn = "%s%s" % (vn_base, suffix)
                    if vn in ds.variables:
                        key = "2d_%s%s_%d" % (vn_base.replace("_final", ""), suffix, icut)
                        result[key] = np.asarray(ds[vn][icut, :, yr_in_file, :, :])

    return result


def _sort_to_ascending(
    lat: npt.NDArray,
    *,
    field: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    if lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1, :]
    return lat, field


def _merge_hemispheres(
    field_sh: npt.NDArray,
    *,
    field_nh: npt.NDArray,
    lat_sh: npt.NDArray,
    lat_nh: npt.NDArray,
    target_lat: npt.NDArray,
) -> npt.NDArray:
    lat_sh, field_sh = _sort_to_ascending(lat_sh, field=field_sh)
    lat_nh, field_nh = _sort_to_ascending(lat_nh, field=field_nh)

    tol = 0.5 * float(np.abs(lat_sh[1] - lat_sh[0]))

    overlap_mask = np.array([
        np.any(np.abs(lat_nh - lat_val) < tol) for lat_val in lat_sh
    ])
    field_sh_trimmed = field_sh[~overlap_mask, :]
    lat_sh_trimmed = lat_sh[~overlap_mask]

    combined_field = np.concatenate([field_sh_trimmed, field_nh], axis=0)
    combined_lat = np.concatenate([lat_sh_trimmed, lat_nh])

    target_descending = target_lat[0] > target_lat[-1]
    sort_idx = np.argsort(combined_lat)
    if target_descending:
        sort_idx = sort_idx[::-1]

    return combined_field[sort_idx, :]


def _compute_yearly_area(
    *,
    year: int,
    mask_sh_path: pathlib.Path,
    mask_nh_path: pathlib.Path,
    target_lat: npt.NDArray,
    target_lon: npt.NDArray,
    n_intensity_cuts: int,
    ocean_mask: npt.NDArray,
    land_mask: npt.NDArray,
) -> typing.Tuple[
    npt.NDArray, npt.NDArray,
    typing.Dict[int, npt.NDArray], typing.Dict[int, npt.NDArray],
    typing.Dict[int, npt.NDArray], typing.Dict[int, npt.NDArray],
]:
    with netCDF4.Dataset(str(mask_sh_path)) as ds:
        flag_C_sh = np.asarray(ds["flag_C"][:])
        flag_A_sh = np.asarray(ds["flag_A"][:])
        int_C_sh = np.asarray(ds["intensity_C"][:])
        int_A_sh = np.asarray(ds["intensity_A"][:])
        lat_sh = np.asarray(ds["lat"][:])

    with netCDF4.Dataset(str(mask_nh_path)) as ds:
        flag_C_nh = np.asarray(ds["flag_C"][:])
        flag_A_nh = np.asarray(ds["flag_A"][:])
        int_C_nh = np.asarray(ds["intensity_C"][:])
        int_A_nh = np.asarray(ds["intensity_A"][:])
        lat_nh = np.asarray(ds["lat"][:])

    n_lat = target_lat.shape[0]
    n_lon = target_lon.shape[0]
    n_months = 12

    interp_shape = (n_lat + 2, n_lon)

    cycl_zon = np.zeros((n_intensity_cuts, n_months, n_lat))
    ant_zon = np.zeros((n_intensity_cuts, n_months, n_lat))
    cycl_land_zon: typing.Dict[int, npt.NDArray] = {
        0: np.zeros((n_months, n_lat)),
        5: np.zeros((n_months, n_lat)),
    }
    cycl_oce_zon: typing.Dict[int, npt.NDArray] = {
        0: np.zeros((n_months, n_lat)),
        5: np.zeros((n_months, n_lat)),
    }
    ant_land_zon: typing.Dict[int, npt.NDArray] = {0: np.zeros((n_months, n_lat))}
    ant_oce_zon: typing.Dict[int, npt.NDArray] = {0: np.zeros((n_months, n_lat))}

    monthnum = constants.MONTH_BOUNDARIES

    for month in range(n_months):
        m1 = monthnum[month]
        m2 = monthnum[month + 1]

        fC_sh = flag_C_sh[m1:m2]
        fA_sh = flag_A_sh[m1:m2]
        iC_sh = int_C_sh[m1:m2]
        iA_sh = int_A_sh[m1:m2]

        fC_nh = flag_C_nh[m1:m2]
        fA_nh = flag_A_nh[m1:m2]
        iC_nh = int_C_nh[m1:m2]
        iA_nh = int_A_nh[m1:m2]

        for cut in range(n_intensity_cuts):
            thresh = constants.INTENSITY_CUTS[cut]

            cyc_sh = fC_sh.copy()
            cyc_sh[iC_sh < thresh] = 0
            ant_sh = fA_sh.copy()
            ant_sh[iA_sh < thresh] = 0

            cyc_nh = fC_nh.copy()
            cyc_nh[iC_nh < thresh] = 0
            ant_nh = fA_nh.copy()
            ant_nh[iA_nh < thresh] = 0

            cyc_avg_sh = np.mean(cyc_sh, axis=0)
            ant_avg_sh = np.mean(ant_sh, axis=0)
            cyc_avg_nh = np.mean(cyc_nh, axis=0)
            ant_avg_nh = np.mean(ant_nh, axis=0)

            cyc_combined = _merge_hemispheres(
                cyc_avg_sh,
                field_nh=cyc_avg_nh,
                lat_sh=lat_sh,
                lat_nh=lat_nh,
                target_lat=target_lat,
            )
            ant_combined = _merge_hemispheres(
                ant_avg_sh,
                field_nh=ant_avg_nh,
                lat_sh=lat_sh,
                lat_nh=lat_nh,
                target_lat=target_lat,
            )

            cyc_hires = _interp_mask_field(cyc_combined, target_shape=interp_shape)[1:-1, :]
            ant_hires = _interp_mask_field(ant_combined, target_shape=interp_shape)[1:-1, :]

            cycl_zon[cut, month, :] = np.mean(cyc_hires, axis=1)
            ant_zon[cut, month, :] = np.mean(ant_hires, axis=1)

            if cut in (0, 5):
                cycl_land_zon[cut][month, :] = np.mean(cyc_hires * land_mask, axis=1)
                cycl_oce_zon[cut][month, :] = np.mean(cyc_hires * ocean_mask, axis=1)
                if cut == 0:
                    ant_land_zon[0][month, :] = np.mean(ant_hires * land_mask, axis=1)
                    ant_oce_zon[0][month, :] = np.mean(ant_hires * ocean_mask, axis=1)

    return cycl_zon, ant_zon, cycl_land_zon, cycl_oce_zon, ant_land_zon, ant_oce_zon


def _slice_mean(
    row: npt.NDArray,
    *,
    centre: int,
    half_win: int,
) -> float:
    i0 = max(0, centre - half_win)
    i1 = min(row.shape[0], centre + half_win)
    return float(np.mean(row[i0:i1]))


def _compute_3term_decomp_yearly(
    F_TE_cycl_zon: npt.NDArray,
    *,
    area_nh: npt.NDArray,
    area_sh: npt.NDArray,
    st_nh: npt.NDArray,
    st_sh: npt.NDArray,
    lat_f: npt.NDArray,
    n_fine: int,
    half_win: int,
) -> typing.Dict[str, npt.NDArray]:
    n_months = F_TE_cycl_zon.shape[0]
    F_TE_cycl_int = _interp_lat_2d(F_TE_cycl_zon, n_fine=n_fine)

    first_term_NH = np.zeros(n_months)
    first_term_SH = np.zeros(n_months)

    for n in range(n_months):
        flux_nh = _slice_mean(F_TE_cycl_int[n], centre=st_nh[n], half_win=half_win)
        flux_sh = _slice_mean(F_TE_cycl_int[n], centre=st_sh[n], half_win=half_win)
        sh_idx = (n - 6) % n_months
        first_term_NH[n] = flux_nh / (np.cos(np.deg2rad(lat_f[st_nh[n]])) * area_nh[n])
        first_term_SH[n] = flux_sh / (np.cos(np.deg2rad(lat_f[st_sh[n]])) * area_sh[sh_idx])

    plot_2_nh = np.zeros(n_months)
    plot_2_sh = np.zeros(n_months)
    plot_4_nh = np.zeros(n_months)
    plot_4_sh = np.zeros(n_months)
    plot_5_nh = np.zeros(n_months)
    plot_5_sh = np.zeros(n_months)

    for n in range(n_months):
        flux_nh = _slice_mean(F_TE_cycl_int[n], centre=st_nh[n], half_win=half_win)
        flux_sh = _slice_mean(F_TE_cycl_int[n], centre=st_sh[n], half_win=half_win)
        sh_idx = (n - 6) % n_months

        plot_2_nh[n] = flux_nh / (np.cos(np.deg2rad(lat_f[st_nh[n]])) * np.mean(area_nh))
        plot_2_sh[n] = flux_sh / (np.cos(np.deg2rad(lat_f[st_sh[n]])) * np.mean(area_sh))
        plot_4_nh[n] = flux_nh / (np.cos(np.deg2rad(lat_f[st_nh[n]])) * area_nh[n])
        plot_4_sh[n] = flux_sh / (np.cos(np.deg2rad(lat_f[st_sh[n]])) * area_sh[sh_idx])
        plot_5_nh[n] = (
            np.mean(first_term_NH)
            * (np.cos(np.deg2rad(lat_f[st_nh[n]])) * area_nh[n])
            / np.mean(area_nh)
        )
        plot_5_sh[n] = (
            np.mean(first_term_SH)
            * (np.cos(np.deg2rad(lat_f[st_sh[n]])) * area_sh[sh_idx])
            / np.mean(area_sh)
        )

    return {
        "plot_2_nh": plot_2_nh - np.mean(plot_2_nh),
        "plot_2_sh": plot_2_sh - np.mean(plot_2_sh),
        "plot_4_nh": plot_4_nh - np.mean(plot_4_nh),
        "plot_4_sh": plot_4_sh - np.mean(plot_4_sh),
        "plot_5_nh": plot_5_nh - np.mean(plot_5_nh),
        "plot_5_sh": plot_5_sh - np.mean(plot_5_sh),
    }


def _compute_DI_yearly(
    *,
    flux_dict: typing.Dict[str, npt.NDArray],
    area_nh_scalar: float,
    area_sh_scalar: float,
    st_nh: npt.NDArray,
    st_sh: npt.NDArray,
    stlat_nh: npt.NDArray,
    stlat_sh: npt.NDArray,
    n_fine: int,
    half_win: int,
    intensity_idx: int,
) -> typing.Dict[str, npt.NDArray]:
    out: typing.Dict[str, npt.NDArray] = {}
    flux_keys = [
        ("tot_energy", "tot_energy"),
        ("F_TE", "F_TE"),
        ("F_Swabs", "F_Swabs"),
        ("F_Olr", "F_Olr"),
        ("F_Dhdt", "F_Dhdt"),
        ("F_UM_z", "F_UM_z"),
    ]

    for out_key, flux_base in flux_keys:
        if intensity_idx == 0:
            if flux_base == "tot_energy":
                field_zon = (
                    (flux_dict["tot_energy_cycl_0"] - flux_dict["tot_energy_cycl_5"])
                    + (flux_dict["F_Dhdt_cycl_0"] - flux_dict["F_Dhdt_cycl_5"])
                )
            elif flux_base == "F_UM_z":
                field_zon = flux_dict["F_TE_z_cycl_0"] - flux_dict["F_TE_z_cycl_5"]
            else:
                field_zon = flux_dict["%s_cycl_0" % flux_base] - flux_dict["%s_cycl_5" % flux_base]
        else:
            if flux_base == "tot_energy":
                field_zon = flux_dict["tot_energy_cycl_5"] + flux_dict["F_Dhdt_cycl_5"]
            elif flux_base == "F_UM_z":
                field_zon = flux_dict.get(
                    "F_UM_z_cycl_5",
                    np.zeros_like(flux_dict["F_TE_cycl_5"]),
                )
            else:
                field_zon = flux_dict["%s_cycl_5" % flux_base]

        fld_int = _interp_lat_2d(field_zon, n_fine=n_fine)

        def norm_factor(lat_deg: npt.NDArray, *, area_mean: float) -> npt.NDArray:
            return (
                constants.EARTH_RADIUS
                * np.cos(np.deg2rad(lat_deg))
                * 2 * np.pi * area_mean
            )

        nh_raw = (
            _mean_around_track(fld_int, idx=st_nh, half_win=half_win)
            / norm_factor(stlat_nh, area_mean=area_nh_scalar)
        )
        sh_raw = (
            _mean_around_track(fld_int, idx=st_sh, half_win=half_win)
            / norm_factor(stlat_sh, area_mean=area_sh_scalar)
        )

        out["D_I_NH_%s%d" % (out_key, intensity_idx)] = nh_raw - np.mean(nh_raw)
        out["D_I_SH_%s%d" % (out_key, intensity_idx)] = sh_raw - np.mean(sh_raw)

    for hemi in ("NH", "SH"):
        tot = out["D_I_%s_tot_energy%d" % (hemi, intensity_idx)]
        olr = out["D_I_%s_F_Olr%d" % (hemi, intensity_idx)]
        sw = out["D_I_%s_F_Swabs%d" % (hemi, intensity_idx)]
        out["D_I_%s_F_Shf%d" % (hemi, intensity_idx)] = tot - olr - sw

    return out


def _compute_area_at_track_yearly(
    cycl_zon: npt.NDArray,
    *,
    ant_zon: npt.NDArray,
    st_nh: npt.NDArray,
    st_sh: npt.NDArray,
    n_fine: int,
    half_win: int,
) -> typing.Dict[str, npt.NDArray]:
    out: typing.Dict[str, npt.NDArray] = {}
    n_months = cycl_zon.shape[1]

    for cut_idx in (0, 5):
        cyc_int = _interp_lat_2d(cycl_zon[cut_idx], n_fine=n_fine)
        ant_int = _interp_lat_2d(ant_zon[cut_idx], n_fine=n_fine)

        nh_cyc = _mean_around_track(cyc_int, idx=st_nh, half_win=half_win)
        sh_cyc = np.zeros(n_months)
        for n in range(n_months):
            sh_cyc[(n - 6) % n_months] = _slice_mean(cyc_int[n], centre=st_sh[n], half_win=half_win)

        nh_ant = _mean_around_track(ant_int, idx=st_nh, half_win=half_win)
        sh_ant = np.zeros(n_months)
        for n in range(n_months):
            sh_ant[(n - 6) % n_months] = _slice_mean(ant_int[n], centre=st_sh[n], half_win=half_win)

        cut_label = 1 if cut_idx == 0 else 6
        out["cycl_NH_%d" % cut_label] = nh_cyc
        out["cycl_SH_%d" % cut_label] = sh_cyc
        out["ant_NH_%d" % cut_label] = nh_ant
        out["ant_SH_%d" % cut_label] = sh_ant

    return out


def compute_interannual_variability(
    *,
    flux_file: pathlib.Path,
    yearly_flux_files: typing.List[pathlib.Path],
    mask_sh_directory: pathlib.Path,
    mask_nh_directory: pathlib.Path,
    output_path: pathlib.Path,
    year_start: int,
    year_end: int,
    years_per_file: int = 5,
    track_half_width_deg: float = _TRACK_HALF_WIDTH_DEG,
    fine_grid_factor: int = _FINE_GRID_FACTOR,
) -> None:
    _LOG.info(
        "Computing interannual variability for years %d-%d",
        year_start, year_end - 1,
    )

    with netCDF4.Dataset(str(flux_file)) as ds:
        latitude = np.asarray(ds["lat"][:])
        if "lon" in ds.variables:
            longitude = np.asarray(ds["lon"][:])
        else:
            n_lon = ds["F_TE_final"].shape[-1]
            longitude = np.linspace(0, 360, n_lon, endpoint=False)

    n_lat = latitude.shape[0]
    n_lon = longitude.shape[0]
    n_fine = n_lat * fine_grid_factor
    lat_fine = _fine_lat(latitude, n_fine=n_fine)
    half_win = _half_win_from_lat(lat_fine, half_width_deg=track_half_width_deg)

    _LOG.info(
        "Grid: n_lat=%d, n_lon=%d, n_fine=%d, half_win=%d (%.1f deg)",
        n_lat, n_lon, n_fine, half_win, track_half_width_deg,
    )

    ocean_mask, land_mask = _build_land_ocean_masks(
        latitude=latitude, longitude=longitude,
    )
    n_intensity_cuts = len(constants.INTENSITY_CUTS)

    n_years = year_end - year_start
    n_months = 12
    n_lines = 4

    fig1_plot2 = np.zeros((n_years, n_lines, n_months))
    fig1_plot5 = np.zeros((n_years, n_lines, n_months))
    fig1_plot4 = np.zeros((n_years, n_lines, n_months))
    fig2a_area = np.zeros((n_years, n_lines, n_months))
    fig4_te_sd = np.zeros((n_years, n_lines))
    fig5_te_sd = np.zeros((n_years, n_lines))

    for yr_idx in range(n_years):
        year = year_start + yr_idx
        _LOG.info("Processing year %d (idx=%d)", year, yr_idx)

        mask_sh_path = mask_sh_directory / ("MASK_SH_%d.nc" % year)
        mask_nh_path = mask_nh_directory / ("MASK_NH_%d.nc" % year)

        if not mask_sh_path.exists() or not mask_nh_path.exists():
            _LOG.warning("Mask files not found for year %d, skipping", year)
            continue

        cycl_zon, ant_zon, cycl_land_zon, cycl_oce_zon, ant_land_zon, ant_oce_zon = (
            _compute_yearly_area(
                year=year,
                mask_sh_path=mask_sh_path,
                mask_nh_path=mask_nh_path,
                target_lat=latitude,
                target_lon=longitude,
                n_intensity_cuts=n_intensity_cuts,
                ocean_mask=ocean_mask,
                land_mask=land_mask,
            )
        )

        fluxes = _load_yearly_fluxes(
            yearly_files=yearly_flux_files,
            year_idx=yr_idx,
            years_per_file=years_per_file,
        )

        F_TE_total_zon = fluxes["F_TE_0"]
        F_TE_total_int = _interp_lat_2d(F_TE_total_zon, n_fine=n_fine)
        st_nh, st_sh, stlat_nh, stlat_sh = _stormtrack_from_total_fte(
            F_TE_total_int, lat_fine=lat_fine,
        )

        area_at_track = _compute_area_at_track_yearly(
            cycl_zon,
            ant_zon=ant_zon,
            st_nh=st_nh,
            st_sh=st_sh,
            n_fine=n_fine,
            half_win=half_win,
        )

        area_weak_nh = area_at_track["cycl_NH_1"] - area_at_track["cycl_NH_6"]
        area_weak_sh = area_at_track["cycl_SH_1"] - area_at_track["cycl_SH_6"]
        area_strong_nh = area_at_track["cycl_NH_6"]
        area_strong_sh = area_at_track["cycl_SH_6"]

        area_weak_nh_mean = float(np.mean(area_weak_nh))
        area_weak_sh_mean = float(np.mean(area_weak_sh))
        area_strong_nh_mean = float(np.mean(area_strong_nh))
        area_strong_sh_mean = float(np.mean(area_strong_sh))

        F_TE_cycl_weak_zon = fluxes["F_TE_cycl_0"] - fluxes["F_TE_cycl_5"]
        decomp_weak = _compute_3term_decomp_yearly(
            F_TE_cycl_weak_zon,
            area_nh=area_weak_nh, area_sh=area_weak_sh,
            st_nh=st_nh, st_sh=st_sh, lat_f=lat_fine,
            n_fine=n_fine, half_win=half_win,
        )

        F_TE_cycl_strong_zon = fluxes["F_TE_cycl_5"]
        decomp_strong = _compute_3term_decomp_yearly(
            F_TE_cycl_strong_zon,
            area_nh=area_strong_nh, area_sh=area_strong_sh,
            st_nh=st_nh, st_sh=st_sh, lat_f=lat_fine,
            n_fine=n_fine, half_win=half_win,
        )

        for key_base, storage in [("plot_2", fig1_plot2),
                                  ("plot_5", fig1_plot5),
                                  ("plot_4", fig1_plot4)]:
            storage[yr_idx, 0, :] = _running_mean(decomp_weak["%s_nh" % key_base])
            storage[yr_idx, 1, :] = _running_mean(decomp_weak["%s_sh" % key_base])
            storage[yr_idx, 2, :] = _running_mean(decomp_strong["%s_nh" % key_base])
            storage[yr_idx, 3, :] = _running_mean(decomp_strong["%s_sh" % key_base])

        AREA_TO_PERCENT = 100.0
        for line_idx, key in enumerate(("cycl_NH_1", "cycl_SH_1", "cycl_NH_6", "cycl_SH_6")):
            series = AREA_TO_PERCENT * area_at_track[key]
            fig2a_area[yr_idx, line_idx, :] = series - np.mean(series)

        DI_weak = _compute_DI_yearly(
            flux_dict=fluxes,
            area_nh_scalar=area_weak_nh_mean,
            area_sh_scalar=area_weak_sh_mean,
            st_nh=st_nh, st_sh=st_sh,
            stlat_nh=stlat_nh, stlat_sh=stlat_sh,
            n_fine=n_fine, half_win=half_win,
            intensity_idx=0,
        )

        DI_strong = _compute_DI_yearly(
            flux_dict=fluxes,
            area_nh_scalar=area_strong_nh_mean,
            area_sh_scalar=area_strong_sh_mean,
            st_nh=st_nh, st_sh=st_sh,
            stlat_nh=stlat_nh, stlat_sh=stlat_sh,
            n_fine=n_fine, half_win=half_win,
            intensity_idx=5,
        )

        te_weak_sh = DI_weak["D_I_SH_F_TE0"] * 1e15 * _PW_FACTOR
        te_weak_nh = DI_weak["D_I_NH_F_TE0"] * 1e15 * _PW_FACTOR
        te_strong_sh = DI_strong["D_I_SH_F_TE5"] * 1e15 * _PW_FACTOR
        te_strong_nh = DI_strong["D_I_NH_F_TE5"] * 1e15 * _PW_FACTOR

        fig4_te_sd[yr_idx, 0] = _seasonal_diff(te_weak_sh)
        fig4_te_sd[yr_idx, 1] = _seasonal_diff(te_weak_nh)
        fig4_te_sd[yr_idx, 2] = _seasonal_diff(te_strong_sh)
        fig4_te_sd[yr_idx, 3] = _seasonal_diff(te_strong_nh)

        fig5_te_sd[yr_idx, :] = 0.0

    _LOG.info("Computing std across years")

    fig1_band_a, fig1_std_a, fig1_std_per_month_a = _band_from_lines(fig1_plot2)
    fig1_band_b, fig1_std_b, fig1_std_per_month_b = _band_from_lines(fig1_plot5)
    fig1_band_c, fig1_std_c, fig1_std_per_month_c = _band_from_lines(fig1_plot4)

    _LOG.info("Fig 1 bands: a=%.6f, b=%.6f, c=%.6f", fig1_band_a, fig1_band_b, fig1_band_c)

    fig2_band_area, fig2_std_area, _ = _band_from_lines(fig2a_area)
    _LOG.info("Fig 2a band (area): %.6f", fig2_band_area)

    fig4_bands = np.std(fig4_te_sd, axis=0)
    _LOG.info("Fig 4 bands: %s", fig4_bands)

    fig5_bands = np.std(fig5_te_sd, axis=0)
    _LOG.info("Fig 5 bands: %s", fig5_bands)

    _LOG.info("Saving to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(str(output_path), "w", format="NETCDF4") as ds:
        ds.createDimension("scalar", 1)
        ds.createDimension("lines4", n_lines)
        ds.createDimension("month", n_months)
        ds.createDimension("panels4", n_lines)

        for name, val in [("fig1_band_a", fig1_band_a),
                          ("fig1_band_b", fig1_band_b),
                          ("fig1_band_c", fig1_band_c)]:
            v = ds.createVariable(name, "f8", ("scalar",))
            v[:] = val

        for name, arr in [("fig1_std_per_line_a", fig1_std_a),
                          ("fig1_std_per_line_b", fig1_std_b),
                          ("fig1_std_per_line_c", fig1_std_c)]:
            v = ds.createVariable(name, "f8", ("lines4",))
            v[:] = arr
            v.line_order = "weak_NH, weak_SH, strong_NH, strong_SH"

        for name, arr in [("fig1_std_month_a", fig1_std_per_month_a),
                          ("fig1_std_month_b", fig1_std_per_month_b),
                          ("fig1_std_month_c", fig1_std_per_month_c)]:
            v = ds.createVariable(name, "f8", ("lines4", "month"))
            v[:] = arr

        v = ds.createVariable("fig2_band_area", "f8", ("scalar",))
        v[:] = fig2_band_area
        v = ds.createVariable("fig2_std_per_line_area", "f8", ("lines4",))
        v[:] = fig2_std_area
        v.line_order = "NH_1-5, SH_1-5, NH_6+, SH_6+"

        v = ds.createVariable("fig4_bands", "f8", ("panels4",))
        v[:] = fig4_bands
        v.panel_order = "SH_weak, NH_weak, SH_strong, NH_strong"

        v = ds.createVariable("fig5_bands", "f8", ("panels4",))
        v[:] = fig5_bands
        v.panel_order = "CA_land, CA_ocean, 6CVU_land, 6CVU_ocean"

        ds.description = (
            "Interannual variability metrics for gray bands in Figures 1, 2, 4, 5. "
            "Computed as std across %d years (%d-%d). "
            "For multi-line panels, band = max(mean(std_per_month)) across lines."
        ) % (n_years, year_start, year_end - 1)
        ds.year_start = year_start
        ds.year_end = year_end
