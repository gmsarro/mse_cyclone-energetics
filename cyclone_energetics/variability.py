from __future__ import annotations

"""Interannual variability computation for confidence bands.

Processes per-year flux and mask data to compute the standard deviation
across years.  This module produces the gray confidence bands shown in
the final figures.

The workflow:
  1. For each year, load per-year fluxes and area fractions from mask files.
  2. Run the decomposition pipeline (3-term decomposition, DI, land/ocean).
  3. Compute std across years for each panel.
  4. Save results to a NetCDF file for use in plotting notebooks.
"""

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_HALF_WIN_FACTOR: float = 2844 / 2
_PW_FACTOR: float = 2 * np.pi * constants.EARTH_RADIUS / 1e15
_SMOOTH_WINDOW: int = 3


def _interp_lat_2d(
    field: npt.NDArray,
    *,
    n_months: int,
    n_lat: int,
    n_fine: int = 25600,
) -> npt.NDArray:
    """Interpolate a (n_months, n_lat) field to fine latitude grid."""
    x_d = np.linspace(0, n_lat - 1, n_lat)
    y_d = np.linspace(0, n_months, n_months)
    x3_d = np.linspace(0, n_lat - 1, n_fine)
    y3_d = np.linspace(0, n_months, n_months)
    spline = scipy.interpolate.RectBivariateSpline(y_d, x_d, field)
    return spline(y3_d, x3_d)


def _fine_lat(
    latitude: npt.NDArray,
    *,
    n_fine: int = 25600,
) -> npt.NDArray:
    """Interpolate latitude array to fine grid."""
    n_lat = latitude.shape[0]
    x_d = np.linspace(0, n_lat - 1, n_lat)
    x_fine = np.linspace(0, n_lat - 1, n_fine)
    return scipy.interpolate.interp1d(x_d, latitude)(x_fine)


def _interp_mask_field(
    field: npt.NDArray,
    *,
    target_shape: typing.Tuple[int, int],
) -> npt.NDArray:
    """Interpolate a mask field to target shape using bilinear."""
    in_ny, in_nx = field.shape
    out_ny, out_nx = target_shape
    y_in = np.linspace(0, in_ny, in_ny)
    x_in = np.linspace(0, in_nx, in_nx)
    y_out = np.linspace(0, in_ny, out_ny)
    x_out = np.linspace(0, in_nx, out_nx)
    spline = scipy.interpolate.RectBivariateSpline(y_in, x_in, field, kx=1, ky=1)
    return spline(y_out, x_out)


def _stormtrack_from_total_fte(
    fte_zon_int: npt.NDArray,
    lat_fine: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Find storm-track indices from TE flux maxima/minima."""
    st_nh = np.argmax(fte_zon_int, axis=1)
    st_sh = np.argmin(fte_zon_int, axis=1)
    return st_nh, st_sh, lat_fine[st_nh], lat_fine[st_sh]


def _mean_around_track(
    field: npt.NDArray,
    idx: npt.NDArray,
    *,
    half_win: int,
) -> npt.NDArray:
    """Compute mean around storm-track latitude for each month."""
    n_months = field.shape[0]
    out = np.zeros(n_months)
    for n in range(n_months):
        i0 = max(0, idx[n] - half_win)
        i1 = min(field.shape[1], idx[n] + half_win)
        out[n] = np.mean(field[n, i0:i1])
    return out


def _running_mean(
    data: npt.NDArray,
    *,
    window_size: int = _SMOOTH_WINDOW,
) -> npt.NDArray:
    """Apply running mean filter."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def _seasonal_diff(x: npt.NDArray) -> float:
    """Compute DJF minus JJA seasonal difference."""
    x = np.asarray(x)
    djf_idx = [11, 0, 1]
    jja_idx = [5, 6, 7]
    return float(np.mean(x[djf_idx]) - np.mean(x[jja_idx]))


def _band_from_lines(
    arr: npt.NDArray,
) -> typing.Tuple[float, npt.NDArray, npt.NDArray]:
    """Compute variability band from multi-line array.

    For shape (n_years, n_lines, n_months): std across years per month,
    mean over months → scalar.  Returns max across lines.
    """
    std_per_month = np.std(arr, axis=0)
    mean_std = np.mean(std_per_month, axis=1)
    return float(np.max(mean_std)), mean_std, std_per_month


def _build_land_ocean_masks(
    *,
    n_lat: int,
    n_lon: int,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """Build ocean and land masks for NH land/ocean decomposition.

    This creates simplified continental masks for the Northern Hemisphere
    based on approximate continental boundaries.  The masks are resolution-
    independent and scale to the provided grid dimensions.
    """
    ocean_mask = np.ones((n_lat, n_lon))

    lat_frac = n_lat / 719
    lon_frac = n_lon / 1440

    def scale_lat(idx: int) -> int:
        return int(round(idx * lat_frac))

    def scale_lon(idx: int) -> int:
        return int(round(idx * lon_frac))

    ocean_mask[scale_lat(361):, :] = 0
    ocean_mask[:, 0:scale_lon(450)] = 0
    ocean_mask[:scale_lat(230), scale_lon(950):scale_lon(1140)] = 0
    ocean_mask[:scale_lat(190), scale_lon(900):scale_lon(950)] = 0
    ocean_mask[:scale_lat(190), scale_lon(1140):scale_lon(1200)] = 0
    ocean_mask[:scale_lat(220), scale_lon(450):scale_lon(550)] = 0
    ocean_mask[:scale_lat(270), scale_lon(450):scale_lon(490)] = 0
    ocean_mask[:scale_lat(130), scale_lon(450):scale_lon(1350)] = 0
    ocean_mask[scale_lat(180):scale_lat(350), scale_lon(1390):] = 0
    ocean_mask[scale_lat(320):, scale_lon(1120):scale_lon(1240)] = 0
    ocean_mask[:scale_lat(40), :] = 1
    ocean_mask[scale_lat(280):scale_lat(361), scale_lon(220):scale_lon(380)] = 1
    ocean_mask[:scale_lat(90), :scale_lon(60)] = 1
    ocean_mask[scale_lat(200):scale_lat(270), scale_lon(490):scale_lon(550)] = 1
    ocean_mask[scale_lat(180):scale_lat(210), 0:scale_lon(90)] = 1
    ocean_mask[scale_lat(230):, :] = 0

    land_mask = (ocean_mask - 1) * (-1)
    land_mask[scale_lat(361):, :] = 0
    land_mask[scale_lat(230):, :] = 0

    return ocean_mask, land_mask


def _load_yearly_fluxes(
    *,
    yearly_files: typing.List[pathlib.Path],
    year_idx: int,
    years_per_file: int,
) -> typing.Dict[str, npt.NDArray]:
    """Load per-year fluxes for intensity indices 0 and 5."""
    stage = year_idx // years_per_file
    yr_in_file = year_idx % years_per_file

    if stage >= len(yearly_files):
        raise IndexError(
            f"Year index {year_idx} requires file index {stage}, "
            f"but only {len(yearly_files)} files available"
        )

    result: typing.Dict[str, npt.NDArray] = {}
    with netCDF4.Dataset(str(yearly_files[stage])) as ds:
        for icut in (0, 5):
            for suffix in ("", "_cycl", "_ant"):
                vn_te = f"F_TE_final{suffix}"
                if vn_te in ds.variables:
                    arr = ds[vn_te][icut, :, yr_in_file, :, :]
                    result[f"F_TE{suffix}_{icut}"] = np.mean(arr, axis=2)

                for vn_base in ("F_Swabs_final", "F_Olr_final", "F_Dhdt_final",
                                "tot_energy_final", "F_TE_z_final", "F_UM_z_final"):
                    vn = f"{vn_base}{suffix}"
                    if vn in ds.variables:
                        arr = ds[vn][icut, :, yr_in_file, :, :]
                        key = f"{vn_base.replace('_final', '')}{suffix}_{icut}"
                        result[key] = np.mean(arr, axis=2)

            for suffix in ("_cycl", "_ant"):
                for vn_base in ("F_TE_final", "F_Swabs_final", "F_Olr_final",
                                "F_Dhdt_final", "tot_energy_final",
                                "F_TE_z_final", "F_UM_z_final"):
                    vn = f"{vn_base}{suffix}"
                    if vn in ds.variables:
                        key = f"2d_{vn_base.replace('_final', '')}{suffix}_{icut}"
                        result[key] = np.asarray(ds[vn][icut, :, yr_in_file, :, :])

    return result


def _compute_yearly_area(
    *,
    year: int,
    mask_sh_path: pathlib.Path,
    mask_nh_path: pathlib.Path,
    n_lat: int,
    n_lon: int,
    n_intensity_cuts: int,
    ocean_mask: npt.NDArray,
    land_mask: npt.NDArray,
) -> typing.Tuple[
    npt.NDArray, npt.NDArray,
    typing.Dict[int, npt.NDArray], typing.Dict[int, npt.NDArray],
    typing.Dict[int, npt.NDArray], typing.Dict[int, npt.NDArray],
]:
    """Compute per-year cyclone/anticyclone area fractions from mask files."""
    with netCDF4.Dataset(str(mask_sh_path)) as ds:
        flag_C_sh = np.asarray(ds["flag_C"][:])
        flag_A_sh = np.asarray(ds["flag_A"][:])
        int_C_sh = np.asarray(ds["intensity_C"][:])
        int_A_sh = np.asarray(ds["intensity_A"][:])

    with netCDF4.Dataset(str(mask_nh_path)) as ds:
        flag_C_nh = np.asarray(ds["flag_C"][:])
        flag_A_nh = np.asarray(ds["flag_A"][:])
        int_C_nh = np.asarray(ds["intensity_C"][:])
        int_A_nh = np.asarray(ds["intensity_A"][:])

    n_months = 12
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

            cyc_combined = np.concatenate(
                [cyc_avg_sh[:-1, :], cyc_avg_nh], axis=0
            )[::-1, :]
            ant_combined = np.concatenate(
                [ant_avg_sh[:-1, :], ant_avg_nh], axis=0
            )[::-1, :]

            target_shape = (n_lat + 2, n_lon)
            cyc_hires = _interp_mask_field(cyc_combined, target_shape=target_shape)[1:-1, :]
            ant_hires = _interp_mask_field(ant_combined, target_shape=target_shape)[1:-1, :]

            cycl_zon[cut, month, :] = np.mean(cyc_hires, axis=1)
            ant_zon[cut, month, :] = np.mean(ant_hires, axis=1)

            if cut in (0, 5):
                cycl_land_zon[cut][month, :] = np.mean(cyc_hires * land_mask, axis=1)
                cycl_oce_zon[cut][month, :] = np.mean(cyc_hires * ocean_mask, axis=1)
                if cut == 0:
                    ant_land_zon[0][month, :] = np.mean(ant_hires * land_mask, axis=1)
                    ant_oce_zon[0][month, :] = np.mean(ant_hires * ocean_mask, axis=1)

    return cycl_zon, ant_zon, cycl_land_zon, cycl_oce_zon, ant_land_zon, ant_oce_zon


def _compute_3term_decomp_yearly(
    *,
    F_TE_cycl_zon: npt.NDArray,
    area_nh: npt.NDArray,
    area_sh: npt.NDArray,
    st_nh: npt.NDArray,
    st_sh: npt.NDArray,
    lat_f: npt.NDArray,
    n_lat: int,
    half_win: int,
) -> typing.Dict[str, npt.NDArray]:
    """Compute 3-term decomposition for a single year."""
    n_months = F_TE_cycl_zon.shape[0]
    F_TE_cycl_int = _interp_lat_2d(F_TE_cycl_zon, n_months=n_months, n_lat=n_lat)

    first_term_NH = np.zeros(n_months)
    first_term_SH = np.zeros(n_months)

    for n in range(n_months):
        i0_nh = max(0, st_nh[n] - half_win)
        i1_nh = min(F_TE_cycl_int.shape[1], st_nh[n] + half_win)
        flux_nh = np.mean(F_TE_cycl_int[n, i0_nh:i1_nh])

        i0_sh = max(0, st_sh[n] - half_win)
        i1_sh = min(F_TE_cycl_int.shape[1], st_sh[n] + half_win)
        flux_sh = np.mean(F_TE_cycl_int[n, i0_sh:i1_sh])

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
        i0_nh = max(0, st_nh[n] - half_win)
        i1_nh = min(F_TE_cycl_int.shape[1], st_nh[n] + half_win)
        flux_nh = np.mean(F_TE_cycl_int[n, i0_nh:i1_nh])

        i0_sh = max(0, st_sh[n] - half_win)
        i1_sh = min(F_TE_cycl_int.shape[1], st_sh[n] + half_win)
        flux_sh = np.mean(F_TE_cycl_int[n, i0_sh:i1_sh])

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
    n_lat: int,
    half_win: int,
    intensity_idx: int,
) -> typing.Dict[str, npt.NDArray]:
    """Compute DI for all flux terms for a single year."""
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
                field_zon = flux_dict[f"{flux_base}_cycl_0"] - flux_dict[f"{flux_base}_cycl_5"]
        else:
            if flux_base == "tot_energy":
                field_zon = flux_dict["tot_energy_cycl_5"] + flux_dict["F_Dhdt_cycl_5"]
            elif flux_base == "F_UM_z":
                field_zon = flux_dict.get("F_UM_z_cycl_5", np.zeros_like(flux_dict["F_TE_cycl_5"]))
            else:
                field_zon = flux_dict[f"{flux_base}_cycl_5"]

        n_months = field_zon.shape[0]
        fld_int = _interp_lat_2d(field_zon, n_months=n_months, n_lat=n_lat)

        def norm_factor(lat_deg: npt.NDArray, area_mean: float) -> npt.NDArray:
            return constants.EARTH_RADIUS * np.cos(np.deg2rad(lat_deg)) * 2 * np.pi * area_mean

        nh_raw = _mean_around_track(fld_int, st_nh, half_win=half_win) / norm_factor(stlat_nh, area_nh_scalar)
        sh_raw = _mean_around_track(fld_int, st_sh, half_win=half_win) / norm_factor(stlat_sh, area_sh_scalar)

        out[f"D_I_NH_{out_key}{intensity_idx}"] = nh_raw - np.mean(nh_raw)
        out[f"D_I_SH_{out_key}{intensity_idx}"] = sh_raw - np.mean(sh_raw)

    for hemi in ("NH", "SH"):
        tot = out[f"D_I_{hemi}_tot_energy{intensity_idx}"]
        olr = out[f"D_I_{hemi}_F_Olr{intensity_idx}"]
        sw = out[f"D_I_{hemi}_F_Swabs{intensity_idx}"]
        out[f"D_I_{hemi}_F_Shf{intensity_idx}"] = tot - olr - sw

    return out


def _compute_area_at_track_yearly(
    cycl_zon: npt.NDArray,
    ant_zon: npt.NDArray,
    st_nh: npt.NDArray,
    st_sh: npt.NDArray,
    *,
    n_lat: int,
    half_win: int,
) -> typing.Dict[str, npt.NDArray]:
    """Compute area fractions at storm track for intensity cuts 0 and 5."""
    out: typing.Dict[str, npt.NDArray] = {}
    n_months = cycl_zon.shape[1]

    for cut_idx in (0, 5):
        cyc_int = _interp_lat_2d(cycl_zon[cut_idx], n_months=n_months, n_lat=n_lat)
        ant_int = _interp_lat_2d(ant_zon[cut_idx], n_months=n_months, n_lat=n_lat)

        nh_cyc = _mean_around_track(cyc_int, st_nh, half_win=half_win)
        sh_cyc = np.zeros(n_months)
        for n in range(n_months):
            i0 = max(0, st_sh[n] - half_win)
            i1 = min(cyc_int.shape[1], st_sh[n] + half_win)
            sh_cyc[(n - 6) % n_months] = np.mean(cyc_int[n, i0:i1])

        nh_ant = _mean_around_track(ant_int, st_nh, half_win=half_win)
        sh_ant = np.zeros(n_months)
        for n in range(n_months):
            i0 = max(0, st_sh[n] - half_win)
            i1 = min(ant_int.shape[1], st_sh[n] + half_win)
            sh_ant[(n - 6) % n_months] = np.mean(ant_int[n, i0:i1])

        cut_label = 1 if cut_idx == 0 else 6
        out[f"cycl_NH_{cut_label}"] = nh_cyc
        out[f"cycl_SH_{cut_label}"] = sh_cyc
        out[f"ant_NH_{cut_label}"] = nh_ant
        out[f"ant_SH_{cut_label}"] = sh_ant

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
) -> None:
    """Compute interannual variability and save to NetCDF.

    Parameters
    ----------
    flux_file
        Path to the main flux file (e.g. Cyclones_Sampled_Poleward_Fluxes.nc)
    yearly_flux_files
        List of per-year flux files (YEARS_0.nc, YEARS_1.nc, ...)
    mask_sh_directory
        Directory containing SH mask files (MASK_SH_{year}.nc)
    mask_nh_directory
        Directory containing NH mask files (MASK_NH_{year}.nc)
    output_path
        Output NetCDF file path
    year_start
        Start year (inclusive)
    year_end
        End year (exclusive)
    years_per_file
        Number of years stored in each yearly flux file
    """
    _LOG.info(
        "Computing interannual variability for years %d-%d",
        year_start, year_end - 1,
    )

    with netCDF4.Dataset(str(flux_file)) as ds:
        latitude = np.asarray(ds["lat"][:])
        n_lat = latitude.shape[0]
        sample_field = ds["F_TE_final"][0, :, :, :]
        n_lon = sample_field.shape[2]

    n_fine = int(round(n_lat * 25600 / 719))
    lat_fine = _fine_lat(latitude, n_fine=n_fine)
    half_win = int(round(_HALF_WIN_FACTOR * n_fine / 25600))

    ocean_mask, land_mask = _build_land_ocean_masks(n_lat=n_lat, n_lon=n_lon)
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

        mask_sh_path = mask_sh_directory / f"MASK_SH_{year}.nc"
        mask_nh_path = mask_nh_directory / f"MASK_NH_{year}.nc"

        if not mask_sh_path.exists() or not mask_nh_path.exists():
            _LOG.warning("Mask files not found for year %d, skipping", year)
            continue

        cycl_zon, ant_zon, cycl_land_zon, cycl_oce_zon, ant_land_zon, ant_oce_zon = (
            _compute_yearly_area(
                year=year,
                mask_sh_path=mask_sh_path,
                mask_nh_path=mask_nh_path,
                n_lat=n_lat,
                n_lon=n_lon,
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
        F_TE_total_int = _interp_lat_2d(F_TE_total_zon, n_months=n_months, n_lat=n_lat, n_fine=n_fine)
        st_nh, st_sh, stlat_nh, stlat_sh = _stormtrack_from_total_fte(F_TE_total_int, lat_fine)

        area_at_track = _compute_area_at_track_yearly(
            cycl_zon, ant_zon, st_nh, st_sh,
            n_lat=n_lat, half_win=half_win,
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
            F_TE_cycl_zon=F_TE_cycl_weak_zon,
            area_nh=area_weak_nh, area_sh=area_weak_sh,
            st_nh=st_nh, st_sh=st_sh, lat_f=lat_fine,
            n_lat=n_lat, half_win=half_win,
        )

        F_TE_cycl_strong_zon = fluxes["F_TE_cycl_5"]
        decomp_strong = _compute_3term_decomp_yearly(
            F_TE_cycl_zon=F_TE_cycl_strong_zon,
            area_nh=area_strong_nh, area_sh=area_strong_sh,
            st_nh=st_nh, st_sh=st_sh, lat_f=lat_fine,
            n_lat=n_lat, half_win=half_win,
        )

        for key_base, storage in [("plot_2", fig1_plot2),
                                  ("plot_5", fig1_plot5),
                                  ("plot_4", fig1_plot4)]:
            storage[yr_idx, 0, :] = _running_mean(decomp_weak[f"{key_base}_nh"])
            storage[yr_idx, 1, :] = _running_mean(decomp_weak[f"{key_base}_sh"])
            storage[yr_idx, 2, :] = _running_mean(decomp_strong[f"{key_base}_nh"])
            storage[yr_idx, 3, :] = _running_mean(decomp_strong[f"{key_base}_sh"])

        AREA_TO_PERCENT = 100.0
        area_nh_15 = area_at_track["cycl_NH_1"]
        area_sh_15 = area_at_track["cycl_SH_1"]
        area_nh_6 = area_at_track["cycl_NH_6"]
        area_sh_6 = area_at_track["cycl_SH_6"]

        fig2a_area[yr_idx, 0, :] = AREA_TO_PERCENT * area_nh_15 - np.mean(AREA_TO_PERCENT * area_nh_15)
        fig2a_area[yr_idx, 1, :] = AREA_TO_PERCENT * area_sh_15 - np.mean(AREA_TO_PERCENT * area_sh_15)
        fig2a_area[yr_idx, 2, :] = AREA_TO_PERCENT * area_nh_6 - np.mean(AREA_TO_PERCENT * area_nh_6)
        fig2a_area[yr_idx, 3, :] = AREA_TO_PERCENT * area_sh_6 - np.mean(AREA_TO_PERCENT * area_sh_6)

        DI_weak = _compute_DI_yearly(
            flux_dict=fluxes,
            area_nh_scalar=area_weak_nh_mean,
            area_sh_scalar=area_weak_sh_mean,
            st_nh=st_nh, st_sh=st_sh,
            stlat_nh=stlat_nh, stlat_sh=stlat_sh,
            n_lat=n_lat, half_win=half_win,
            intensity_idx=0,
        )

        DI_strong = _compute_DI_yearly(
            flux_dict=fluxes,
            area_nh_scalar=area_strong_nh_mean,
            area_sh_scalar=area_strong_sh_mean,
            st_nh=st_nh, st_sh=st_sh,
            stlat_nh=stlat_nh, stlat_sh=stlat_sh,
            n_lat=n_lat, half_win=half_win,
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
            f"Interannual variability metrics for gray bands in Figures 1, 2, 4, 5. "
            f"Computed as std across {n_years} years ({year_start}-{year_end - 1}). "
            f"For multi-line panels, band = max(mean(std_per_month)) across lines."
        )
        ds.year_start = year_start
        ds.year_end = year_end

    _LOG.info("Done!")
