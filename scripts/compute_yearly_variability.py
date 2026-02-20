#!/usr/bin/env python3
"""
Precompute interannual variability for gray bands in Figures 1, 2, 4, 5.

For each year 2000-2014:
  - Load per-year fluxes from YEARS files
  - Compute per-year area fractions from mask files
  - Run full decomposition pipeline (3-term decomp, DI, land/ocean DI, area anomaly)
Then compute std across years, take max across lines per panel, save to NetCDF.
"""

import pathlib
import sys
import time as _time

import netCDF4
import numpy as np
import pandas
import scipy.interpolate

# ---------------------------------------------------------------------------
# Constants (must match final_figures.ipynb exactly)
# ---------------------------------------------------------------------------
A_EARTH = 6.371e6
AREA_CUT = "0.225"
N_YEARS = 15
YEAR_START = 2000
YEAR_END = 2015
HALF_WIN = int(2844 / 2)
PW_FACTOR = 2 * np.pi * A_EARTH / 1e15
INTENSITY_CUT = np.array([1, 2, 3, 4, 5, 6])
SMOOTH_WINDOW = 3

MONTHNUM = np.array([0, 124, 236, 360, 480, 604, 724, 848, 972, 1092, 1216, 1336, 1460])
SEASON_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE = pathlib.Path("/project2/tas1/gmsarro")
NC_FLUX = BASE / "cyclone_centered" / f"WITH_INT_Cyclones_Sampled_Poleward_Fluxes_{AREA_CUT}.nc"
NC_CYC_INT = BASE / "track" / "final" / "cyclonic_intensity.nc"
NC_ANT_INT = BASE / "track" / "final" / "anticyclonic_intensity.nc"
CSV_DIR = BASE / "track" / "final"
YEARLY_FILES = [
    BASE / "cyclone_centered" / f"WITH_INT_Cyclones_Sampled_Poleward_Fluxes_YEARS_{s}.nc"
    for s in range(3)
]
MASK_SH_DIR = BASE / "cyclone_centered" / "masks" / f"SH{AREA_CUT}"
MASK_NH_DIR = BASE / "cyclone_centered" / "masks" / f"NH{AREA_CUT}"

# Interpolation grids
NLAT = 719
x_d = np.linspace(0, NLAT - 1, NLAT)
y_d = np.linspace(0, 12, 12)
x3_d = np.linspace(0, NLAT - 1, 25600)
y3_d = np.linspace(0, 12, 12)

# Mask grid → flux grid interpolation
x_mask = np.linspace(0, 240, 240)
y_mask = np.linspace(0, 121, 121)
xi_mask = np.linspace(0, 240, 1440)
yi_mask = np.linspace(0, 121, 721)


# ---------------------------------------------------------------------------
# Static land/ocean masks (from Cell 21 of final_figures.ipynb)
# ---------------------------------------------------------------------------
def build_land_ocean_masks():
    map_mask = np.ones((719, 1440))
    map_mask[361:, :] = 0
    map_mask[:, 0:450] = 0
    map_mask[:230, 950:1140] = 0
    map_mask[:190, 900:950] = 0
    map_mask[:190, 1140:1200] = 0
    map_mask[:220, 450:550] = 0
    map_mask[:270, 450:490] = 0
    map_mask[:130, 450:1350] = 0
    map_mask[180:350, 1390:] = 0
    map_mask[320:, 1120:1240] = 0
    map_mask[:40, :] = 1
    map_mask[280:361, 220:380] = 1
    map_mask[:90, :60] = 1
    map_mask[200:270, 490:550] = 1
    map_mask[180:210, 0:90] = 1
    map_mask[230:, :] = 0

    land_mask = (map_mask - 1) * (-1)
    land_mask[361:, :] = 0
    land_mask[230:, :] = 0

    return map_mask, land_mask


OCEAN_MASK, LAND_MASK = build_land_ocean_masks()


# ---------------------------------------------------------------------------
# Utility functions (matching final_figures.ipynb Cells 2 & 4)
# ---------------------------------------------------------------------------
def interp_lat_2d(field_12x_nlat):
    spline = scipy.interpolate.RectBivariateSpline(y_d, x_d, field_12x_nlat)
    return spline(y3_d, x3_d)


def fine_lat(latitude):
    f = scipy.interpolate.interp1d(x_d, latitude)
    return f(x3_d)


def stormtrack_from_total_fte(fte_zon_int, lat_fine):
    st_nh = np.argmax(fte_zon_int, axis=1)
    st_sh = np.argmin(fte_zon_int, axis=1)
    return st_nh, st_sh, lat_fine[st_nh], lat_fine[st_sh]


def mean_around_track(field_12x_nfine, idx_12, half_win=HALF_WIN):
    out = np.zeros(12)
    for n in range(12):
        i0 = idx_12[n] - half_win
        i1 = idx_12[n] + half_win
        out[n] = np.mean(field_12x_nfine[n, i0:i1])
    return out


def running_mean(data, window_size=SMOOTH_WINDOW):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def seasonal_diff(x):
    x = np.asarray(x)
    return np.mean(x[[11, 0, 1]]) - np.mean(x[[5, 6, 7]])


def interp_mask_field(field_121x240):
    """Interpolate a single (121, 240) mask field to (721, 1440) using bilinear."""
    spline = scipy.interpolate.RectBivariateSpline(
        y_mask, x_mask, field_121x240, kx=1, ky=1,
    )
    return spline(yi_mask, xi_mask)


# ---------------------------------------------------------------------------
# Load reference latitude from the flux file
# ---------------------------------------------------------------------------
with netCDF4.Dataset(str(NC_FLUX)) as ds:
    LATITUDE = np.asarray(ds["lat"][:])

LAT_FINE = fine_lat(LATITUDE)


# ---------------------------------------------------------------------------
# Per-year area fractions from mask files
# ---------------------------------------------------------------------------
def compute_yearly_area(year_val):
    """Compute per-year cycl/ant area fractions at zonal-mean and land/ocean.

    Returns:
      cycl_zon: (6, 12, 719) zonal-mean cyclone area fractions
      ant_zon:  (6, 12, 719) zonal-mean anticyclone area fractions
      cycl_land_zon: dict {0: (12,719), 5: (12,719)} land-masked cycl area
      cycl_oce_zon:  dict {0: (12,719), 5: (12,719)} ocean-masked cycl area
      ant_land_zon:  dict {0: (12,719)} land-masked ant area (cut 0 only)
      ant_oce_zon:   dict {0: (12,719)} ocean-masked ant area (cut 0 only)
    """
    sh_path = str(MASK_SH_DIR / f"MASK_SH_{year_val}.nc")
    nh_path = str(MASK_NH_DIR / f"MASK_NH_{year_val}.nc")

    with netCDF4.Dataset(sh_path) as ds:
        flag_C_sh = np.asarray(ds["flag_C"][:])
        flag_A_sh = np.asarray(ds["flag_A"][:])
        int_C_sh = np.asarray(ds["intensity_C"][:])
        int_A_sh = np.asarray(ds["intensity_A"][:])

    with netCDF4.Dataset(nh_path) as ds:
        flag_C_nh = np.asarray(ds["flag_C"][:])
        flag_A_nh = np.asarray(ds["flag_A"][:])
        int_C_nh = np.asarray(ds["intensity_C"][:])
        int_A_nh = np.asarray(ds["intensity_A"][:])

    cycl_zon = np.zeros((6, 12, NLAT))
    ant_zon = np.zeros((6, 12, NLAT))
    cycl_land_zon = {0: np.zeros((12, NLAT)), 5: np.zeros((12, NLAT))}
    cycl_oce_zon = {0: np.zeros((12, NLAT)), 5: np.zeros((12, NLAT))}
    ant_land_zon = {0: np.zeros((12, NLAT))}
    ant_oce_zon = {0: np.zeros((12, NLAT))}

    for month in range(12):
        m1 = MONTHNUM[month]
        m2 = MONTHNUM[month + 1]

        fC_sh = flag_C_sh[m1:m2]
        fA_sh = flag_A_sh[m1:m2]
        iC_sh = int_C_sh[m1:m2]
        iA_sh = int_A_sh[m1:m2]

        fC_nh = flag_C_nh[m1:m2]
        fA_nh = flag_A_nh[m1:m2]
        iC_nh = int_C_nh[m1:m2]
        iA_nh = int_A_nh[m1:m2]

        for cut in range(6):
            thresh = INTENSITY_CUT[cut]

            cyc_sh = fC_sh.copy()
            cyc_sh[iC_sh < thresh] = 0
            ant_sh = fA_sh.copy()
            ant_sh[iA_sh < thresh] = 0

            cyc_nh = fC_nh.copy()
            cyc_nh[iC_nh < thresh] = 0
            ant_nh = fA_nh.copy()
            ant_nh[iA_nh < thresh] = 0

            # Average over time steps first (then interpolate once)
            cyc_avg_sh = np.mean(cyc_sh, axis=0)  # (61, 240)
            ant_avg_sh = np.mean(ant_sh, axis=0)
            cyc_avg_nh = np.mean(cyc_nh, axis=0)
            ant_avg_nh = np.mean(ant_nh, axis=0)

            # Concatenate SH + NH (drop overlapping equator from SH), reverse lat
            cyc_combined = np.concatenate(
                [cyc_avg_sh[:-1, :], cyc_avg_nh], axis=0
            )[::-1, :]  # (121, 240)
            ant_combined = np.concatenate(
                [ant_avg_sh[:-1, :], ant_avg_nh], axis=0
            )[::-1, :]

            # Interpolate to (721, 1440) then trim poles → (719, 1440)
            cyc_hires = interp_mask_field(cyc_combined)[1:-1, :]  # (719, 1440)
            ant_hires = interp_mask_field(ant_combined)[1:-1, :]

            cycl_zon[cut, month, :] = np.mean(cyc_hires, axis=1)
            ant_zon[cut, month, :] = np.mean(ant_hires, axis=1)

            if cut in (0, 5):
                cycl_land_zon[cut][month, :] = np.mean(
                    cyc_hires * LAND_MASK, axis=1,
                )
                cycl_oce_zon[cut][month, :] = np.mean(
                    cyc_hires * OCEAN_MASK, axis=1,
                )
                if cut == 0:
                    ant_land_zon[0][month, :] = np.mean(
                        ant_hires * LAND_MASK, axis=1,
                    )
                    ant_oce_zon[0][month, :] = np.mean(
                        ant_hires * OCEAN_MASK, axis=1,
                    )

    return cycl_zon, ant_zon, cycl_land_zon, cycl_oce_zon, ant_land_zon, ant_oce_zon


# ---------------------------------------------------------------------------
# Per-year flux loading from YEARS files
# ---------------------------------------------------------------------------
def load_yearly_fluxes(year_idx):
    """Load per-year fluxes for intensity indices 0 and 5.

    year_idx: 0-14.
    Returns dict of (12, 719, 1440) or (12, 719) zonal-mean arrays.
    """
    stage = year_idx // 5
    yr_in_file = year_idx % 5

    result = {}
    with netCDF4.Dataset(str(YEARLY_FILES[stage])) as ds:
        lat = np.asarray(ds["lat"][:])

        for icut in (0, 5):
            for suffix in ("", "_cycl", "_ant"):
                vn_te = f"F_TE_final{suffix}"
                result[f"F_TE{suffix}_{icut}"] = np.mean(
                    ds[vn_te][icut, :, yr_in_file, :, :], axis=2,
                )  # (12, 719) zonal mean

                for vn_base in ("F_Swabs_final", "F_Olr_final", "F_Dhdt_final",
                                "tot_energy_final", "F_TE_z_final", "F_UM_z_final"):
                    vn = f"{vn_base}{suffix}"
                    if vn in ds.variables:
                        result[f"{vn_base.replace('_final','')}{suffix}_{icut}"] = np.mean(
                            ds[vn][icut, :, yr_in_file, :, :], axis=2,
                        )

            # For Figure 5: need full 2D fields for land/ocean masking (NH only)
            for suffix in ("_cycl", "_ant"):
                for vn_base in ("F_TE_final", "F_Swabs_final", "F_Olr_final",
                                "F_Dhdt_final", "tot_energy_final",
                                "F_TE_z_final", "F_UM_z_final"):
                    vn = f"{vn_base}{suffix}"
                    if vn in ds.variables:
                        key = f"2d_{vn_base.replace('_final','')}{suffix}_{icut}"
                        result[key] = np.asarray(
                            ds[vn][icut, :, yr_in_file, :, :],
                        )  # (12, 719, 1440)

    return result


# ---------------------------------------------------------------------------
# Figure 1: 3-term decomposition per year
# ---------------------------------------------------------------------------
def compute_3term_decomp_yearly(*, F_TE_cycl_zon, area_nh, area_sh,
                                st_nh, st_sh, lat_f):
    """Replicate _compute_3term_decomposition for a single year.

    F_TE_cycl_zon: (12, 719) zonal-mean cyclone TE flux
    area_nh, area_sh: (12,) monthly area at storm track
    st_nh, st_sh: (12,) storm-track indices in fine grid
    lat_f: fine-lat array (25600,)
    """
    F_TE_cycl_int = interp_lat_2d(F_TE_cycl_zon)  # (12, 25600)

    first_term_NH = np.zeros(12)
    first_term_SH = np.zeros(12)
    for n in range(12):
        flux_nh = np.mean(F_TE_cycl_int[n, st_nh[n] - HALF_WIN:st_nh[n] + HALF_WIN])
        flux_sh = np.mean(F_TE_cycl_int[n, st_sh[n] - HALF_WIN:st_sh[n] + HALF_WIN])
        first_term_NH[n] = flux_nh / (np.cos(np.deg2rad(lat_f[st_nh[n]])) * area_nh[n])
        first_term_SH[n] = flux_sh / (np.cos(np.deg2rad(lat_f[st_sh[n]])) * area_sh[n - 6])

    plot_2_nh = np.zeros(12)
    plot_2_sh = np.zeros(12)
    plot_4_nh = np.zeros(12)
    plot_4_sh = np.zeros(12)
    plot_5_nh = np.zeros(12)
    plot_5_sh = np.zeros(12)

    for n in range(12):
        flux_nh = np.mean(F_TE_cycl_int[n, st_nh[n] - HALF_WIN:st_nh[n] + HALF_WIN])
        flux_sh = np.mean(F_TE_cycl_int[n, st_sh[n] - HALF_WIN:st_sh[n] + HALF_WIN])

        plot_2_nh[n] = flux_nh / (np.cos(np.deg2rad(lat_f[st_nh[n]])) * np.mean(area_nh))
        plot_2_sh[n] = flux_sh / (np.cos(np.deg2rad(lat_f[st_sh[n]])) * np.mean(area_sh))
        plot_4_nh[n] = flux_nh / (np.cos(np.deg2rad(lat_f[st_nh[n]])) * area_nh[n])
        plot_4_sh[n] = flux_sh / (np.cos(np.deg2rad(lat_f[st_sh[n]])) * area_sh[n - 6])
        plot_5_nh[n] = (
            np.mean(first_term_NH)
            * (np.cos(np.deg2rad(lat_f[st_nh[n]])) * area_nh[n])
            / np.mean(area_nh)
        )
        plot_5_sh[n] = (
            np.mean(first_term_SH)
            * (np.cos(np.deg2rad(lat_f[st_sh[n]])) * area_sh[n - 6])
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


# ---------------------------------------------------------------------------
# Figure 4: DI per year (matching compute_DI_for_intensity in Cell 9)
# ---------------------------------------------------------------------------
def compute_DI_yearly(*, flux_dict, area_nh_scalar, area_sh_scalar,
                      st_nh, st_sh, stlat_nh, stlat_sh, lat_f,
                      intensity_idx):
    """Compute DI for all flux terms for a single year."""
    out = {}
    flux_keys = [
        ("tot_energy", "tot_energy"),
        ("F_TE", "F_TE"),
        ("F_Swabs", "F_Swabs"),
        ("F_Olr", "F_Olr"),
        ("F_Dhdt", "F_Dhdt"),
        ("F_UM_z", "F_UM_z"),
    ]

    for out_key, flux_base in flux_keys:
        # Build the zonal-mean field
        if intensity_idx == 0:
            if flux_base == "tot_energy":
                field_zon = (
                    (flux_dict[f"tot_energy_cycl_0"] - flux_dict[f"tot_energy_cycl_5"])
                    + (flux_dict[f"F_Dhdt_cycl_0"] - flux_dict[f"F_Dhdt_cycl_5"])
                )
            elif flux_base == "F_UM_z":
                field_zon = (
                    flux_dict[f"F_TE_z_cycl_0"] - flux_dict[f"F_TE_z_cycl_5"]
                )
            else:
                field_zon = (
                    flux_dict[f"{flux_base}_cycl_0"] - flux_dict[f"{flux_base}_cycl_5"]
                )
        else:
            if flux_base == "tot_energy":
                field_zon = (
                    flux_dict[f"tot_energy_cycl_5"]
                    + flux_dict[f"F_Dhdt_cycl_5"]
                )
            elif flux_base == "F_UM_z":
                field_zon = flux_dict[f"F_UM_z_cycl_5"]
            else:
                field_zon = flux_dict[f"{flux_base}_cycl_5"]

        fld_int = interp_lat_2d(field_zon)

        def norm_factor(lat_deg, area_mean):
            return A_EARTH * np.cos(np.deg2rad(lat_deg)) * 2 * np.pi * area_mean

        nh_raw = mean_around_track(fld_int, st_nh) / norm_factor(stlat_nh, area_nh_scalar)
        sh_raw = mean_around_track(fld_int, st_sh) / norm_factor(stlat_sh, area_sh_scalar)

        out[f"D_I_NH_{out_key}{intensity_idx}"] = nh_raw - np.mean(nh_raw)
        out[f"D_I_SH_{out_key}{intensity_idx}"] = sh_raw - np.mean(sh_raw)

    # Compute Shf = tot_energy - Olr - Swabs
    for hemi in ("NH", "SH"):
        tot = out[f"D_I_{hemi}_tot_energy{intensity_idx}"]
        olr = out[f"D_I_{hemi}_F_Olr{intensity_idx}"]
        sw = out[f"D_I_{hemi}_F_Swabs{intensity_idx}"]
        out[f"D_I_{hemi}_F_Shf{intensity_idx}"] = tot - olr - sw

    return out


# ---------------------------------------------------------------------------
# Figure 5: land/ocean DI per year (NH only, matching Cell 22)
# ---------------------------------------------------------------------------
def compute_DI_NH_land_ocean_yearly(*, flux_2d_dict, area_values, denom_mode,
                                    mask_2d, st_nh, lat_f):
    """Compute DI for NH with a land/ocean mask, for a single year."""
    out = {}
    denom_mean = float(np.mean(area_values))
    mask_expanded = mask_2d[np.newaxis, :, :]  # (1, 719, 1440)

    flux_map = [
        ("F_TE", "F_TE"), ("F_Swabs", "F_Swabs"), ("F_Olr", "F_Olr"),
        ("F_Dhdt", "F_Dhdt"), ("F_UM_z", "F_UM_z"), ("tot_energy", "tot_energy"),
    ]

    for out_key, flux_base in flux_map:
        arr_masked = flux_2d_dict[flux_base] * mask_expanded
        arr_zon = np.mean(arr_masked, axis=2)  # (12, 719)
        arr_int = interp_lat_2d(arr_zon)
        maxN = mean_around_track(arr_int, st_nh)
        if denom_mode == "per_month":
            norm = A_EARTH * np.cos(np.deg2rad(LAT_FINE[st_nh])) * 2 * np.pi * area_values
        else:
            norm = A_EARTH * np.cos(np.deg2rad(LAT_FINE[st_nh])) * 2 * np.pi * denom_mean
        series = maxN / norm
        out[out_key] = series - np.mean(series)

    out["F_Shf"] = out["tot_energy"] - out["F_Olr"] - out["F_Swabs"]
    return out


def build_bar_values_pw(DI):
    """Compute winter-summer seasonal difference for each flux term (PW)."""
    te = DI["F_TE"] * 1e15
    swabs = DI["F_Swabs"] * 1e15
    olr = DI["F_Olr"] * 1e15
    shf = DI["F_Shf"] * 1e15
    dhdt = -DI["F_Dhdt"] * 1e15
    umz = -DI["F_UM_z"] * 1e15
    tot = DI["tot_energy"] * 1e15
    residual = te - (tot - (-dhdt)) + (-umz)
    values = [
        seasonal_diff(te), seasonal_diff(swabs + olr), seasonal_diff(shf),
        seasonal_diff(dhdt), seasonal_diff(umz), seasonal_diff(residual),
    ]
    return [v * PW_FACTOR for v in values]


# ---------------------------------------------------------------------------
# Figure 2a: area anomaly at storm track per year
# ---------------------------------------------------------------------------
def compute_area_at_track_yearly(cycl_zon, ant_zon, st_nh, st_sh):
    """Compute area fractions at storm track for cuts 0 and 5.

    Returns dict of 12-element arrays.
    """
    out = {}
    for cut_idx in (0, 5):
        cyc_int = interp_lat_2d(cycl_zon[cut_idx])
        ant_int = interp_lat_2d(ant_zon[cut_idx])

        nh_cyc = mean_around_track(cyc_int, st_nh)
        sh_cyc = np.zeros(12)
        for n in range(12):
            sh_cyc[n - 6] = np.mean(
                cyc_int[n, st_sh[n] - HALF_WIN:st_sh[n] + HALF_WIN]
            )

        nh_ant = mean_around_track(ant_int, st_nh)
        sh_ant = np.zeros(12)
        for n in range(12):
            sh_ant[n - 6] = np.mean(
                ant_int[n, st_sh[n] - HALF_WIN:st_sh[n] + HALF_WIN]
            )

        cut_label = 1 if cut_idx == 0 else 6
        out[f"cycl_NH_{cut_label}"] = nh_cyc
        out[f"cycl_SH_{cut_label}"] = sh_cyc
        out[f"ant_NH_{cut_label}"] = nh_ant
        out[f"ant_SH_{cut_label}"] = sh_ant

    return out


# ---------------------------------------------------------------------------
# Figure 2b: track density per year (6CVU only, 5 years)
# ---------------------------------------------------------------------------
def compute_track_density_per_year():
    """Compute track density at storm track per year for 6CVU (2000-2004)."""
    lat_grid = np.linspace(-90, 90, 181)
    cosine_weights = np.cos(np.radians(lat_grid))
    lat_res = 181

    # Load climatological storm track for consistent referencing
    with netCDF4.Dataset(str(NC_FLUX)) as ds:
        F_TE_total = ds["F_TE_final"][0, :, :, :]
    F_TE_total_zon = np.mean(F_TE_total, axis=2)
    F_TE_int = interp_lat_2d(F_TE_total_zon)
    st_nh_clim, st_sh_clim, _, _ = stormtrack_from_total_fte(F_TE_int, LAT_FINE)

    all_year_td = {}
    available_years = range(2000, 2005)

    for yr in available_years:
        count_6cvu = np.zeros((12, lat_res))
        count_all = np.zeros((12, lat_res))

        for k, month_name in enumerate(SEASON_LABELS):
            csv_6 = CSV_DIR / f"{month_name}_count_6CVU_{yr}.csv"
            if csv_6.exists():
                gc6 = pandas.read_csv(str(csv_6)).values
                count_6cvu[k, :] = np.nanmean(gc6, axis=1) * cosine_weights * 3

        td_nh_6 = np.zeros(12)
        td_sh_6 = np.zeros(12)

        for n in range(12):
            for k in range(lat_res // 2, lat_res):
                if lat_grid[k] == int(LAT_FINE[st_nh_clim[n]]):
                    td_nh_6[n] = np.mean(count_6cvu[n, k - 10:k + 11])
            for k in range(0, lat_res // 2):
                if lat_grid[k] == int(LAT_FINE[st_sh_clim[n]]):
                    td_sh_6[n - 6] = np.mean(count_6cvu[n, k - 10:k + 11])

        all_year_td[yr] = {
            "td_nh_6": td_nh_6 - np.mean(td_nh_6),
            "td_sh_6": td_sh_6 - np.mean(td_sh_6),
        }

    return all_year_td


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = _time.time()
    print("=" * 70)
    print("Computing interannual variability for gray bands")
    print("=" * 70)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Storage: per-year results
    # ------------------------------------------------------------------
    # Fig 1: 3-term decomp, 4 lines per panel, 12 months
    # Lines: weak_NH, weak_SH, strong_NH, strong_SH
    fig1_plot2 = np.zeros((N_YEARS, 4, 12))  # panel (a): total transport
    fig1_plot5 = np.zeros((N_YEARS, 4, 12))  # panel (b): footprint
    fig1_plot4 = np.zeros((N_YEARS, 4, 12))  # panel (c): efficiency

    # Fig 2a: area anomaly, 4 lines (NH 1-5, SH 1-5, NH 6+, SH 6+)
    fig2a_area = np.zeros((N_YEARS, 4, 12))

    # Fig 4: seasonal diff of TE for 4 panels
    fig4_te_sd = np.zeros((N_YEARS, 4))  # (SH weak, NH weak, SH strong, NH strong)

    # Fig 5: seasonal diff of TE for 4 panels (NH only)
    fig5_te_sd = np.zeros((N_YEARS, 4))  # (CA land, CA oce, 6CVU land, 6CVU oce)

    # ------------------------------------------------------------------
    # Process each year
    # ------------------------------------------------------------------
    for yr_idx in range(N_YEARS):
        year_val = YEAR_START + yr_idx
        print(f"\n--- Year {year_val} (idx={yr_idx}) ---")
        sys.stdout.flush()
        t_yr = _time.time()

        # --- A. Per-year area fractions from masks ---
        print(f"  Computing area fractions from masks...", end=" ")
        sys.stdout.flush()
        (cycl_zon, ant_zon,
         cycl_land_zon, cycl_oce_zon,
         ant_land_zon, ant_oce_zon) = compute_yearly_area(year_val)
        print(f"done ({_time.time() - t_yr:.1f}s)")
        sys.stdout.flush()

        # --- B. Per-year fluxes from YEARS files ---
        print(f"  Loading per-year fluxes...")
        sys.stdout.flush()
        fluxes = load_yearly_fluxes(yr_idx)

        # --- C. Per-year storm track ---
        F_TE_total_zon = fluxes["F_TE_0"]  # total F_TE (all cyclones), zonal mean
        F_TE_total_int = interp_lat_2d(F_TE_total_zon)
        st_nh, st_sh, stlat_nh, stlat_sh = stormtrack_from_total_fte(
            F_TE_total_int, LAT_FINE,
        )

        # --- D. Per-year area at storm track ---
        area_at_track = compute_area_at_track_yearly(cycl_zon, ant_zon, st_nh, st_sh)

        # Weak (1-5 CVU): differential area = cut1 - cut6
        area_weak_nh = area_at_track["cycl_NH_1"] - area_at_track["cycl_NH_6"]
        area_weak_sh = area_at_track["cycl_SH_1"] - area_at_track["cycl_SH_6"]
        # Strong (6+ CVU)
        area_strong_nh = area_at_track["cycl_NH_6"]
        area_strong_sh = area_at_track["cycl_SH_6"]

        # Mean scalars for DI normalization
        area_weak_nh_mean = float(np.mean(area_weak_nh))
        area_weak_sh_mean = float(np.mean(area_weak_sh))
        area_strong_nh_mean = float(np.mean(area_strong_nh))
        area_strong_sh_mean = float(np.mean(area_strong_sh))

        # --- E. Figure 1: 3-term decomposition ---
        print(f"  Computing 3-term decomposition...")
        sys.stdout.flush()

        # Weak: F_TE_cycl = cycl[0] - cycl[5]
        F_TE_cycl_weak_zon = fluxes["F_TE_cycl_0"] - fluxes["F_TE_cycl_5"]
        decomp_weak = compute_3term_decomp_yearly(
            F_TE_cycl_zon=F_TE_cycl_weak_zon,
            area_nh=area_weak_nh, area_sh=area_weak_sh,
            st_nh=st_nh, st_sh=st_sh, lat_f=LAT_FINE,
        )

        # Strong: F_TE_cycl = cycl[5]
        F_TE_cycl_strong_zon = fluxes["F_TE_cycl_5"]
        decomp_strong = compute_3term_decomp_yearly(
            F_TE_cycl_zon=F_TE_cycl_strong_zon,
            area_nh=area_strong_nh, area_sh=area_strong_sh,
            st_nh=st_nh, st_sh=st_sh, lat_f=LAT_FINE,
        )

        # Store smoothed values (matching the notebook's running_mean)
        for key_base, storage in [("plot_2", fig1_plot2),
                                  ("plot_5", fig1_plot5),
                                  ("plot_4", fig1_plot4)]:
            storage[yr_idx, 0, :] = running_mean(decomp_weak[f"{key_base}_nh"])
            storage[yr_idx, 1, :] = running_mean(decomp_weak[f"{key_base}_sh"])
            storage[yr_idx, 2, :] = running_mean(decomp_strong[f"{key_base}_nh"])
            storage[yr_idx, 3, :] = running_mean(decomp_strong[f"{key_base}_sh"])

        # --- F. Figure 2a: area anomaly ---
        AREA_TO_PERCENT = 100.0
        area_nh_15 = area_at_track["cycl_NH_1"]
        area_sh_15 = area_at_track["cycl_SH_1"]
        area_nh_6 = area_at_track["cycl_NH_6"]
        area_sh_6 = area_at_track["cycl_SH_6"]

        fig2a_area[yr_idx, 0, :] = AREA_TO_PERCENT * area_nh_15 - np.mean(AREA_TO_PERCENT * area_nh_15)
        fig2a_area[yr_idx, 1, :] = AREA_TO_PERCENT * area_sh_15 - np.mean(AREA_TO_PERCENT * area_sh_15)
        fig2a_area[yr_idx, 2, :] = AREA_TO_PERCENT * area_nh_6 - np.mean(AREA_TO_PERCENT * area_nh_6)
        fig2a_area[yr_idx, 3, :] = AREA_TO_PERCENT * area_sh_6 - np.mean(AREA_TO_PERCENT * area_sh_6)

        # --- G. Figure 4: DI seasonal differences ---
        print(f"  Computing DI for Figs 4 & 5...")
        sys.stdout.flush()

        DI_weak = compute_DI_yearly(
            flux_dict=fluxes,
            area_nh_scalar=area_weak_nh_mean,
            area_sh_scalar=area_weak_sh_mean,
            st_nh=st_nh, st_sh=st_sh,
            stlat_nh=stlat_nh, stlat_sh=stlat_sh,
            lat_f=LAT_FINE,
            intensity_idx=0,
        )

        DI_strong = compute_DI_yearly(
            flux_dict=fluxes,
            area_nh_scalar=area_strong_nh_mean,
            area_sh_scalar=area_strong_sh_mean,
            st_nh=st_nh, st_sh=st_sh,
            stlat_nh=stlat_nh, stlat_sh=stlat_sh,
            lat_f=LAT_FINE,
            intensity_idx=5,
        )

        # TE seasonal diff for each Fig 4 panel
        te_weak_sh = DI_weak["D_I_SH_F_TE0"] * 1e15 * PW_FACTOR
        te_weak_nh = DI_weak["D_I_NH_F_TE0"] * 1e15 * PW_FACTOR
        te_strong_sh = DI_strong["D_I_SH_F_TE5"] * 1e15 * PW_FACTOR
        te_strong_nh = DI_strong["D_I_NH_F_TE5"] * 1e15 * PW_FACTOR

        fig4_te_sd[yr_idx, 0] = seasonal_diff(te_weak_sh)
        fig4_te_sd[yr_idx, 1] = seasonal_diff(te_weak_nh)
        fig4_te_sd[yr_idx, 2] = seasonal_diff(te_strong_sh)
        fig4_te_sd[yr_idx, 3] = seasonal_diff(te_strong_nh)

        # --- H. Figure 5: land/ocean DI (NH only) ---
        # Build full 2D flux dicts for land/ocean masking
        # C+A intensity 0
        ca_fields_2d = {}
        for flux_base in ("F_TE", "F_Swabs", "F_Olr", "F_Dhdt", "F_UM_z"):
            vn_c = f"2d_{flux_base}_cycl_0"
            vn_a = f"2d_{flux_base}_ant_0"
            if vn_c in fluxes and vn_a in fluxes:
                ca_fields_2d[flux_base] = fluxes[vn_c] + fluxes[vn_a]

        ca_tot_2d = (
            fluxes.get("2d_tot_energy_cycl_0", np.zeros((12, NLAT, 1440)))
            + fluxes.get("2d_tot_energy_ant_0", np.zeros((12, NLAT, 1440)))
            + fluxes.get("2d_F_Dhdt_cycl_0", np.zeros((12, NLAT, 1440)))
            + fluxes.get("2d_F_Dhdt_ant_0", np.zeros((12, NLAT, 1440)))
        )
        ca_fields_2d["tot_energy"] = ca_tot_2d

        # C+A area at track
        area_CA_nh = area_at_track["cycl_NH_1"] + area_at_track["ant_NH_1"]

        # Land/ocean fractions at track (using climatological masks)
        ocean_mask_avg = np.mean(OCEAN_MASK[np.newaxis, :, :], axis=2)  # (1, 719)
        land_mask_avg = np.mean(LAND_MASK[np.newaxis, :, :], axis=2)
        ocean_mask_rep = np.repeat(ocean_mask_avg, 12, axis=0)  # (12, 719)
        land_mask_rep = np.repeat(land_mask_avg, 12, axis=0)
        ocean_mask_int = scipy.interpolate.RectBivariateSpline(
            y_d, x_d, ocean_mask_rep,
        )(y3_d, x3_d)
        land_mask_int = scipy.interpolate.RectBivariateSpline(
            y_d, x_d, land_mask_rep,
        )(y3_d, x3_d)
        ocean_frac = np.array([ocean_mask_int[0, i] for i in st_nh])
        land_frac = np.array([land_mask_int[0, i] for i in st_nh])

        area_oce_CA = area_CA_nh * ocean_frac
        area_lan_CA = area_CA_nh * land_frac

        DI_ca_oce = compute_DI_NH_land_ocean_yearly(
            flux_2d_dict=ca_fields_2d, area_values=area_oce_CA,
            denom_mode="per_month", mask_2d=OCEAN_MASK, st_nh=st_nh, lat_f=LAT_FINE,
        )
        DI_ca_lan = compute_DI_NH_land_ocean_yearly(
            flux_2d_dict=ca_fields_2d, area_values=area_lan_CA,
            denom_mode="per_month", mask_2d=LAND_MASK, st_nh=st_nh, lat_f=LAT_FINE,
        )

        # Cycl 6+ CVU
        c5_fields_2d = {}
        for flux_base in ("F_TE", "F_Swabs", "F_Olr", "F_Dhdt", "F_UM_z"):
            vn_c = f"2d_{flux_base}_cycl_5"
            if vn_c in fluxes:
                c5_fields_2d[flux_base] = fluxes[vn_c]

        c5_tot_2d = (
            fluxes.get("2d_tot_energy_cycl_5", np.zeros((12, NLAT, 1440)))
            + fluxes.get("2d_F_Dhdt_cycl_5", np.zeros((12, NLAT, 1440)))
        )
        c5_fields_2d["tot_energy"] = c5_tot_2d

        # Area for 6+ CVU land/ocean from masks
        cyc6_land_int = interp_lat_2d(cycl_land_zon[5])
        cyc6_oce_int = interp_lat_2d(cycl_oce_zon[5])
        area_6cvu_land = mean_around_track(cyc6_land_int, st_nh)
        area_6cvu_oce = mean_around_track(cyc6_oce_int, st_nh)
        area_6cvu_total = mean_around_track(
            interp_lat_2d(cycl_zon[5]), st_nh,
        )

        DI_c5_land = compute_DI_NH_land_ocean_yearly(
            flux_2d_dict=c5_fields_2d, area_values=area_6cvu_land,
            denom_mode="mean", mask_2d=LAND_MASK, st_nh=st_nh, lat_f=LAT_FINE,
        )
        DI_c5_oce = compute_DI_NH_land_ocean_yearly(
            flux_2d_dict=c5_fields_2d, area_values=area_6cvu_oce,
            denom_mode="mean", mask_2d=OCEAN_MASK, st_nh=st_nh, lat_f=LAT_FINE,
        )

        # TE seasonal diff for Fig 5 panels
        fig5_te_sd[yr_idx, 0] = build_bar_values_pw(DI_ca_lan)[0]
        fig5_te_sd[yr_idx, 1] = build_bar_values_pw(DI_ca_oce)[0]
        fig5_te_sd[yr_idx, 2] = build_bar_values_pw(DI_c5_land)[0]
        fig5_te_sd[yr_idx, 3] = build_bar_values_pw(DI_c5_oce)[0]

        print(f"  Year {year_val} done ({_time.time() - t_yr:.1f}s total)")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Compute track density variability (limited to 5 years for 6CVU)
    # ------------------------------------------------------------------
    print("\nComputing track density variability...")
    sys.stdout.flush()
    td_yearly = compute_track_density_per_year()
    td_years = sorted(td_yearly.keys())
    n_td = len(td_years)
    td_nh_6_arr = np.array([running_mean(td_yearly[y]["td_nh_6"]) for y in td_years])
    td_sh_6_arr = np.array([running_mean(td_yearly[y]["td_sh_6"]) for y in td_years])

    # ------------------------------------------------------------------
    # Compute variability metrics
    # ------------------------------------------------------------------
    print("\nComputing std across years...")
    sys.stdout.flush()

    def band_from_lines(arr_NxLx12):
        """arr shape: (N_YEARS, n_lines, 12).
        For each line: std across years per month, mean over months → scalar.
        Return max across lines.
        """
        std_per_month = np.std(arr_NxLx12, axis=0)  # (n_lines, 12)
        mean_std = np.mean(std_per_month, axis=1)    # (n_lines,)
        return float(np.max(mean_std)), mean_std, std_per_month

    # Figure 1
    fig1_band_a, fig1_std_a, fig1_std_per_month_a = band_from_lines(fig1_plot2)
    fig1_band_b, fig1_std_b, fig1_std_per_month_b = band_from_lines(fig1_plot5)
    fig1_band_c, fig1_std_c, fig1_std_per_month_c = band_from_lines(fig1_plot4)

    print(f"  Fig 1 bands: a={fig1_band_a:.6f}, b={fig1_band_b:.6f}, c={fig1_band_c:.6f}")
    print(f"    Per-line std (a): {fig1_std_a}")
    print(f"    Per-line std (b): {fig1_std_b}")
    print(f"    Per-line std (c): {fig1_std_c}")

    # Figure 2a: area anomaly
    fig2_band_area, fig2_std_area, _ = band_from_lines(fig2a_area)
    print(f"  Fig 2a band (area): {fig2_band_area:.6f}")
    print(f"    Per-line std: {fig2_std_area}")

    # Figure 2b: track density (6CVU only, 5 years)
    td_combined = np.stack([td_nh_6_arr, td_sh_6_arr], axis=1)  # (5, 2, 12)
    fig2_band_track, fig2_std_track, _ = band_from_lines(td_combined)
    print(f"  Fig 2b band (track): {fig2_band_track:.6f} (from {n_td} years, 6CVU only)")

    # Figure 4: std of seasonal diff across years, per panel
    fig4_bands = np.std(fig4_te_sd, axis=0)  # (4,)
    print(f"  Fig 4 bands: {fig4_bands}")

    # Figure 5: std of seasonal diff across years, per panel
    fig5_bands = np.std(fig5_te_sd, axis=0)  # (4,)
    print(f"  Fig 5 bands: {fig5_bands}")

    # ------------------------------------------------------------------
    # Save to NetCDF
    # ------------------------------------------------------------------
    out_path = str(BASE / "mse_cyclone-energetics" / "interannual_variability.nc")
    print(f"\nSaving to {out_path}...")
    sys.stdout.flush()

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("scalar", 1)
        ds.createDimension("lines4", 4)
        ds.createDimension("month", 12)
        ds.createDimension("panels4", 4)

        # Figure 1 bands (scalars)
        for name, val in [("fig1_band_a", fig1_band_a),
                          ("fig1_band_b", fig1_band_b),
                          ("fig1_band_c", fig1_band_c)]:
            v = ds.createVariable(name, "f8", ("scalar",))
            v[:] = val

        # Figure 1 per-line std
        for name, arr in [("fig1_std_per_line_a", fig1_std_a),
                          ("fig1_std_per_line_b", fig1_std_b),
                          ("fig1_std_per_line_c", fig1_std_c)]:
            v = ds.createVariable(name, "f8", ("lines4",))
            v[:] = arr
            v.line_order = "weak_NH, weak_SH, strong_NH, strong_SH"

        # Figure 1 per-line per-month std
        for name, arr in [("fig1_std_month_a", fig1_std_per_month_a),
                          ("fig1_std_month_b", fig1_std_per_month_b),
                          ("fig1_std_month_c", fig1_std_per_month_c)]:
            v = ds.createVariable(name, "f8", ("lines4", "month"))
            v[:] = arr

        # Figure 2 bands
        v = ds.createVariable("fig2_band_area", "f8", ("scalar",))
        v[:] = fig2_band_area
        v = ds.createVariable("fig2_band_track", "f8", ("scalar",))
        v[:] = fig2_band_track
        v = ds.createVariable("fig2_std_per_line_area", "f8", ("lines4",))
        v[:] = fig2_std_area
        v.line_order = "NH_1-5, SH_1-5, NH_6+, SH_6+"

        # Figure 4 bands
        v = ds.createVariable("fig4_bands", "f8", ("panels4",))
        v[:] = fig4_bands
        v.panel_order = "SH_weak, NH_weak, SH_strong, NH_strong"

        # Figure 5 bands
        v = ds.createVariable("fig5_bands", "f8", ("panels4",))
        v[:] = fig5_bands
        v.panel_order = "CA_land, CA_ocean, 6CVU_land, 6CVU_ocean"

        ds.description = (
            "Interannual variability metrics for gray bands in Figures 1, 2, 4, 5. "
            "Computed as std across 15 years (2000-2014). "
            "For multi-line panels, band = max(mean(std_per_month)) across lines."
        )

    total_time = _time.time() - t0
    print(f"\nDone! Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
