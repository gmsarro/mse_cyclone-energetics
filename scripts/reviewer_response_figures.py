"""Figures for the response to reviewers (not part of the manuscript).

R1: raw vs Hoskins-filtered 6-hourly total energy flux divergence.
R2: NH cyclone-centered annual-mean zonal MSE advection composite.
R3: SH cyclone-centered annual-mean SHF composite and its meridional anomaly.
R4: NH DJF-JJA SHF change split by feature masks.

All inputs are pre-computed netCDF files; this script only plots.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = "/project2/tas1/gmsarro"
OUT_DIR = os.path.join(ROOT, "mse_cyclone-energetics", "figures", "reviewer_response")
os.makedirs(OUT_DIR, exist_ok=True)

EARTH_RADIUS = 6.371e6

_COLORS = [
    "#4B0082", "#3F51B5", "#2196F3", "dodgerblue", "skyblue",
    "#FFFFFF", "#FFFFFF",
    "lightpink", "#FF9800", "#F44336", "#B71C1C", "maroon",
]
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "PurpleBlue_White_OrangeRed", _COLORS, N=256,
)


def figure_r1_smoothing():
    """Raw vs Hoskins-filtered 6-hourly total energy flux divergence.

    Shows the exact field used in the manuscript budget (from which SHF is
    diagnosed as a residual): the full (zonal + meridional) divergence of
    the vertically integrated energy flux, composed of the 6-hourly ERA5
    vertical-integral products p85.162 (geopotential flux divergence),
    p84.162 (moisture flux divergence, times Lv) and p83.162 (thermal
    energy flux divergence), as in cyclone_energetics.integration.poleward.
    Maps and time series are weighted by cos(lat) to remove the metric
    amplification at high latitudes. The stored filtered fields were produced
    with the Hoskins filter at n0=27, r=1 (the parameters used for the
    divergence in the manuscript). Note: the raw and filtered files store
    latitude in opposite order, so each field uses its own coordinates.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    LV = 2.501e6  # cyclone_energetics.constants.LATENT_HEAT_VAPORIZATION

    raw = xr.open_dataset(
        "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/vint/"
        "era5_vint_2000_01.6hrly.nc")
    smo = xr.open_dataset(f"{ROOT}/smoothed_vint/era5_vint_2000_01_filtered.nc")

    def total_div(ds, suffix=""):
        return (ds["p85.162" + suffix] + ds["p84.162" + suffix] * LV
                + ds["p83.162" + suffix])

    raw_tot = total_div(raw)
    smo_tot = total_div(smo, "_filtered")

    t0 = 0

    # Time series at a NH stormtrack point (45N, 180E), selected on each
    # dataset's own coordinates and weighted by cos(45).
    cos45 = np.cos(np.deg2rad(45.0))
    ts_raw = raw_tot.sel(latitude=45.0, longitude=180.0,
                         method="nearest").values * cos45
    ts_smo = smo_tot.sel(latitude=45.0, longitude=180.0,
                         method="nearest").values * cos45
    time_days = np.arange(ts_raw.size) * 0.25

    def _to_180(lon):
        lon_p = np.where(lon > 180, lon - 360, lon)
        order = np.argsort(lon_p)
        return lon_p[order], order

    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.85],
                          hspace=0.25, wspace=0.07,
                          left=0.055, right=0.90, top=0.93, bottom=0.09)

    levels_snap = np.arange(-2500, 2501, 250)
    levels_snap = levels_snap[levels_snap != 0]
    panels = [
        (gs[0, 0], raw_tot[t0].values, raw["latitude"].values,
         raw["longitude"].values,
         "(a) $\\cos\\phi\\,\\nabla\\cdot\\langle\\mathbf{v}m\\rangle$ "
         "2000-01-01 00UTC\nraw ERA5 (0.25$^\\circ$)"),
        (gs[0, 1], smo_tot[t0].values, smo["latitude"].values,
         smo["longitude"].values,
         "(b) $\\cos\\phi\\,\\nabla\\cdot\\langle\\mathbf{v}m\\rangle$ "
         "2000-01-01 00UTC\nHoskins-filtered ($n_0=27$, $r=1$)"),
    ]
    for spec, field, lat, lon, title in panels:
        lon_p, order = _to_180(lon)
        field_w = field[:, order] * np.cos(np.deg2rad(lat))[:, None]
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -88, 88], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                       linewidth=0.6, edgecolor="0.25")
        cf_snap = ax.contourf(lon_p, lat, field_w, levels=levels_snap,
                              cmap=CMAP, extend="both",
                              transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=11)
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-80, 81, 40), crs=ccrs.PlateCarree())
        ax.tick_params(labelsize=8)
    cax = fig.add_axes([0.907, 0.52, 0.013, 0.38])
    fig.colorbar(cf_snap, cax=cax).set_label("W m$^{-2}$")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(time_days, ts_raw, color="0.6", lw=0.8, label="raw")
    ax.plot(time_days, ts_smo, color="crimson", lw=1.6, label="Hoskins-filtered")
    ax.set_xlabel("days (January 2000)")
    ax.set_ylabel("W m$^{-2}$")
    ax.set_title(
        "(c) 6-hourly $\\cos\\phi\\,\\nabla\\cdot\\langle\\mathbf{v}m\\rangle$ "
        "at 45$^\\circ$N, 180$^\\circ$E",
        fontsize=11,
    )
    ax.legend(frameon=False)

    out = os.path.join(OUT_DIR, "R1_div_vm_raw_vs_smoothed.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    raw.close()
    smo.close()
    print("saved", out)


def _weight_cube(y_grid, stormtrack_lat12, nx):
    w = np.empty((12, y_grid.size, nx))
    for i in range(12):
        lat2d = np.tile((y_grid + stormtrack_lat12[i])[:, None], (1, nx))
        w[i] = EARTH_RADIUS * np.cos(np.deg2rad(lat2d)) * 2.0 * np.pi * 1e-15
    return w


def _stormtrack_lat12(hemisphere):
    """Monthly stormtrack latitude from the zonal-mean F_TE_final maximum."""
    from scipy import interpolate

    nc_flux = f"{ROOT}/cyclone_centered/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
    with xr.open_dataset(nc_flux) as f:
        lat = f["lat"].values
        fte = f["F_TE_final"][0].values
    fte_zon = np.nanmean(fte, axis=2)
    xd = np.arange(lat.size, dtype=float)
    x_hi = np.linspace(0, lat.size - 1, 25600)
    lat_hi = interpolate.interp1d(xd, lat)(x_hi)
    yd = np.linspace(0, 12, 12)
    fte_int = interpolate.RectBivariateSpline(yd, xd, fte_zon)(yd, x_hi)
    if hemisphere == "NH":
        return lat_hi[np.argmax(fte_int, axis=1)]
    return lat_hi[np.argmin(fte_int, axis=1)]


def figure_r2_za_composite_nh():
    """NH annual-mean cyclone-centered zonal MSE advection.

    Top row: pole-to-phi integrated transport form used in the manuscript
    (same class of quantity as the Fig 5-6 composites).
    Bottom row: local vertically integrated zonal MSE advection
    <u dm/dx> (W/m2), recovered as the meridional derivative of the
    composited integral.
    """
    st_lat = _stormtrack_lat12("NH")
    mean_lat = float(np.mean(st_lat))
    pw_factor = 2.0 * np.pi * EARTH_RADIUS / 1e15

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    lev_t = np.arange(-12, 13, 1)
    lev_local = np.arange(-300, 301, 25)

    handles = {}
    for k, (tag, title) in enumerate([("Intense", "intense cyclones"),
                                      ("Weak", "weak cyclones")]):
        with xr.open_dataset(
            f"{ROOT}/cyclone_centered/Composites_{tag}_NH_noleap.nc"
        ) as d:
            y = d["y"].values
            x = d["x"].values
            w = _weight_cube(y, st_lat, x.size)
            t_za = pw_factor * (d["composite_u_mse"].values / w).mean(axis=0)
            vo = d["composite_VO"].values.mean(axis=0)

        # local <u dm/dx> = d(T_ZA)/dphi / (2 pi a^2 cos(lat)), W/m2
        dphi = np.deg2rad(y[1] - y[0])
        coslat = np.cos(np.deg2rad(y + mean_lat))[:, None]
        local_za = (
            np.gradient(t_za * 1e15, axis=0) / dphi
            / (2.0 * np.pi * EARTH_RADIUS**2 * coslat)
        )

        for row, (field, levels, label) in enumerate([
            (t_za, lev_t,
             "$\\hat{T}_{c;\\mathrm{ZA}}^{L}$ (pole-to-$\\phi$ integral, PW)"),
            (local_za, lev_local,
             "$\\langle u\\,\\partial_x m\\rangle$ (local, W m$^{-2}$)"),
        ]):
            ax = axes[row, k]
            cf = ax.contourf(x, y, field, levels=levels, cmap=CMAP,
                             extend="both")
            cc = ax.contour(x, y, vo, levels=[1], colors="purple",
                            linewidths=1)
            ax.clabel(cc, fmt={1: "1 CVU"})
            ax.set_title(f"({chr(97 + 2 * row + k)}) {label}\nNH {title}",
                         fontsize=11)
            if row == 1:
                ax.set_xlabel("rlon")
            if k == 0:
                ax.set_ylabel("rlat (poleward positive)")
            handles[row] = cf

    fig.subplots_adjust(right=0.88, hspace=0.30, wspace=0.12)
    for row, unit in [(0, "PW"), (1, "W m$^{-2}$")]:
        cax = fig.add_axes([0.90, 0.55 - row * 0.42, 0.017, 0.32])
        fig.colorbar(handles[row], cax=cax).set_label(unit)
    fig.suptitle(
        "NH annual-mean cyclone-centered zonal MSE advection",
        y=0.96, fontweight="bold",
    )
    out = os.path.join(OUT_DIR, "R2_ZA_composite_NH.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def figure_r3_shf_anomaly_sh():
    """SH annual-mean cyclone-centered SHF (W/m2) and its meridional anomaly.

    Top row: the budget-residual surface heat flux F_SHF composited around
    intense / weak SH cyclones (composite_Shf_wm), with the 2-m temperature
    anomaly marking the warm and cold sectors.
    Bottom row: the meridional SHF anomaly F_SHF - mbar (mbar = instantaneous
    per-longitude meridional mean), i.e. the integrand of the poleward
    integral I_SHF whose composite is contoured in manuscript Figs 5d/6d.
    It is obtained exactly as the meridional derivative of the composite of
    the poleward integral (composite_Shf, PW), since compositing and
    differentiation commute:  S' = dF/dphi * 1e15 / (2 pi a^2 cos phi).
    The Fig. 5d/6d contours (-I_SHF / cos phi, annual mean) are overlaid.

    Display follows manuscript Figs 5-6: meridional axis reversed so the
    poleward side of the cyclone is on the positive y-axis.
    """
    import re

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    lev_full = np.arange(-200, 201, 10)
    lev_anom = np.arange(-200, 201, 10)
    lev_pw = np.arange(-20, 21, 4)

    handles = {}
    for k, (tag, title) in enumerate([("Intense", "intense cyclones"),
                                      ("Weak", "weak cyclones")]):
        with xr.open_dataset(
            f"{ROOT}/cyclone_centered/Composites_{tag}_SH_noleap.nc"
        ) as d:
            y = d["y"].values
            x = d["x"].values
            st = np.array([float(v) for v in
                           re.findall(r"-?\d+\.\d+", d.attrs["storm_track_latitudes"])])
            shf_wm = d["composite_Shf_wm"].values            # (12, y, x) W m-2
            i_shf = d["composite_Shf"].values                # (12, y, x) PW
            vo = d["composite_VO"].values.mean(axis=0)
            t2m = d["composite_T"].values.mean(axis=0)

        cosphi = np.cos(np.deg2rad(st[:, None, None] + y[None, :, None]))
        dphi = np.deg2rad(y[1] - y[0])
        # integrand of I_SHF (W m-2), per month then annual mean
        sprime = (np.gradient(i_shf * 1e15, axis=1) / dphi
                  / (2.0 * np.pi * EARTH_RADIUS ** 2 * cosphi)).mean(axis=0)
        # manuscript Fig 5d/6d field: -I_SHF / cos phi, annual mean
        p_shf = (-i_shf / cosphi).mean(axis=0)
        shf_ann = shf_wm.mean(axis=0)
        t2m_anom = t2m - t2m.mean(axis=1, keepdims=True)

        for row, (field, levels, label) in enumerate([
            (shf_ann, lev_full, "$F_{\\mathrm{SHF}}$ composite"),
            (sprime, lev_anom, "meridional anomaly $F_{\\mathrm{SHF}}-\\bar{m}$"),
        ]):
            ax = axes[row, k]
            cf = ax.contourf(x, y, field[::-1, :], levels=levels,
                             cmap=CMAP, extend="both")
            cc = ax.contour(x, y, vo[::-1, :], levels=[-1],
                            colors="purple", linewidths=1)
            ax.clabel(cc, fmt={-1: "1 CVU"})
            if row == 0:
                ct = ax.contour(x, y, t2m_anom[::-1, :],
                                levels=[-1.5, -1, -0.5, 0.5, 1, 1.5], colors="k",
                                linewidths=0.8, linestyles=["--"] * 3 + ["-"] * 3)
                ax.clabel(ct, fmt="%.1f K", fontsize=7)
            else:
                cp = ax.contour(x, y, p_shf[::-1, :], levels=lev_pw, colors="k",
                                linewidths=0.7)
                ax.clabel(cp, fmt="%d", fontsize=7)
            ax.set_title(f"{label}, SH {title}", fontsize=10.5)
            if row == 1:
                ax.set_xlabel("rlon")
            if k == 0:
                ax.set_ylabel("rlat (poleward positive)")
            ax.text(-0.12, 1.03, f"({chr(97 + 2 * row + k)})",
                    transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", clip_on=False)
            handles[row] = cf

    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.12)
    for row in range(2):
        cax = fig.add_axes([0.90, 0.55 - row * 0.42, 0.017, 0.32])
        fig.colorbar(handles[row], cax=cax).set_label("W m$^{-2}$")
    fig.suptitle(
        "SH annual-mean cyclone-centered surface heat flux "
        "(axis reversed as in manuscript Figs 5-6)",
        y=0.97, fontweight="bold",
    )
    out = os.path.join(OUT_DIR, "R3_SHF_composite_anomaly_SH.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def figure_r4_shf_remainder_nh():
    """NH DJF-JJA change of the surface heat flux, split by feature masks.

    Maps of the local field (W m-2): total, part sampled inside the
    cyclone+anticyclone masks, and the remainder (outside all tracked
    features), plus their NH zonal means. A final panel shows the
    poleward-integrated version (PW) of the same split, computed from the
    sampled budget fields as SHF = tot_energy + dhdt - OLR - SWABS (the
    exact quantity whose stormtrack value gives the Fig 3a,b SHF bars).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    djf = [11, 0, 1]
    jja = [5, 6, 7]

    nc = f"{ROOT}/cyclone_centered/WATTS_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
    with xr.open_dataset(nc) as d:
        lat = d["lat"].values
        lon = d["lon"].values

        def seas(v):
            x = d[v][0].values
            return np.nanmean(x[djf], axis=0) - np.nanmean(x[jja], axis=0)

        tot = seas("F_Shf_final")
        feat = seas("F_Shf_final_cycl") + seas("F_Shf_final_ant")
    rem = tot - feat

    nc_pw = f"{ROOT}/cyclone_centered/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
    with xr.open_dataset(nc_pw) as d:
        lat_pw = d["lat"].values

        def shf_zon(suffix):
            combo = (d[f"tot_energy_final{suffix}"][0].values
                     + d[f"F_Dhdt_final{suffix}"][0].values
                     - d[f"F_Olr_final{suffix}"][0].values
                     - d[f"F_Swabs_final{suffix}"][0].values)
            zon = np.nanmean(combo, axis=2)
            return np.mean(zon[djf], axis=0) - np.mean(zon[jja], axis=0)

        tot_pw = shf_zon("")
        feat_pw = shf_zon("_cycl") + shf_zon("_ant")
    rem_pw = tot_pw - feat_pw

    lon_p = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_p)
    lon_p = lon_p[order]

    levels = np.concatenate([np.arange(-100, -9, 10), np.arange(10, 101, 10)])

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.14,
                          left=0.06, right=0.89, top=0.88, bottom=0.07)
    specs = [
        (gs[0, 0], tot, "(a) Total $\\Delta$SHF (DJF$-$JJA)"),
        (gs[0, 1], feat, "(b) Cyclone + anticyclone part"),
        (gs[1, 0], rem, "(c) Remainder = (a) $-$ (b)"),
    ]
    for spec, field, title in specs:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6)
        cf = ax.contourf(lon_p, lat, field[:, order], levels=levels,
                         cmap=CMAP, extend="both",
                         transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=13)
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(0, 81, 20), crs=ccrs.PlateCarree())
        ax.tick_params(labelsize=8)
    cax = fig.add_axes([0.905, 0.55, 0.015, 0.33])
    fig.colorbar(cf, cax=cax).set_label("W m$^{-2}$")

    st_lat = float(np.mean(_stormtrack_lat12("NH")))
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.10)
    axz = fig.add_subplot(sub[0, 0])
    axp = fig.add_subplot(sub[0, 1], sharey=axz)
    nh = lat > 20
    nh_pw = lat_pw > 20
    for (field, field_pw), label, color in [
        ((tot, tot_pw), "total", "black"),
        ((feat, feat_pw), "cyclones + anticyclones", "crimson"),
        ((rem, rem_pw), "remainder", "royalblue"),
    ]:
        axz.plot(np.nanmean(field, axis=1)[nh], lat[nh], color=color,
                 lw=1.8, label=label)
        axp.plot(field_pw[nh_pw], lat_pw[nh_pw], color=color, lw=1.8)
    for ax in (axz, axp):
        ax.axvline(0, color="0.6", lw=0.8)
        ax.axhline(st_lat, color="0.4", lw=0.9, ls="--")
        ax.set_ylim(20, 90)
        ax.tick_params(labelsize=9)
    axz.set_xlabel("W m$^{-2}$", fontsize=10)
    axz.set_ylabel("latitude")
    axz.set_title("(d) NH zonal mean\nof (a)-(c)", fontsize=11)
    axz.legend(frameon=False, fontsize=8, loc="lower left")
    axp.set_xlabel("PW", fontsize=10)
    axp.set_title("(e) poleward-integrated\n$\\Delta I_{\\mathrm{SHF}}$ (as Fig. 3b)", fontsize=11)
    plt.setp(axp.get_yticklabels(), visible=False)

    fig.suptitle(
        "NH DJF$-$JJA change of the surface heat flux, split by feature masks",
        y=0.96, fontweight="bold", fontsize=15,
    )
    out = os.path.join(OUT_DIR, "R4_SHF_remainder_NH.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def figure_r4b_withint_shf_maps():
    """Check maps for R4 panel (e): the poleward-integrated SHF data.

    Maps the NH DJF-JJA change of the poleward-integrated SHF budget combo
    (tot_energy + dhdt - OLR - SWABS from the WITH_INT sampled file, i.e.
    the exact data behind the Fig 3a,b bars) for the total field, the
    cyclone+anticyclone part, and the remainder. The zonal means of these
    maps are the curves of R4 panel (e). Each gridpoint holds the
    integral accumulated from the south at that longitude, scaled so that
    the zonal mean is in PW. Diagnostic only; not in the response letter.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    djf = [11, 0, 1]
    jja = [5, 6, 7]

    nc_pw = f"{ROOT}/cyclone_centered/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
    with xr.open_dataset(nc_pw) as d:
        lat = d["lat"].values
        lon = d["lon"].values

        def combo2d(suffix):
            combo = (d[f"tot_energy_final{suffix}"][0].values
                     + d[f"F_Dhdt_final{suffix}"][0].values
                     - d[f"F_Olr_final{suffix}"][0].values
                     - d[f"F_Swabs_final{suffix}"][0].values)
            return (np.nanmean(combo[djf], axis=0)
                    - np.nanmean(combo[jja], axis=0))

        tot = combo2d("")
        feat = combo2d("_cycl") + combo2d("_ant")
    rem = tot - feat

    st_lat = float(np.mean(_stormtrack_lat12("NH")))

    lon_p = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_p)
    lon_p = lon_p[order]

    band = (lat > 20) & (lat < 90)
    vmax = np.nanpercentile(np.abs(tot[band]), 98)
    vmax = np.ceil(vmax * 2) / 2
    levels = np.linspace(-vmax, vmax, 21)

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.28, left=0.07, right=0.87,
                          top=0.90, bottom=0.05)
    specs = [
        (tot, "(a) Total $\\Delta I_{\\mathrm{SHF}}$ (DJF$-$JJA)"),
        (feat, "(b) Cyclone + anticyclone part"),
        (rem, "(c) Remainder = (a) $-$ (b)"),
    ]
    for row, (field, title) in enumerate(specs):
        ax = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6)
        cf = ax.contourf(lon_p, lat, field[:, order], levels=levels,
                         cmap=CMAP, extend="both",
                         transform=ccrs.PlateCarree())
        ax.plot([-180, 180], [st_lat, st_lat], ls="--", color="0.3",
                lw=0.9, transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=13)
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(0, 81, 20), crs=ccrs.PlateCarree())
        ax.tick_params(labelsize=8)
    cax = fig.add_axes([0.89, 0.20, 0.017, 0.55])
    fig.colorbar(cf, cax=cax).set_label(
        "PW (per-longitude value; zonal mean = R4e curves)")

    fig.suptitle(
        "Poleward-integrated SHF data behind the Fig. 3 bars:\n"
        "NH DJF$-$JJA change, split by feature masks (dashed: stormtrack latitude)",
        y=0.975, fontweight="bold", fontsize=14,
    )
    out = os.path.join(OUT_DIR, "R4b_SHF_withint_maps_NH.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure_r1_smoothing()
    figure_r2_za_composite_nh()
    figure_r3_shf_anomaly_sh()
    figure_r4_shf_remainder_nh()
    print("all reviewer figures done")
