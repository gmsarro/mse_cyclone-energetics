"""Figures for the response to reviewers (not part of the manuscript).

R1: raw vs Hoskins-filtered 6-hourly meridional MSE flux divergence.
R2: NH cyclone-centered annual-mean zonal MSE advection composite.
R3: SH cyclone-centered annual-mean SHF composite and its zonal anomaly.

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
    """Raw vs Hoskins-filtered 6-hourly divergence of meridional MSE flux."""
    raw = xr.open_dataset(f"{ROOT}/VM_adj_ERA5/Aaron_VM_2000_01.nc")
    smo = xr.open_dataset(f"{ROOT}/smoothed_div_VM/Aaron_VM_2000_01_filtered.nc")

    raw_te = raw["TE"]
    smo_te = smo["TE_filtered"]

    t0 = 0
    lat = raw["latitude"].values
    lon = raw["longitude"].values

    # Time series at a NH stormtrack point (45N, 180E).
    ilat = int(np.argmin(np.abs(lat - 45.0)))
    ilon = int(np.argmin(np.abs(lon - 180.0)))
    ts_raw = raw_te[:, ilat, ilon].values
    ts_smo = smo_te[:, ilat, ilon].values
    time_days = np.arange(ts_raw.size) * 0.25

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.32, wspace=0.12)

    levels = np.arange(-8000, 8001, 800)
    for k, (field, title) in enumerate([
        (raw_te[t0].values, "raw ERA5 (0.25$^\\circ$)"),
        (smo_te[t0].values, "Hoskins-filtered ($n_0=60$, $r=1$)"),
    ]):
        ax = fig.add_subplot(gs[0, k])
        cf = ax.contourf(lon, lat, field, levels=levels, cmap=CMAP, extend="both")
        ax.set_title(
            f"({chr(97 + k)}) $\\nabla\\cdot\\langle vm\\rangle$ "
            f"2000-01-01 00UTC\n{title}",
            fontsize=11,
        )
        ax.set_xlabel("longitude")
        if k == 0:
            ax.set_ylabel("latitude")
    cax = fig.add_axes([0.92, 0.55, 0.015, 0.32])
    fig.colorbar(cf, cax=cax).set_label("W m$^{-2}$")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(time_days, ts_raw, color="0.6", lw=0.8, label="raw")
    ax.plot(time_days, ts_smo, color="crimson", lw=1.6, label="Hoskins-filtered")
    ax.set_xlabel("days (January 2000)")
    ax.set_ylabel("W m$^{-2}$")
    ax.set_title(
        "6-hourly $\\nabla\\cdot\\langle vm\\rangle$ at 45$^\\circ$N, 180$^\\circ$E",
        fontsize=11,
    )
    ax.legend(frameon=False)
    ax.set_title("(c) " + ax.get_title(), fontsize=11)

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
    """SH annual-mean SHF composite (W/m2) and its zonal anomaly.

    Display follows manuscript Figs 5-6: meridional axis reversed so the
    poleward side of the cyclone is on the positive y-axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    lev_full = np.arange(-200, 201, 10)
    lev_anom = np.arange(-60, 61, 5)

    handles = {}
    for k, (tag, title) in enumerate([("Intense", "intense cyclones"),
                                      ("Weak", "weak cyclones")]):
        with xr.open_dataset(
            f"{ROOT}/cyclone_centered/Composites_{tag}_SH_noleap.nc"
        ) as d:
            y = d["y"].values
            x = d["x"].values
            shf = d["composite_Shf_wm"].values.mean(axis=0)
            vo = d["composite_VO"].values.mean(axis=0)

        shf_anom = shf - shf.mean(axis=1, keepdims=True)

        for row, (field, levels, label) in enumerate([
            (shf, lev_full, "$\\widehat{\\mathrm{SHF}}$"),
            (shf_anom, lev_anom, "$\\widehat{\\mathrm{SHF}}^{*}$ (zonal anomaly)"),
        ]):
            ax = axes[row, k]
            cf = ax.contourf(x, y, field[::-1, :], levels=levels,
                             cmap=CMAP, extend="both")
            cc = ax.contour(x, y, vo[::-1, :], levels=[-1],
                            colors="purple", linewidths=1)
            ax.clabel(cc, fmt={-1: "1 CVU"})
            ax.set_title(f"{label}, SH {title}", fontsize=11)
            if row == 1:
                ax.set_xlabel("rlon")
            if k == 0:
                ax.set_ylabel("rlat (poleward positive)")
            ax.text(-0.03, 1.03, f"({chr(97 + 2 * row + k)})",
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
    """NH DJF-JJA change of the local surface heat flux (W m-2).

    Maps of the total field, the part sampled inside cyclone+anticyclone
    masks, and the remainder (outside all tracked features), plus their NH
    zonal means. Same sampled fields (intensity cut 1 CVU) used for the
    manuscript budgets, before poleward integration.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    nc = f"{ROOT}/cyclone_centered/WATTS_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
    with xr.open_dataset(nc) as d:
        lat = d["lat"].values
        lon = d["lon"].values
        djf = [11, 0, 1]
        jja = [5, 6, 7]

        def seas(v):
            x = d[v][0].values
            return np.nanmean(x[djf], axis=0) - np.nanmean(x[jja], axis=0)

        tot = seas("F_Shf_final")
        feat = seas("F_Shf_final_cycl") + seas("F_Shf_final_ant")
    rem = tot - feat

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

    axz = fig.add_subplot(gs[1, 1])
    nh = lat > 0
    for field, label, color in [
        (tot, "total", "black"),
        (feat, "cyclones + anticyclones", "crimson"),
        (rem, "remainder", "royalblue"),
    ]:
        axz.plot(np.nanmean(field, axis=1)[nh], lat[nh], color=color,
                 lw=1.8, label=label)
    axz.axvline(0, color="0.6", lw=0.8)
    axz.set_ylim(0, 90)
    axz.set_xlabel("W m$^{-2}$")
    axz.set_ylabel("latitude")
    axz.set_title("(d) NH zonal mean of (a)-(c)", fontsize=13)
    axz.legend(frameon=False, fontsize=10, loc="lower left")

    fig.suptitle(
        "NH DJF$-$JJA change of the surface heat flux, split by feature masks",
        y=0.96, fontweight="bold", fontsize=15,
    )
    out = os.path.join(OUT_DIR, "R4_SHF_remainder_NH.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure_r1_smoothing()
    figure_r2_za_composite_nh()
    figure_r3_shf_anomaly_sh()
    figure_r4_shf_remainder_nh()
    print("all reviewer figures done")
