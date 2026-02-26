#!/usr/bin/env python3
"""
Figure 7–style cyclone-centred composites using the new composite NetCDF files.
SHF panels use composite_Shf_wm (W/m² budget residual, computed from energy_wm,
dhdt_wm, swabs_wm, olr_wm during compositing).  TE panels use the PW composite
converted to W/m^-1 via the weight cube (same as the notebook).

Produces one figure per hemisphere, saved as PNG.
Run on compute node, not login node.
"""
import os
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc

BASE = "/project2/tas1/gmsarro"
CYCLONE_DIR = os.path.join(BASE, "cyclone_centered")
OUT_DIR = os.path.join(BASE, "mse_cyclone-energetics", "notebooks")
FLUX_ASSIGN = os.path.join(CYCLONE_DIR, "WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc")

FILES = {
    "NH": {
        "weak":    os.path.join(CYCLONE_DIR, "Composites_Weak_NH_noleap.nc"),
        "intense": os.path.join(CYCLONE_DIR, "Composites_Intense_NH_noleap.nc"),
    },
    "SH": {
        "weak":    os.path.join(CYCLONE_DIR, "Composites_Weak_SH_noleap.nc"),
        "intense": os.path.join(CYCLONE_DIR, "Composites_Intense_SH_noleap.nc"),
    },
}

A_EARTH = 6.371e6


def compute_stormtrack_latitudes():
    with nc.Dataset(FLUX_ASSIGN) as ds:
        lat = ds["lat"][:]
        fte = ds["F_TE_final"][0, :, :, :]
    fte_zon = np.mean(fte, axis=2)
    nlat = len(lat)
    x_d = np.linspace(0, nlat - 1, nlat)
    y_d = np.linspace(0, 12, 12)
    x3_d = np.linspace(0, nlat - 1, 25600)
    lat_fine = scipy.interpolate.interp1d(x_d, lat)(x3_d)
    fte_interp = scipy.interpolate.RectBivariateSpline(y_d, x_d, fte_zon)(y_d, x3_d)
    return lat_fine[np.argmax(fte_interp, axis=1)], lat_fine[np.argmin(fte_interp, axis=1)]


def build_weight_cube(lat_row_base, maxpos12):
    lat_row_base = np.asarray(lat_row_base, float)
    maxpos12 = np.asarray(maxpos12, float).reshape(12)
    weight = np.empty((12, lat_row_base.shape[0], 120), float)
    for i in range(12):
        row_deg = lat_row_base + maxpos12[i]
        lat2d = np.tile(row_deg[:, None], (1, 120))
        weight[i] = A_EARTH * np.cos(np.deg2rad(lat2d)) * 2 * np.pi * 1e-15
    return weight


def monthly_scaler(cnt12):
    cnt12 = np.asarray(cnt12, float)
    tot = np.nansum(cnt12)
    return (12.0 * cnt12 / tot) if tot > 0 else np.ones(12, float)


def season_weighted(field_m, weights, months_1based):
    idx = [m - 1 for m in months_1based]
    return sum(weights[i] * field_m[i] for i in idx) / 3.0


def sym_levels(*arrays, n=31, pct=99):
    a = np.nanmax([np.nanpercentile(np.abs(x), pct) for x in arrays])
    a = float(a) if np.isfinite(a) and a > 0 else 1.0
    return np.linspace(-a, a, n), -a, a


def pos_levels(*arrays, n=9, pct=99):
    vals = [np.nanpercentile(x, pct) for x in arrays if np.any(np.isfinite(x))]
    a = float(np.nanmax(vals)) if vals else 1.0
    a = a if np.isfinite(a) and a > 0 else 1.0
    return np.linspace(0, a, n), 0, a


def nice_met_levels(zmin, zmax, interval=60.0):
    lo = interval * np.floor(zmin / interval)
    hi = interval * np.ceil(zmax / interval)
    nlev = max(4, int((hi - lo) / interval) + 1)
    return np.linspace(lo, hi, nlev)


def gray_outside_exact(ax, VO_ref, lon, lat, threshold, alpha=0.88):
    if threshold < 0:
        vmax = float(np.nanmax(VO_ref)) + 1.0
        ax.contourf(lon, lat, VO_ref, levels=[threshold, vmax],
                    colors=["#505050"], alpha=alpha, zorder=15)
        ax.contour(lon, lat, VO_ref, levels=[threshold],
                   colors="#2f2f2f", linewidths=1.6, zorder=20)
    else:
        vmin = float(np.nanmin(VO_ref)) - 1.0
        ax.contourf(lon, lat, VO_ref, levels=[vmin, threshold],
                    colors=["#505050"], alpha=alpha, zorder=15)
        ax.contour(lon, lat, VO_ref, levels=[threshold],
                   colors="#2f2f2f", linewidths=1.6, zorder=20)


def masked_longitudinal_mean(field2d, mask2d):
    fld = np.where(mask2d, field2d, np.nan)
    with np.errstate(all='ignore'):
        return np.nanmean(fld, axis=1)


def lat_band_from_mask(mask2d, lat):
    rows = np.where(mask2d.any(axis=1))[0]
    if rows.size == 0:
        return None
    ylo, yhi = float(lat[rows[0]]), float(lat[rows[-1]])
    if ylo > yhi:
        ylo, yhi = yhi, ylo
    return ylo, yhi


def build_fig7(hemisphere, stormtrack_nh, stormtrack_sh):
    paths = FILES[hemisphere]
    stormtrack = stormtrack_sh if hemisphere == "SH" else stormtrack_nh
    vo_threshold = -1.0 if hemisphere == "SH" else 1.0

    with nc.Dataset(paths["intense"], "r") as dsi, nc.Dataset(paths["weak"], "r") as dsw:
        x = np.asarray(dsw["x"][:])
        y = np.asarray(dsw["y"][:])
        cnt_int  = np.asarray(dsi["count"][:], dtype=float)
        cnt_weak = np.asarray(dsw["count"][:], dtype=float)

        # PW fields (TE only — used for TE panels via weight conversion)
        TE_int   = np.asarray(dsi["composite_TE"][:])
        TE_weak  = np.asarray(dsw["composite_TE"][:])

        # W/m² fields: SHF (budget residual), VO, Z
        SHF_wm_int  = np.asarray(dsi["composite_Shf_wm"][:])
        SHF_wm_weak = np.asarray(dsw["composite_Shf_wm"][:])
        VO_int   = np.asarray(dsi["composite_VO"][:])
        VO_weak  = np.asarray(dsw["composite_VO"][:])
        Z_int    = np.asarray(dsi["composite_Z"][:])
        Z_weak   = np.asarray(dsw["composite_Z"][:])

    # TE: PW → W m^-1 (same as notebook, for TE panels)
    wgt = build_weight_cube(y, stormtrack)
    TE_int_Wm  = TE_int  / wgt
    TE_weak_Wm = TE_weak / wgt

    w_int  = monthly_scaler(cnt_int)
    w_weak = monthly_scaler(cnt_weak)
    DJF, JJA = [12, 1, 2], [6, 7, 8]
    SCALE_TE = 1e8

    # DJF−JJA for TE (in 1e8 W m^-1 for map display)
    B_te  = season_weighted(TE_int_Wm,  w_int,  DJF) - season_weighted(TE_int_Wm,  w_int,  JJA)
    F_te  = season_weighted(TE_weak_Wm, w_weak, DJF) - season_weighted(TE_weak_Wm, w_weak, JJA)
    B_te_map = B_te / SCALE_TE
    F_te_map = F_te / SCALE_TE

    # DJF−JJA for SHF (already W/m²)
    A_shf = season_weighted(SHF_wm_int,  w_int,  DJF) - season_weighted(SHF_wm_int,  w_int,  JJA)
    E_shf = season_weighted(SHF_wm_weak, w_weak, DJF) - season_weighted(SHF_wm_weak, w_weak, JJA)

    # VO and Z
    VO_mean_int  = 0.5 * (season_weighted(VO_int,  w_int,  DJF) + season_weighted(VO_int,  w_int,  JJA))
    VO_mean_weak = 0.5 * (season_weighted(VO_weak, w_weak, DJF) + season_weighted(VO_weak, w_weak, JJA))
    Z_avg_int  = 0.5 * ((Z_int[11]  + Z_int[0]  + Z_int[1])  / 3.0 + np.mean(Z_int[5:8],  axis=0))
    Z_avg_weak = 0.5 * ((Z_weak[11] + Z_weak[0] + Z_weak[1]) / 3.0 + np.mean(Z_weak[5:8], axis=0))

    # Annual-mean SHF (W/m²) for col 3
    D_shf = np.nansum(SHF_wm_int  * cnt_int[:,  None, None], axis=0) / max(np.nansum(cnt_int),  1)
    G_shf = np.nansum(SHF_wm_weak * cnt_weak[:, None, None], axis=0) / max(np.nansum(cnt_weak), 1)

    # Masks
    if hemisphere == "SH":
        mask_int  = (VO_mean_int  < vo_threshold)
        mask_weak = (VO_mean_weak < vo_threshold)
    else:
        mask_int  = (VO_mean_int  > vo_threshold)
        mask_weak = (VO_mean_weak > vo_threshold)

    # Zonal-mean profiles (within cyclone mask)
    C_shf = masked_longitudinal_mean(A_shf, mask_int)
    C_te  = masked_longitudinal_mean(B_te,  mask_int)
    H_shf = masked_longitudinal_mean(E_shf, mask_weak)
    H_te  = masked_longitudinal_mean(F_te,  mask_weak)
    band_int  = lat_band_from_mask(mask_int,  y)
    band_weak = lat_band_from_mask(mask_weak, y)

    # Contour levels
    lev_shf_diff, vmin_shf, vmax_shf = sym_levels(A_shf, E_shf, n=31, pct=99)
    lev_te_diff, vmin_te, vmax_te = sym_levels(B_te_map, F_te_map, n=31, pct=99)
    lev_shf_ann, v0_s, v1_s = pos_levels(D_shf, G_shf, n=9, pct=99)
    zmin = float(np.nanmin([np.nanmin(Z_avg_int), np.nanmin(Z_avg_weak)]))
    zmax = float(np.nanmax([np.nanmax(Z_avg_int), np.nanmax(Z_avg_weak)]))
    levels_Z = nice_met_levels(zmin, zmax, interval=60.0)

    # ── Plot ─────────────────────────────────────────────────────────────
    mpl.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    X, Y = np.meshgrid(x, y)
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))
    plt.subplots_adjust(wspace=0.30, hspace=0.32, left=0.18, right=0.90, top=0.93, bottom=0.10)
    ctr_lw, lab_fs = 2.0, 12

    def storm_lat_line(ax):
        ax.axhline(0, color="k", lw=1.2, ls="--", zorder=40)

    # === Row 0: 6+ CVU ===
    # (a) DJF−JJA SHF in W/m²
    ax = axs[0, 0]
    im_shf = ax.contourf(X, Y, A_shf, levels=lev_shf_diff, cmap="bwr",
                          vmin=vmin_shf, vmax=vmax_shf, extend="both", zorder=1)
    gray_outside_exact(ax, VO_mean_int, x, y, vo_threshold)
    csZ = ax.contour(x, y, Z_avg_int, levels=levels_Z, colors="white", linewidths=ctr_lw, zorder=25)
    ax.clabel(csZ, fmt="%1.0f", fontsize=lab_fs, colors="white", inline=0.01)
    ax.plot(0, 0, "bx", ms=7, mew=2, zorder=35); storm_lat_line(ax)
    ax.set_title(r"(a) $\Delta I_{\mathrm{SHF}}^L$, 6+ CVU")
    ax.set_xlabel("Lon from center (\u00b0)"); ax.set_ylabel("Lat from center (\u00b0)")

    # (b) DJF−JJA TE (1e8 W m^-1)
    ax = axs[0, 1]
    im_te = ax.contourf(X, Y, B_te_map, levels=lev_te_diff, cmap="bwr",
                         vmin=vmin_te, vmax=vmax_te, extend="both", zorder=1)
    gray_outside_exact(ax, VO_mean_int, x, y, vo_threshold)
    csZ = ax.contour(x, y, Z_avg_int, levels=levels_Z, colors="white", linewidths=ctr_lw, zorder=25)
    ax.clabel(csZ, fmt="%1.0f", fontsize=lab_fs, colors="white", inline=0.01)
    ax.plot(0, 0, "bx", ms=7, mew=2, zorder=35); storm_lat_line(ax)
    ax.set_title(r"(b) $\Delta I^L$, 6+ CVU")
    ax.set_xlabel("Lon from center (\u00b0)")

    # (c) Zonal mean: TE and SHF (same panel)
    ax_c1 = axs[0, 2]
    ax_c1.set_ylim(y.min(), y.max())
    ax_c1.axvline(0, color="k", lw=1.2)
    l1, = ax_c1.plot(C_te / SCALE_TE, y, color="#1f4e79", lw=2.8, label=r"$\Delta I^L$")
    ax_c1.set_xlabel(r"$\Delta I^L$: $10^8$ W m$^{-1}$", color="#1f4e79")
    ax_c1.tick_params(axis="x", labelcolor="#1f4e79")
    ax_c1_twin = ax_c1.twiny()
    l2, = ax_c1_twin.plot(C_shf, y, color="#b22222", lw=2.8, label=r"$\Delta I_{\mathrm{SHF}}^L$")
    ax_c1_twin.set_xlabel(r"$\Delta I_{\mathrm{SHF}}^L$: W m$^{-2}$", color="#b22222")
    ax_c1_twin.tick_params(axis="x", labelcolor="#b22222")
    storm_lat_line(ax_c1)
    if band_int is not None:
        ax_c1.axhspan(y.min(), band_int[0], color="#d0d0d0", zorder=0)
        ax_c1.axhspan(band_int[1], y.max(), color="#d0d0d0", zorder=0)
    ax_c1.set_title(r"(c) Zonal mean $\Delta I^L$, $\Delta I_{\mathrm{SHF}}^L$, 6+ CVU")
    ax_c1.legend([l1, l2], [r"$\Delta I^L$", r"$\Delta I_{\mathrm{SHF}}^L$"], frameon=False, fontsize=11)

    # (d) Annual-mean SHF (W/m², no mask)
    ax = axs[0, 3]
    im_ann = ax.contourf(X, Y, D_shf, levels=lev_shf_ann, cmap="afmhot_r",
                          vmin=v0_s, vmax=v1_s, extend="max")
    csZ = ax.contour(x, y, Z_avg_int, levels=levels_Z, colors="white", linewidths=ctr_lw)
    ax.clabel(csZ, fmt="%1.0f", fontsize=lab_fs, colors="white", inline=0.01)
    ax.contour(x, y, VO_mean_int, levels=[vo_threshold], colors="k", linewidths=ctr_lw)
    ax.plot(0, 0, "bx", ms=7, mew=2); storm_lat_line(ax)
    ax.set_title(r"(d) Annual $I_{\mathrm{SHF}}^L$, 6+ CVU")
    ax.set_xlabel("Lon from center (\u00b0)")

    # === Row 1: 1–5 CVU ===
    ax = axs[1, 0]
    ax.contourf(X, Y, E_shf, levels=lev_shf_diff, cmap="bwr",
                vmin=vmin_shf, vmax=vmax_shf, extend="both", zorder=1)
    gray_outside_exact(ax, VO_mean_weak, x, y, vo_threshold)
    csZ = ax.contour(x, y, Z_avg_weak, levels=levels_Z, colors="white", linewidths=ctr_lw, zorder=25)
    ax.clabel(csZ, fmt="%1.0f", fontsize=lab_fs, colors="white", inline=0.01)
    ax.plot(0, 0, "bx", ms=7, mew=2, zorder=35); storm_lat_line(ax)
    ax.set_title(r"(e) $\Delta I_{\mathrm{SHF}}^L$, 1–5 CVU")
    ax.set_xlabel("Lon from center (\u00b0)"); ax.set_ylabel("Lat from center (\u00b0)")

    ax = axs[1, 1]
    ax.contourf(X, Y, F_te_map, levels=lev_te_diff, cmap="bwr",
                vmin=vmin_te, vmax=vmax_te, extend="both", zorder=1)
    gray_outside_exact(ax, VO_mean_weak, x, y, vo_threshold)
    csZ = ax.contour(x, y, Z_avg_weak, levels=levels_Z, colors="white", linewidths=ctr_lw, zorder=25)
    ax.clabel(csZ, fmt="%1.0f", fontsize=lab_fs, colors="white", inline=0.01)
    ax.plot(0, 0, "bx", ms=7, mew=2, zorder=35); storm_lat_line(ax)
    ax.set_title(r"(f) $\Delta I^L$, 1–5 CVU")
    ax.set_xlabel("Lon from center (\u00b0)")

    ax_g1 = axs[1, 2]
    ax_g1.set_ylim(y.min(), y.max())
    ax_g1.axvline(0, color="k", lw=1.2)
    l3, = ax_g1.plot(H_te / SCALE_TE, y, color="#1f4e79", lw=2.8, label=r"$\Delta I^L$")
    ax_g1.set_xlabel(r"$\Delta I^L$: $10^8$ W m$^{-1}$", color="#1f4e79")
    ax_g1.tick_params(axis="x", labelcolor="#1f4e79")
    ax_g1_twin = ax_g1.twiny()
    l4, = ax_g1_twin.plot(H_shf, y, color="#b22222", lw=2.8, label=r"$\Delta I_{\mathrm{SHF}}^L$")
    ax_g1_twin.set_xlabel(r"$\Delta I_{\mathrm{SHF}}^L$: W m$^{-2}$", color="#b22222")
    ax_g1_twin.tick_params(axis="x", labelcolor="#b22222")
    storm_lat_line(ax_g1)
    if band_weak is not None:
        ax_g1.axhspan(y.min(), band_weak[0], color="#d0d0d0", zorder=0)
        ax_g1.axhspan(band_weak[1], y.max(), color="#d0d0d0", zorder=0)
    ax_g1.set_title(r"(g) Zonal mean $\Delta I^L$, $\Delta I_{\mathrm{SHF}}^L$, 1–5 CVU")
    ax_g1.legend([l3, l4], [r"$\Delta I^L$", r"$\Delta I_{\mathrm{SHF}}^L$"], frameon=False, fontsize=11)

    ax = axs[1, 3]
    ax.contourf(X, Y, G_shf, levels=lev_shf_ann, cmap="afmhot_r",
                vmin=v0_s, vmax=v1_s, extend="max")
    csZ = ax.contour(x, y, Z_avg_weak, levels=levels_Z, colors="white", linewidths=ctr_lw)
    ax.clabel(csZ, fmt="%1.0f", fontsize=lab_fs, colors="white", inline=0.01)
    ax.contour(x, y, VO_mean_weak, levels=[vo_threshold], colors="k", linewidths=ctr_lw)
    ax.plot(0, 0, "bx", ms=7, mew=2); storm_lat_line(ax)
    ax.set_title(r"(h) Annual $I_{\mathrm{SHF}}^L$, 1–5 CVU")
    ax.set_xlabel("Lon from center (\u00b0)")

    # Shared axis limits for zonal-mean panels
    te_vals = np.concatenate([C_te[np.isfinite(C_te)], H_te[np.isfinite(H_te)]]) / SCALE_TE
    if te_vals.size > 0:
        xlim_te = np.max(np.abs(te_vals)) * 1.1
        ax_c1.set_xlim(-xlim_te, xlim_te)
        ax_g1.set_xlim(-xlim_te, xlim_te)
    shf_vals = np.concatenate([C_shf[np.isfinite(C_shf)], H_shf[np.isfinite(H_shf)]])
    if shf_vals.size > 0:
        xlim_shf = np.max(np.abs(shf_vals)) * 1.1
        ax_c1_twin.set_xlim(-xlim_shf, xlim_shf)
        ax_g1_twin.set_xlim(-xlim_shf, xlim_shf)

    # Colorbars
    cax_shf = fig.add_axes([0.04, 0.12, 0.018, 0.76])
    cb_shf = fig.colorbar(im_shf, cax=cax_shf)
    cb_shf.set_label(r"$\Delta I_{\mathrm{SHF}}^L$ (W m$^{-2}$)", rotation=90, labelpad=18, fontsize=16)
    cb_shf.ax.yaxis.set_ticks_position("left")
    cb_shf.ax.yaxis.set_label_position("left")
    cb_shf.ax.tick_params(labelsize=13, length=5, width=1.2)

    cax_te = fig.add_axes([0.295, 0.02, 0.15, 0.018])
    cb_te = fig.colorbar(im_te, cax=cax_te, orientation="horizontal")
    cb_te.set_label(r"$\Delta I^L$ ($10^8$ W m$^{-1}$)", fontsize=14)
    cb_te.ax.tick_params(labelsize=12)

    cax_ann = fig.add_axes([0.95, 0.12, 0.018, 0.76])
    cb_ann = fig.colorbar(im_ann, cax=cax_ann)
    cb_ann.set_label(r"$I_{\mathrm{SHF}}^L$ (W m$^{-2}$)", rotation=270, labelpad=18, fontsize=16)
    cb_ann.ax.tick_params(labelsize=13, length=5, width=1.2)

    fig.suptitle(f"Figure 7 \u2014 Cyclone-centred composites ({hemisphere})", fontsize=18, y=0.98)
    out_path = os.path.join(OUT_DIR, f"fig7_composites_{hemisphere}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    stormtrack_nh, stormtrack_sh = compute_stormtrack_latitudes()
    print(f"Storm-track NH: {stormtrack_nh}")
    print(f"Storm-track SH: {stormtrack_sh}")
    for hemi in ["SH", "NH"]:
        build_fig7(hemi, stormtrack_nh, stormtrack_sh)
    print("Done.")


if __name__ == "__main__":
    main()
