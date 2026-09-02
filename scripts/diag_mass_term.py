"""Test whether the budget non-closure around cyclones is the column-mass
change term omitted from the storage computation.

The storage term is computed as the tendency of (cp T + Lv q) integrated over
fixed pressure levels masked with the monthly-mean surface pressure. The full
tendency of the column energy also contains the lower-boundary contribution
    M = h_s * (dp_s/dt) / g            (W/m2)
which is O(+/-300 W/m2) in the front/rear sectors of a moving, deepening
cyclone and vanishes in monthly means and in the footprint integral. With
S_comp ~ S_true - M, the residual SHF_res = div + S_comp - rad = SHF_true - M,
so the non-closure should be ~ -M: positive where p_s falls (ahead of the
cyclone), negative where it rises.

Composites M, dp_s/dt and h_s over the same SH intense snapshots (2005-2009)
as diag_budget_terms.py, and compares with the non-closure from that script.
"""

import csv
import os
import sys
import time as _time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

BASE = "/project2/tas1/gmsarro"
ERA5 = "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
PS_FMT = ERA5 + "/ps/era5_ps_{y:04d}_{m:02d}.6hrly.nc"
T2M_FMT = ERA5 + "/t2m/era5_t2m_{y:04d}_{m:02d}.6hrly.nc"
INT_FMT = BASE + "/cyclone_centered/Integrated_TE/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
CSV = BASE + "/cyclone_centered/Composites_Intense_SH_noleap_center_samples.csv"
NOLEAP = BASE + "/cyclone_centered/Composites_Intense_SH_noleap.nc"
OUT_DIR = BASE + "/mse_cyclone-energetics/figures/reviewer_response"
NPZ = OUT_DIR + "/diag_mass_term_SH.npz"
NPZ_B = OUT_DIR + "/diag_budget_terms_SH.npz"

CP, LV, G = 1004.0, 2.501e6, 9.81
DT = 6 * 3600.0
YEARS = range(2005, 2010)
R_WINDOW_DEG, DRES = 15.0, 0.25
NY = NX = int(round(2 * R_WINDOW_DEG / DRES))
LAT_REL = np.linspace(-R_WINDOW_DEG + DRES / 2, R_WINDOW_DEG - DRES / 2, NY)
LON_REL = np.linspace(-R_WINDOW_DEG + DRES / 2, R_WINDOW_DEG - DRES / 2, NX)


def patch(arr2d_desc, lat_desc, lat0, lon0):
    tgt_lat = lat0 + LAT_REL
    ilat = np.clip(np.round((lat_desc[0] - tgt_lat) * 4).astype(int), 0, lat_desc.size - 1)
    ilon = (np.round(((lon0 + LON_REL) % 360.0) * 4).astype(int)) % 1440
    return arr2d_desc[ilat[:, None], ilon[None, :]]


def q_sat(T, p):
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return 0.622 * es / (p - 0.378 * es)


def load_month(yy, mm):
    with netCDF4.Dataset(PS_FMT.format(y=yy, m=mm)) as ds:
        lat = np.asarray(ds["latitude"][:])
        assert lat[0] > lat[-1]
        ps = np.asarray(ds["sp"][:], dtype=np.float32)
    with netCDF4.Dataset(T2M_FMT.format(y=yy, m=mm)) as ds:
        assert np.asarray(ds["latitude"][:])[0] > 0
        t2m = np.asarray(ds["t2m"][:ps.shape[0]], dtype=np.float32)
    nt = min(ps.shape[0], t2m.shape[0])
    ps, t2m = ps[:nt], t2m[:nt]
    dpsdt = np.empty_like(ps)
    dpsdt[1:-1] = (ps[2:] - ps[:-2]) / (2 * DT)          # same stencil as storage.py
    dpsdt[0] = (ps[1] - ps[0]) / DT
    dpsdt[-1] = (ps[-1] - ps[-2]) / DT
    hs = CP * t2m + LV * 0.8 * q_sat(t2m, ps)
    M = hs * dpsdt / G
    # poleward integral of M exactly as the production integrates SHF (PW,
    # on lat[1:-1]); trailing axes vectorised
    x = np.moveaxis(M.astype(np.float64), 1, 0)
    FM = integrated_prod(x.reshape(x.shape[0], -1), lat)
    FM = np.moveaxis(FM.reshape(-1, x.shape[1], x.shape[2]), 0, 1).astype(np.float32)
    # the stored production F_Shf_final (PW) for the like-for-like composite
    with netCDF4.Dataset(INT_FMT.format(y=yy, m=mm)) as ds:
        FS = np.asarray(ds["F_Shf_final"][:nt], dtype=np.float32)
        assert np.allclose(np.asarray(ds["lat"][:]), lat[1:-1])
    return {"M": M, "dpsdt": dpsdt, "hs": hs, "FM": FM, "FS": FS}, lat


def integrated_prod(x, lat):
    """Verbatim production integration (make_TE_int.py), vectorised."""
    import scipy.integrate as integrate
    l = np.deg2rad(lat)
    w = np.cos(l)
    x = x - np.average(x, weights=w, axis=0)
    x = x * w.reshape((-1,) + (1,) * (x.ndim - 1))
    int_x = integrate.cumtrapz(x[::-1], l[::-1], axis=0, initial=None)
    int_x_r = integrate.cumtrapz(x, l, axis=0, initial=None)
    return 2 * np.pi * 6.371e6 ** 2 * (int_x[::-1][1:] + int_x_r[:-1]) / 2 / 1e15


def compute():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            yy, mm = int(r["year"]), int(r["month"])
            if yy in YEARS:
                lt = (int(r["day"]) - 1) * 4 + int(r["hour6"]) // 6
                rows.append((yy, mm, lt, float(r["lat"]), float(r["lon"])))
    months = sorted({(yy, mm) for yy, mm, *_ in rows})
    sums = {k: np.zeros((12, NY, NX)) for k in ("M", "dpsdt", "hs", "FM", "FS")}
    counts = np.zeros(12, dtype=int)
    for gi, (yy, mm) in enumerate(months):
        t0 = _time.time()
        data, lat = load_month(yy, mm)
        lat_int = lat[1:-1]
        nt = data["M"].shape[0]
        members = [r for r in rows if r[0] == yy and r[1] == mm]
        for _, _, lt, la, lo in members:
            lt = min(lt, nt - 1)
            if lt == 0 or lt == nt - 1:
                continue                                   # same exclusion as diag_budget_terms
            for k in sums:
                sums[k][mm - 1] += patch(data[k][lt], lat_int if k in ("FM", "FS") else lat, la, lo)
            counts[mm - 1] += 1
        print(f"[{gi + 1}/{len(months)}] {yy}-{mm:02d} {len(members)} snaps {_time.time() - t0:.0f}s", flush=True)
    np.savez(NPZ, counts=counts, **sums)
    print("saved", NPZ)


def plot():
    d = np.load(NPZ)
    n = d["counts"].sum()
    ann = {k: (d[k].sum(axis=0) / n)[::-1, :] for k in ("M", "dpsdt", "hs")}
    xr_, yr_ = LON_REL, -LAT_REL[::-1]
    with netCDF4.Dataset(NOLEAP) as ds:
        t2m = np.asarray(ds["composite_T"][:]).mean(axis=0)
        vo = np.asarray(ds["composite_VO"][:]).mean(axis=0)
        y15, x15 = np.asarray(ds["y"][:]), np.asarray(ds["x"][:])
    t2a = (t2m - t2m.mean(axis=1, keepdims=True))[::-1, :]
    vo_d = vo[::-1, :]
    p15 = -y15[::-1]

    panels = [("(c) dp_s/dt (hPa per hour)", ann["dpsdt"] * 36.0, 1.5),
              ("(d) -M = -h_s (dp_s/dt)/g  [omitted mass term, sign as it enters SHF_res]", -ann["M"], 500)]
    have_b = os.path.exists(NPZ_B)
    if have_b:
        b = np.load(NPZ_B)
        nb = b["counts"].sum()
        annb = lambda k: (b[k].sum(axis=0) / nb)[::-1, :]
        res1 = annb("res1")
        closed = annb("stf") + annb("lw_sfc")
        nonc = res1 - closed
        panels = [("(a) production SHF_res (tend v1)", res1, 500),
                  ("(b) non-closure = SHF_res - (ERA5 turb + sfc LW)", nonc, 500)] + panels + [
                  ("(e) SHF_res + M  (mass term restored)", res1 + ann["M"], 500),
                  ("(f) ERA5 turb + sfc LW (closed-budget SHF)", closed, 500)]
        inner = (np.abs(yr_) <= 12)[:, None] & (np.abs(xr_) <= 12)[None, :]
        r = np.corrcoef(nonc[inner], (-ann["M"])[inner])[0, 1]
        slope = np.polyfit((-ann["M"])[inner], nonc[inner], 1)[0]
        r2 = np.corrcoef((res1 + ann["M"])[inner], closed[inner])[0, 1]
        r0 = np.corrcoef(res1[inner], closed[inner])[0, 1]
        print(f"non-closure vs -M (|x|,|y|<=12): pattern r={r:.3f}, regression slope={slope:.2f}")
        print(f"SHF_res vs closed-budget SHF: r={r0:.3f};  after adding M: r={r2:.3f}")
        print(f"rms non-closure {nonc[inner].std():.0f} -> after M {(nonc - (-ann['M']))[inner].std():.0f} W/m2")
        print(f"footprint means: SHF_res {res1[inner].mean():+.1f}, closed {closed[inner].mean():+.1f}, "
              f"-M {(-ann['M'])[inner].mean():+.1f} W/m2")
    ncol = 3 if have_b else 2
    fig, axes = plt.subplots(2 if have_b else 1, ncol, figsize=(6.2 * ncol, 5.5 * (2 if have_b else 1)))
    for ax, (title, field, vmax) in zip(np.atleast_1d(axes).flat, panels):
        cf = ax.contourf(xr_, yr_, field, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both")
        ax.contour(x15, p15, vo_d, levels=[-1], colors="purple", linewidths=1.0)
        ax.contour(x15, p15, t2a, levels=[-3, -1.5], colors="k", linewidths=0.7, linestyles="--")
        ax.contour(x15, p15, t2a, levels=[1.5, 3], colors="k", linewidths=0.7)
        ax.set_title(title, fontsize=10)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    fig.suptitle(f"SH intense cyclones 2005-2009 ({int(n)} snapshots), annual mean, poleward up", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR + "/diag_mass_term_SH.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print("saved", out)

    # ---- PW: production F_SHF composite (black contours of Fig. 5d) vs F_M ----
    if "FM" not in d.files:
        return
    cnt = d["counts"].astype(float)
    with netCDF4.Dataset(NOLEAP) as ds:
        st = np.asarray(ds["stormtrack_lat"][:]) if "stormtrack_lat" in ds.variables else None
    lat_c = -50.0 if st is None else float(np.mean(st))
    cosw = np.cos(np.deg2rad(lat_c + LAT_REL))[:, None]

    def disp(k, months):
        sel = np.zeros(12, bool)
        sel[months] = True
        f = d[k][sel].sum(axis=0) / cnt[sel].sum()
        return (-f / cosw)[::-1, :]                     # paper convention: poleward positive, upside down

    jja, djf, all12 = [5, 6, 7], [11, 0, 1], list(range(12))
    rows_ = [("annual mean", all12), ("JJA - DJF", None)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for i, (lab, months) in enumerate(rows_):
        if months is not None:
            fs, fm = disp("FS", months), disp("FM", months)
        else:
            fs = disp("FS", jja) - disp("FS", djf)
            fm = disp("FM", jja) - disp("FM", djf)
        vmax = 20 if months is not None else 12
        for ax, (title, field) in zip(axes[i], [
                (f"production $F_{{SHF}}$ composite, {lab} (PW)", fs),
                (f"$F_M$: poleward-integrated mass term, {lab} (PW)", fm),
                (f"$F_{{SHF}} + F_M$ (mass term restored), {lab} (PW)", fs + fm)]):
            cf = ax.contourf(xr_, yr_, field, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both")
            ax.contour(x15, p15, vo_d, levels=[-1], colors="purple", linewidths=1.0)
            ax.contour(x15, p15, t2a, levels=[-3, -1.5], colors="k", linewidths=0.7, linestyles="--")
            ax.contour(x15, p15, t2a, levels=[1.5, 3], colors="k", linewidths=0.7)
            ic, jc = NX // 2, NY // 2
            ie, iw = np.argmin(np.abs(xr_ - 7)), np.argmin(np.abs(xr_ + 7))
            ax.set_title(f"{title}\nC {field[jc, ic]:+.1f}  E {field[jc, ie]:+.1f}  W {field[jc, iw]:+.1f}  "
                         f"mean {field.mean():+.2f}", fontsize=10)
            fig.colorbar(cf, ax=ax, shrink=0.85)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
        inner = (np.abs(yr_) <= 12)[:, None] & (np.abs(xr_) <= 12)[None, :]
        print(f"PW {lab}: pattern r(F_SHF, F_M) = {np.corrcoef(fs[inner], fm[inner])[0, 1]:.3f}; "
              f"footprint means F_SHF {fs[inner].mean():+.2f}, F_M {fm[inner].mean():+.2f}, "
              f"F_SHF+F_M {(fs + fm)[inner].mean():+.2f} PW; "
              f"E-W dipole F_SHF {fs[NY // 2, ie] - fs[NY // 2, iw]:+.1f}, F_M {fm[NY // 2, ie] - fm[NY // 2, iw]:+.1f}")
    fig.suptitle("SH intense cyclones 2005-2009: how much of the poleward-integrated SHF composite is the omitted mass term",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR + "/diag_mass_term_PW_SH.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        compute()
    plot()
