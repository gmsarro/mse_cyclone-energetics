"""Check the SHF-residual pipeline against the stored Fig. 3 production data,
then composite every budget term separately around SH intense cyclones.

Part A (one month): rebuild tot_energy_final, F_Swabs_final, F_Olr_final,
F_Dhdt_final and F_Shf_final with the production formula and orientation of
cyclone_centered/make_TE_int.py and compare with the stored Integrated_Fluxes
file (r, rms, max |diff|), using both storage variants (tend_*_filtered.nc and
tend_*_filtered_2.nc). This pins down exactly what the Fig. 3 SHF contains.

Part B (2005-2009, same accepted snapshots as the paper's composites): +/-15 deg
cyclone-centered composites of each term of
    SHF_res = div_filt - (TSR-SSR) - TTR + tend_filt
plus the ERA5 surface fluxes for reference (turbulent -(SSHF+SLHF), surface
net longwave -STR, both into the atmosphere), the residual with the other tend
variant, and the residual with tend shifted by +/-1 time step (a timing
mismatch between div and tend would show up as an east-west dipole).
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
import scipy.integrate as integrate

BASE = "/project2/tas1/gmsarro"
ERA5 = "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
VINT_FMT = BASE + "/smoothed_vint/era5_vint_{y:04d}_{m:02d}_filtered.nc"
TEND1_FMT = BASE + "/smoothed_dh_dt_ERA5/tend_{y:04d}_{m:02d}_filtered.nc"
TEND2_FMT = BASE + "/smoothed_dh_dt_ERA5/tend_{y:04d}_{m:02d}_filtered_2.nc"
RAD_FMT = ERA5 + "/rad/era5_rad_{y:04d}_{m:02d}.6hrly.nc"
STF_FMT = ERA5 + "/stf/era5_stf_{y:04d}_{m:02d}.6hrly.nc"
INT_FMT = BASE + "/cyclone_centered/Integrated_TE/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
CSV = BASE + "/cyclone_centered/Composites_Intense_SH_noleap_center_samples.csv"
NOLEAP = BASE + "/cyclone_centered/Composites_Intense_SH_noleap.nc"
OUT_DIR = BASE + "/mse_cyclone-energetics/figures/reviewer_response"
NPZ = OUT_DIR + "/diag_budget_terms_SH.npz"

LV = 2.501e6           # as in make_TE_int.py
A_EARTH = 6.371e6
YEARS = range(2005, 2010)
R_WINDOW_DEG, DRES = 15.0, 0.25
NY = NX = int(round(2 * R_WINDOW_DEG / DRES))
LAT_REL = np.linspace(-R_WINDOW_DEG + DRES / 2, R_WINDOW_DEG - DRES / 2, NY)
LON_REL = np.linspace(-R_WINDOW_DEG + DRES / 2, R_WINDOW_DEG - DRES / 2, NX)

_VINT_MAPS = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]


def integrated_prod(x, lat):
    """Verbatim production integration (make_TE_int.py), vectorised over the
    trailing axes: x is (lat, ...) and lat the matching latitude array."""
    l = np.deg2rad(lat)
    w = np.cos(l)
    x = x - np.average(x, weights=w, axis=0)
    x = x * w.reshape((-1,) + (1,) * (x.ndim - 1))
    int_x = integrate.cumtrapz(x[::-1], l[::-1], axis=0, initial=None)
    int_x_r = integrate.cumtrapz(x, l, axis=0, initial=None)
    return 2 * np.pi * A_EARTH ** 2 * (int_x[::-1][1:] + int_x_r[:-1]) / 2 / 1e15


def _f32(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32))


def load_raw(yy, mm, nt_max=None):
    """Production-oriented (descending latitude, as make_TE_int.py) monthly
    stacks of every term, W/m2, on the 721x1440 grid."""
    out = {}
    with netCDF4.Dataset(VINT_FMT.format(y=yy, m=mm)) as ds:
        names = next(m for m in _VINT_MAPS if all(v in ds.variables for v in m.values()))
        lat_v = np.asarray(ds["latitude"][:])
        nt = ds[names["vigd"]].shape[0] if nt_max is None else min(nt_max, ds[names["vigd"]].shape[0])
        first = _f32(ds[names["vigd"]][:nt])
        second = _f32(ds[names["vimdf"]][:nt])
        third = _f32(ds[names["vithed"]][:nt])
    assert lat_v[0] < lat_v[-1], "smoothed vint expected ascending (production flips it)"
    out["div"] = (first + second * np.float32(LV) + third)[:, ::-1, :]
    del first, second, third
    for key, fmt in (("tend1", TEND1_FMT), ("tend2", TEND2_FMT)):
        with netCDF4.Dataset(fmt.format(y=yy, m=mm)) as ds:
            lat_t = np.asarray(ds["latitude"][:])
            assert lat_t[0] < lat_t[-1]
            out[key] = _f32(ds["tend_filtered"][:nt])[:, ::-1, :]
    with netCDF4.Dataset(RAD_FMT.format(y=yy, m=mm)) as ds:
        lat_r = np.asarray(ds["latitude"][:])
        assert lat_r[0] > lat_r[-1], "rad expected descending (production reads it as is)"
        tsr = _f32(ds["tsr"][:nt]) / 3600.0
        ssr = _f32(ds["ssr"][:nt]) / 3600.0
        out["olr"] = _f32(ds["ttr"][:nt]) / 3600.0
        out["lw_sfc"] = -_f32(ds["str"][:nt]) / 3600.0        # into the atmosphere
    out["swabs"] = tsr - ssr
    del tsr, ssr
    with netCDF4.Dataset(STF_FMT.format(y=yy, m=mm)) as ds:
        lat_s = np.asarray(ds["latitude"][:])
        assert lat_s[0] > lat_s[-1]
        out["stf"] = -(_f32(ds["sshf"][:nt]) + _f32(ds["slhf"][:nt])) / 3600.0
    out["lat_desc"] = lat_r
    return out


# --------------------------------------------------------------------------
def part_a(yy=2005, mm=7):
    print(f"=== Part A: rebuild Integrated_Fluxes_{yy}_{mm:02d} with the production code ===")
    with netCDF4.Dataset(INT_FMT.format(y=yy, m=mm)) as ds:
        stored = {k: np.asarray(ds[k][:]) for k in
                  ("tot_energy_final", "F_Swabs_final", "F_Olr_final", "F_Dhdt_final", "F_Shf_final")}
        lat_int = np.asarray(ds["lat"][:])
    nt = stored["F_Shf_final"].shape[0]
    raw = load_raw(yy, mm, nt_max=nt)
    lat = raw["lat_desc"]
    assert np.allclose(lat[1:-1], lat_int), "stored lat != production lat[1:-1]"

    def integ(field):        # (t, lat, lon) -> (t, lat-2, lon)
        x = np.moveaxis(field.astype(np.float64), 1, 0)               # (lat, t, lon)
        res = integrated_prod(x.reshape(x.shape[0], -1), lat)
        return np.moveaxis(res.reshape(-1, x.shape[1], x.shape[2]), 0, 1)

    def report(name, mine, ref):
        d = mine - ref
        r = np.corrcoef(mine.ravel(), ref.ravel())[0, 1]
        print(f"  {name:28s} r={r:.7f}  rms diff={np.sqrt(np.mean(d**2)):.4e} PW  "
              f"max|diff|={np.abs(d).max():.4e} PW  (ref rms {np.sqrt(np.mean(ref**2)):.3f})")

    report("tot_energy (div)", integ(raw["div"]), stored["tot_energy_final"])
    report("F_Swabs", integ(raw["swabs"]), stored["F_Swabs_final"])
    report("F_Olr", integ(raw["olr"]), stored["F_Olr_final"])
    report("F_Dhdt  [tend_filtered]", integ(raw["tend1"]), stored["F_Dhdt_final"])
    report("F_Dhdt  [tend_filtered_2]", integ(raw["tend2"]), stored["F_Dhdt_final"])
    for key in ("tend1", "tend2"):
        shf = raw["div"] - raw["swabs"] - raw["olr"] + raw[key]
        report(f"F_Shf   [{key}]", integ(shf), stored["F_Shf_final"])
    # sanity: derivative of the stored F_Shf recovers the mean-removed residual
    shf2 = raw["div"] - raw["swabs"] - raw["olr"] + raw["tend2"]
    l = np.deg2rad(lat)
    w = np.cos(l)
    mbar = np.average(shf2, weights=w, axis=1)                          # (t, lon)
    sprime = shf2 - mbar[:, None, :]
    dF = np.gradient(stored["F_Shf_final"] * 1e15, l[1:-1], axis=1)      # d/dphi
    implied = dF / (2 * np.pi * A_EARTH ** 2 * w[None, 1:-1, None])
    sel = slice(2, -2)
    rr = np.corrcoef(implied[:, sel].ravel(), sprime[:, 1:-1][:, sel].ravel())[0, 1]
    print(f"  dF_stored/dphi vs (SHF_res - mbar) pointwise: r={rr:.5f}")

    # premise check: monthly-mean residual vs ERA5 surface energy flux into the
    # atmosphere (turbulent + net LW), map and zonal mean, and pointwise 6-hourly
    era5_sfc = raw["stf"] + raw["lw_sfc"]
    mm_res, mm_era = shf2.mean(axis=0), era5_sfc.mean(axis=0)
    r_map = np.corrcoef(mm_res.ravel(), mm_era.ravel())[0, 1]
    print(f"  monthly-mean map  : r(SHF_res, ERA5 stf+LW)={r_map:.3f}  "
          f"rms(res-era5)={np.sqrt(np.mean((mm_res - mm_era) ** 2)):.1f} W/m2  "
          f"global means {np.average(mm_res, weights=np.broadcast_to(w[:, None], mm_res.shape)):+.1f} vs "
          f"{np.average(mm_era, weights=np.broadcast_to(w[:, None], mm_era.shape)):+.1f}")
    zr, ze = mm_res.mean(axis=1), mm_era.mean(axis=1)
    print("  zonal means (lat: res / era5) at 60S..30S:",
          [(int(lat[j]), round(float(zr[j])), round(float(ze[j]))) for j in range(0, 721, 30) if -62 < lat[j] < -28])
    r6h = np.corrcoef(shf2[:, 100:620].ravel(), era5_sfc[:, 100:620].ravel())[0, 1]
    print(f"  pointwise 6-hourly (65S-65N): r(SHF_res, ERA5 stf+LW)={r6h:.3f}  "
          f"std res={shf2[:, 100:620].std():.0f}  std era5={era5_sfc[:, 100:620].std():.0f} W/m2")
    sys.stdout.flush()


# --------------------------------------------------------------------------
def patch(arr2d_desc, lat_desc, lat0, lon0):
    """+/-15 deg window on the production (descending-lat) frame; returns the
    window with ascending relative latitude (south at row 0)."""
    tgt_lat = lat0 + LAT_REL
    ilat = np.clip(np.round((lat_desc[0] - tgt_lat) * 4).astype(int), 0, lat_desc.size - 1)
    ilon = (np.round(((lon0 + LON_REL) % 360.0) * 4).astype(int)) % 1440
    return arr2d_desc[ilat[:, None], ilon[None, :]]


TERMS = ("div", "tend1", "tend2", "swabs", "olr", "lw_sfc", "stf",
         "res2", "res1", "res2_lagm", "res2_lagp")


def part_b():
    print("=== Part B: term-by-term composites, SH intense, 2005-2009 ===")
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            yy, mm = int(r["year"]), int(r["month"])
            if yy in YEARS:
                lt = (int(r["day"]) - 1) * 4 + int(r["hour6"]) // 6
                rows.append((yy, mm, lt, float(r["lat"]), float(r["lon"])))
    print(f"  {len(rows)} snapshots")
    months = sorted({(yy, mm) for yy, mm, *_ in rows})
    sums = {k: np.zeros((12, NY, NX)) for k in TERMS}
    counts = np.zeros(12, dtype=int)
    for gi, (yy, mm) in enumerate(months):
        t0 = _time.time()
        raw = load_raw(yy, mm)
        lat = raw["lat_desc"]
        base = raw["div"] - raw["swabs"] - raw["olr"]
        raw["res2"] = base + raw["tend2"]
        raw["res1"] = base + raw["tend1"]
        lagm = np.roll(raw["tend2"], 1, axis=0)          # tend at t-1 paired with div at t
        lagp = np.roll(raw["tend2"], -1, axis=0)         # tend at t+1
        raw["res2_lagm"] = base + lagm
        raw["res2_lagp"] = base + lagp
        del base, lagm, lagp
        nt = raw["div"].shape[0]
        members = [r for r in rows if r[0] == yy and r[1] == mm]
        for _, _, lt, la, lo in members:
            lt = min(lt, nt - 1)
            if lt == 0 or lt == nt - 1:
                continue                                  # lag terms undefined at month ends
            for k in TERMS:
                sums[k][mm - 1] += patch(raw[k][lt], lat, la, lo)
            counts[mm - 1] += 1
        del raw
        print(f"  [{gi + 1}/{len(months)}] {yy}-{mm:02d} {len(members)} snaps {_time.time() - t0:.0f}s",
              flush=True)
    np.savez(NPZ, counts=counts, lat_rel=LAT_REL, lon_rel=LON_REL,
             **{k: sums[k] for k in TERMS})
    print("  saved", NPZ)


# --------------------------------------------------------------------------
def plot():
    d = np.load(NPZ)
    cnt = d["counts"].astype(float)
    ann = {k: (d[k].sum(axis=0) / cnt.sum())[::-1, :] for k in TERMS}   # poleward up (SH)
    xr_, yr_ = d["lon_rel"], -d["lat_rel"][::-1]
    with netCDF4.Dataset(NOLEAP) as ds:
        t2m = np.asarray(ds["composite_T"][:]).mean(axis=0)
        vo = np.asarray(ds["composite_VO"][:]).mean(axis=0)
        y15 = np.asarray(ds["y"][:])
        x15 = np.asarray(ds["x"][:])
    t2a = (t2m - t2m.mean(axis=1, keepdims=True))[::-1, :]
    vo_d = vo[::-1, :]
    p15 = -y15[::-1]

    rad = -ann["swabs"] - ann["olr"]
    closed = ann["stf"] + ann["lw_sfc"]
    panels = [
        ("(a) div_filt  [vigd + Lv vimdf + vithed]", ann["div"], 600),
        ("(b) tend_filt  [tend_filtered_2]", ann["tend2"], 600),
        ("(c) -(TSR-SSR) - TTR  (radiation)", rad, 600),
        ("(d) SHF_res = (a)+(b)+(c)  [production SHF]", ann["res2"], 600),
        ("(e) ERA5 turbulent -(SSHF+SLHF)", ann["stf"], 300),
        ("(f) ERA5 surface net LW  -STR", ann["lw_sfc"], 300),
        ("(g) (e)+(f): SHF if the budget closed", closed, 300),
        ("(h) non-closure = (d) - (g)", ann["res2"] - closed, 600),
        ("(i) SHF_res with tend_filtered (v1)", ann["res1"], 600),
        ("(j) SHF_res with tend at t-6h", ann["res2_lagm"], 600),
        ("(k) SHF_res with tend at t+6h", ann["res2_lagp"], 600),
        ("(l) tend v1 - tend v2", ann["tend1"] - ann["tend2"], 600),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 13.5))
    for ax, (title, field, vmax) in zip(axes.flat, panels):
        cf = ax.contourf(xr_, yr_, field, levels=np.linspace(-vmax, vmax, 25),
                         cmap="RdBu_r", extend="both")
        ax.contour(x15, p15, vo_d, levels=[-1], colors="purple", linewidths=1.0)
        ax.contour(x15, p15, t2a, levels=[-3, -1.5], colors="k", linewidths=0.7, linestyles="--")
        ax.contour(x15, p15, t2a, levels=[1.5, 3], colors="k", linewidths=0.7)
        ax.set_title(f"{title}\nmean {field.mean():+.0f}, E-W (x=+7 minus x=-7 at y=0) "
                     f"{field[NY // 2, np.argmin(np.abs(xr_ - 7))] - field[NY // 2, np.argmin(np.abs(xr_ + 7))]:+.0f} W/m2",
                     fontsize=9.5)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    fig.suptitle(f"SH intense cyclones 2005-2009 ({int(cnt.sum())} snapshots): annual-mean composites "
                 "of each budget term (W m$^{-2}$), production orientation and formula; poleward up",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR + "/diag_budget_terms_SH.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print("saved", out)
    print("annual-mean center / E / W values (W/m2):")
    ic, jc = NX // 2, NY // 2
    ie, iw = np.argmin(np.abs(xr_ - 7)), np.argmin(np.abs(xr_ + 7))
    for k in TERMS:
        f = ann[k]
        print(f"  {k:10s} C {f[jc, ic]:+8.1f}  E {f[jc, ie]:+8.1f}  W {f[jc, iw]:+8.1f}  mean {f.mean():+8.1f}")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        part_a()
        part_b()
    plot()
