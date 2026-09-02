"""Stage 4 of the corrected-storage chain: re-sample the storage-dependent
poleward-flux fields inside cyclone / anticyclone masks, reproducing
cyclone_centered/assign_fluxes_intensity.py (the producer of
WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc) for

    F_Dhdt_final{,_cycl,_ant}, F_Shf_final{,_cycl,_ant}   (PW, 719 x 1440)

with F_Dhdt_final / F_Shf_final read from Integrated_TE_v3, plus the local
(W/m2, 721 x 1440) residual surface flux SHF = div - SWabs - OLR + storage_v3
sampled the same way (replacement for the WATTS_* file used in the response
letter), plus tot_energy_final{,_cycl} as an exact-reproduction check against
the stored WITH_INT file.

Algorithm (verbatim from the original): for every month, masks flag_C/flag_A
(1.5 deg TRACK grid) are zeroed where the feature intensity is below the cut
(1..6 CVU), concatenated SH+NH and flipped to descending latitude, bilinearly
interpolated onto the 0.25 deg grid (interp2d linear == tensor-product linear
interpolation, implemented here with 1-D weight matrices), and the flux field
is zeroed where the interpolated mask < 0.5 (rows 1:-1). Monthly means are
averaged over 2000-2014 (no-leap: first 112 February steps).

One task per calendar month:  python resample_with_int_v3.py MONTH
Merge + write:                 python resample_with_int_v3.py --merge
"""

import os
import sys
import time as _time

import netCDF4
import numpy as np

BASE = "/project2/tas1/gmsarro"
ERA5 = "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
CC = BASE + "/cyclone_centered"
V3_FMT = CC + "/Integrated_TE_v3/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
V1_FMT = CC + "/Integrated_TE/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
MASK_FMT = CC + "/masks/{h}0.225/MASK_{h}_{y:04d}.nc"
VINT_FMT = BASE + "/smoothed_vint/era5_vint_{y:04d}_{m:02d}_filtered.nc"
DHDT_FMT = BASE + "/smoothed_dh_dt_ERA5_v3/tend_{y:04d}_{m:02d}_filtered_3.nc"
RAD_FMT = ERA5 + "/rad/era5_rad_{y:04d}_{m:02d}.6hrly.nc"
WITH_INT = CC + "/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
OUT_WITH_INT = CC + "/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225_v3.nc"
OUT_WATTS = CC + "/WATTS_v3_Cyclones_Sampled_SHF_0.225.nc"
PART_DIR = CC + "/resample_v3_parts"

YEARS = range(2000, 2015)
CUTS = np.array([1, 2, 3, 4, 5, 6])
MONTHNUM = np.array([0, 124, 236, 360, 480, 604, 724, 848, 972, 1092, 1216, 1336, 1460])
MONTH_IND = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 4
LV = 2.501e6
_VINT_MAPS = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]


def linear_weights(src, dst):
    """(len(dst), len(src)) matrix of 1-D linear interpolation weights."""
    W = np.zeros((dst.size, src.size))
    pos = np.clip(np.searchsorted(src, dst), 1, src.size - 1)
    x0, x1 = src[pos - 1], src[pos]
    f = (dst - x0) / (x1 - x0)
    W[np.arange(dst.size), pos - 1] = 1.0 - f
    W[np.arange(dst.size), pos] += f
    return W


WY = linear_weights(np.linspace(0, 121, 121), np.linspace(0, 121, 721))     # (721, 121)
WX = linear_weights(np.linspace(0, 240, 240), np.linspace(0, 240, 1440))    # (1440, 240)


def interp_masks(m):
    """(nt, 121, 240) -> (nt, 721, 1440) bilinear, as interp2d(kind='linear')."""
    tmp = m @ WX.T                  # (nt, 121, 1440)
    return WY @ tmp                 # (nt, 721, 1440)


def _f32(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32))


def load_masks(yy, mi):
    t0, t1 = MONTHNUM[mi], MONTHNUM[mi + 1]
    parts = {}
    for h in ("SH", "NH"):
        with netCDF4.Dataset(MASK_FMT.format(h=h, y=yy)) as ds:
            parts[h] = {k: np.asarray(ds[k][t0:t1]) for k in ("flag_A", "flag_C", "intensity_A", "intensity_C")}
            parts[h]["lat"] = np.asarray(ds["lat"][:])
    out = {}
    for k in ("flag_A", "flag_C", "intensity_A", "intensity_C"):
        out[k] = np.concatenate((parts["SH"][k][:, :-1, :], parts["NH"][k]), axis=1)[:, ::-1, :]
    lat_t = np.concatenate((parts["SH"]["lat"][:-1], parts["NH"]["lat"]))[::-1]
    assert lat_t[0] > lat_t[-1] and out["flag_C"].shape[1:] == (121, 240), (lat_t[:3], out["flag_C"].shape)
    return out


def load_wm2_shf(yy, mm, nt):
    """Local residual SHF (W/m2) with the v3 storage, production orientation (descending lat)."""
    with netCDF4.Dataset(VINT_FMT.format(y=yy, m=mm)) as ds:
        names = next(m for m in _VINT_MAPS if all(v in ds.variables for v in m.values()))
        lat = np.asarray(ds["latitude"][:])
        div = (_f32(ds[names["vigd"]][:nt]) + _f32(ds[names["vimdf"]][:nt]) * np.float32(LV)
               + _f32(ds[names["vithed"]][:nt]))
    if lat[0] < lat[-1]:
        div = div[:, ::-1, :]
    with netCDF4.Dataset(DHDT_FMT.format(y=yy, m=mm)) as ds:
        lat = np.asarray(ds["latitude"][:])
        tend = _f32(ds["tend_filtered"][:nt])
    if lat[0] < lat[-1]:
        tend = tend[:, ::-1, :]
    with netCDF4.Dataset(RAD_FMT.format(y=yy, m=mm)) as ds:
        lat = np.asarray(ds["latitude"][:])
        assert lat[0] > lat[-1]
        rad = (_f32(ds["tsr"][:nt]) - _f32(ds["ssr"][:nt]) + _f32(ds["ttr"][:nt])) / np.float32(3600.0)
    return div - rad + tend, tend


def run_month(mi):
    """Accumulate the 2000-2014 mean of monthly means for calendar month index mi (0-based)."""
    mm = mi + 1
    nt = int(MONTH_IND[mi])
    acc = {
        "Dhdt_all": np.zeros((719, 1440)), "Shf_all": np.zeros((719, 1440)), "tot_all": np.zeros((719, 1440)),
        "ShfW_all": np.zeros((721, 1440)), "DhdtW_all": np.zeros((721, 1440)),
    }
    for k in ("Dhdt", "Shf", "tot"):
        for s in ("cycl", "ant"):
            acc[f"{k}_{s}"] = np.zeros((6, 719, 1440))
    for s in ("cycl", "ant"):
        acc[f"ShfW_{s}"] = np.zeros((6, 721, 1440))
        acc[f"DhdtW_{s}"] = np.zeros((6, 721, 1440))

    for yy in YEARS:
        t0 = _time.time()
        with netCDF4.Dataset(V3_FMT.format(y=yy, m=mm)) as ds:
            dh = _f32(ds["F_Dhdt_final"][:nt])
            shf = _f32(ds["F_Shf_final"][:nt])
            lat = np.asarray(ds["lat"][:])
            lon = np.asarray(ds["lon"][:])
        with netCDF4.Dataset(V1_FMT.format(y=yy, m=mm)) as ds:
            tot = _f32(ds["tot_energy_final"][:nt])
            assert np.allclose(np.asarray(ds["lat"][:]), lat)
        shfw, dhw = load_wm2_shf(yy, mm, nt)
        assert dh.shape[0] == nt and shfw.shape[0] == nt, (dh.shape, shfw.shape, nt)
        masks = load_masks(yy, mi)
        assert masks["flag_C"].shape[0] == nt, (masks["flag_C"].shape, nt)

        acc["Dhdt_all"] += dh.mean(axis=0)
        acc["Shf_all"] += shf.mean(axis=0)
        acc["tot_all"] += tot.mean(axis=0)
        acc["ShfW_all"] += shfw.mean(axis=0)
        acc["DhdtW_all"] += dhw.mean(axis=0)
        for ci, cut in enumerate(CUTS):
            for s, flag, inten in (("cycl", "flag_C", "intensity_C"), ("ant", "flag_A", "intensity_A")):
                m = np.where(masks[inten] < cut, 0.0, masks[flag]).astype(np.float64)
                mi_ = interp_masks(m) >= 0.5                       # (nt, 721, 1440)
                keep = mi_[:, 1:-1, :]
                acc[f"Dhdt_{s}"][ci] += (dh * keep).mean(axis=0)
                acc[f"Shf_{s}"][ci] += (shf * keep).mean(axis=0)
                acc[f"tot_{s}"][ci] += (tot * keep).mean(axis=0)
                acc[f"ShfW_{s}"][ci] += (shfw * mi_).mean(axis=0)
                acc[f"DhdtW_{s}"][ci] += (dhw * mi_).mean(axis=0)
        print(f"month {mm:02d} year {yy} done ({_time.time() - t0:.0f}s)", flush=True)

    ny = float(len(YEARS))
    os.makedirs(PART_DIR, exist_ok=True)
    np.savez(os.path.join(PART_DIR, f"part_{mm:02d}.npz"), lat=lat, lon=lon,
             **{k: (v / ny).astype(np.float32) for k, v in acc.items()})
    print("saved part", mm, flush=True)


def merge():
    parts = [np.load(os.path.join(PART_DIR, f"part_{mm:02d}.npz")) for mm in range(1, 13)]
    lat, lon = parts[0]["lat"], parts[0]["lon"]
    stack = lambda k: np.stack([p[k] for p in parts], axis=-3)     # month axis before (lat, lon)

    with netCDF4.Dataset(WITH_INT) as ds:
        ref_tot = np.asarray(ds["tot_energy_final"][:])            # (6, 12, 719, 1440)
        ref_totc = np.asarray(ds["tot_energy_final_cycl"][:])
        ref_shf = np.asarray(ds["F_Shf_final"][:])
        ref_shfc = np.asarray(ds["F_Shf_final_cycl"][:])
        ref_dhc = np.asarray(ds["F_Dhdt_final_cycl"][:])
    tot_all = np.broadcast_to(stack("tot_all"), ref_tot.shape)
    tot_c = stack("tot_cycl")

    def rep(name, mine, ref):
        d = mine - ref
        print(f"  {name:22s} rms ref {np.sqrt(np.mean(ref**2)):.4f}  rms diff {np.sqrt(np.mean(d**2)):.2e}  "
              f"max |diff| {np.abs(d).max():.2e}  r={np.corrcoef(mine.ravel(), ref.ravel())[0, 1]:.6f}")
    print("reproduction check (unchanged variables, must match the stored WITH_INT file):")
    rep("tot_energy_final", tot_all, ref_tot)
    rep("tot_energy_final_cycl", tot_c, ref_totc)
    print("changes (v3 vs stored):")
    rep("F_Shf_final", np.broadcast_to(stack("Shf_all"), ref_shf.shape), ref_shf)
    rep("F_Shf_final_cycl", stack("Shf_cycl"), ref_shfc)
    rep("F_Dhdt_final_cycl", stack("Dhdt_cycl"), ref_dhc)

    # WITH_INT v3: copy of the stored file with the storage-dependent variables replaced
    import shutil
    shutil.copy2(WITH_INT, OUT_WITH_INT)
    with netCDF4.Dataset(OUT_WITH_INT, "a") as ds:
        ds["F_Dhdt_final"][:] = np.broadcast_to(stack("Dhdt_all"), ref_tot.shape)
        ds["F_Shf_final"][:] = np.broadcast_to(stack("Shf_all"), ref_tot.shape)
        for s in ("cycl", "ant"):
            ds[f"F_Dhdt_final_{s}"][:] = stack(f"Dhdt_{s}")
            ds[f"F_Shf_final_{s}"][:] = stack(f"Shf_{s}")
        ds.history = (getattr(ds, "history", "") + f" | {_time.strftime('%Y-%m-%d')}: F_Dhdt_final* and "
                      "F_Shf_final* rebuilt with the corrected storage term (Integrated_TE_v3, "
                      "resample_with_int_v3.py); all other variables unchanged")
    print("wrote", OUT_WITH_INT)

    with netCDF4.Dataset(OUT_WATTS, "w", format="NETCDF4_CLASSIC") as w:
        lat721 = np.arange(90, -90.01, -0.25)
        w.createDimension("lon", 1440); w.createDimension("lat", 721)
        w.createDimension("time", 12); w.createDimension("intensity", 6)
        v = w.createVariable("lon", "f4", ("lon",)); v[:] = lon
        v = w.createVariable("lat", "f4", ("lat",)); v[:] = lat721
        v = w.createVariable("time", "f4", ("time",)); v[:] = np.arange(12)
        v = w.createVariable("intensity", "f4", ("intensity",)); v[:] = CUTS
        for name, key in (("F_Shf_final", "ShfW"), ("F_Dhdt_final", "DhdtW")):
            v = w.createVariable(name, "f4", ("intensity", "time", "lat", "lon"))
            v[:] = np.broadcast_to(stack(f"{key}_all"), (6, 12, 721, 1440)); v.units = "W m-2"
            for s in ("cycl", "ant"):
                v = w.createVariable(f"{name}_{s}", "f4", ("intensity", "time", "lat", "lon"))
                v[:] = stack(f"{key}_{s}"); v.units = "W m-2"
        w.description = ("Local residual surface heat flux SHF = div - SWabs - OLR + storage and the "
                         "storage term itself (corrected storage, tend_filtered_3), 2000-2014 monthly "
                         "means, all / inside cyclone masks / inside anticyclone masks for intensity "
                         "cuts 1..6 CVU")
    print("wrote", OUT_WATTS)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        run_month(int(sys.argv[1]) - 1)
