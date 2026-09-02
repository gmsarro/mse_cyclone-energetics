"""Stage 5 of the corrected-storage chain: rebuild every storage-dependent
variable of the cyclone-centered composites (Composites_*_noleap.nc) with the
corrected storage term, leaving all other composite variables untouched.

Rebuilt variables
  PW   : composite_Shf, composite_Dhdt          (from Integrated_TE_v3)
  W/m2 : composite_Shf_wm, composite_Dhdt_wm, composite_energy_wm,
         composite_Swabs_wm, composite_Olr_wm  (from the filtered vint, the
         v3 filtered storage and the ERA5 radiation files, correct orientation)

Snapshots (centres and times) are replayed from the *_center_samples.csv files
written by the composite builders, so the counts match the stored files
exactly (asserted). The NH Land/Ocean composites have no CSV; their snapshot
sets are the NH sets split by the ERA5 land-sea mask at the centre
(lsm > 0.5, nearest grid point, as in cyclone_energetics.composites.land_fraction),
and the resulting counts are asserted against the stored files as well.

The PW patch extraction follows cyclone_energetics.composites.builder
(ascending latitude, nearest index); it is validated against the *_center
columns of the CSV on the first processed month.

Usage: python regenerate_composites_v3.py [--write-only]
"""

import collections
import csv
import shutil
import sys
import time as _time

import netCDF4
import numpy as np

BASE = "/project2/tas1/gmsarro"
ERA5 = "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
VINT_FMT = BASE + "/smoothed_vint/era5_vint_{y:04d}_{m:02d}_filtered.nc"
DHDT_FMT = BASE + "/smoothed_dh_dt_ERA5_v3/tend_{y:04d}_{m:02d}_filtered_3.nc"
RAD_FMT = ERA5 + "/rad/era5_rad_{y:04d}_{m:02d}.6hrly.nc"
STF_FMT = ERA5 + "/stf/era5_stf_{y:04d}_{m:02d}.6hrly.nc"
PW_V3_FMT = BASE + "/cyclone_centered/Integrated_TE_v3/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
PW_V1_FMT = BASE + "/cyclone_centered/Integrated_TE/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
LSM_PATH = ERA5 + "/lsm/lsm.nc"
CKPT = BASE + "/cyclone_centered/regenerate_composites_v3_checkpoint.npz"

LV = 2.501e6
R_WINDOW_DEG, DRES = 15.0, 0.25
NY = NX = int(round(2 * R_WINDOW_DEG / DRES))
LAT_REL = np.linspace(-R_WINDOW_DEG + DRES / 2, R_WINDOW_DEG - DRES / 2, NY)
LON_REL = np.linspace(-R_WINDOW_DEG + DRES / 2, R_WINDOW_DEG - DRES / 2, NX)

CSV_TARGETS = [("Intense", "NH"), ("Weak", "NH"), ("Intense", "SH"), ("Weak", "SH")]
WM2_FIELDS = ("Shf_wm", "Dhdt_wm", "energy_wm", "Swabs_wm", "Olr_wm")
PW_FIELDS = ("Shf", "Dhdt")
# ERA5 surface turbulent flux and net surface LW into the atmosphere: composited for the
# response letter only (independent check of the residual), never written to the files
CHECK_FIELDS = ("Stf_wm", "Lwsfc_wm")

_VINT_MAPS = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]


def _to_ascending(arr, lat):
    return arr[:, ::-1, :] if lat[0] > lat[-1] else arr


def _nearest_idx(a_sorted, q):
    pos = np.clip(np.searchsorted(a_sorted, q), 1, a_sorted.size - 1)
    prev, nxt = a_sorted[pos - 1], a_sorted[pos]
    idx = pos.copy()
    choose_prev = (q - prev) <= (nxt - q)
    idx[choose_prev] = pos[choose_prev] - 1
    return idx


def load_lsm():
    with netCDF4.Dataset(LSM_PATH) as ds:
        lat = np.asarray(ds["latitude"][:], dtype=np.float32)
        lon = np.asarray(ds["longitude"][:], dtype=np.float32)
        lsm = np.asarray(ds["lsm"][:])
    if lsm.ndim == 3:
        lsm = lsm[0]
    lon = (lon % 360 + 360) % 360
    order = np.argsort(lon)
    lon, lsm = lon[order], lsm[:, order]
    if lat[0] > lat[-1]:
        lat, lsm = lat[::-1], lsm[::-1, :]
    return lat, lon, lsm.astype(np.float32)


def center_over_land(lat0, lon0, lsm_lat, lsm_lon, lsm):
    ilat = _nearest_idx(lsm_lat, np.array([lat0], dtype=np.float32))[0]
    ilon = _nearest_idx(lsm_lon, np.array([lon0 % 360], dtype=np.float32))[0]
    return bool(lsm[ilat, ilon] > 0.5)


def load_month(yy, mm):
    """W/m2 stacks on the ascending-lat 721x1440 grid and PW stacks on the
    ascending-lat 719x1440 grid (v3 storage)."""
    out = {}
    with netCDF4.Dataset(VINT_FMT.format(y=yy, m=mm)) as ds:
        names = next(m for m in _VINT_MAPS if all(v in ds.variables for v in m.values()))
        lat = np.asarray(ds["latitude"][:])
        energy = (np.asarray(ds[names["vigd"]][:], dtype=np.float32)
                  + np.asarray(ds[names["vimdf"]][:], dtype=np.float32) * np.float32(LV)
                  + np.asarray(ds[names["vithed"]][:], dtype=np.float32))
    energy = _to_ascending(energy, lat)
    with netCDF4.Dataset(DHDT_FMT.format(y=yy, m=mm)) as ds:
        lat = np.asarray(ds["latitude"][:])
        dhdt = np.nan_to_num(np.asarray(ds["tend_filtered"][:], dtype=np.float32), nan=0.0)
    dhdt = _to_ascending(dhdt, lat)
    with netCDF4.Dataset(RAD_FMT.format(y=yy, m=mm)) as ds:
        lat = np.asarray(ds["latitude"][:])
        nt = min(energy.shape[0], dhdt.shape[0], ds["tsr"].shape[0])
        tsr = np.nan_to_num(np.asarray(ds["tsr"][:nt], dtype=np.float32) / 3600.0)
        ssr = np.nan_to_num(np.asarray(ds["ssr"][:nt], dtype=np.float32) / 3600.0)
        ttr = np.nan_to_num(np.asarray(ds["ttr"][:nt], dtype=np.float32) / 3600.0)
        lw = -np.nan_to_num(np.asarray(ds["str"][:nt], dtype=np.float32)) / np.float32(3600.0)
    out["Swabs_wm"] = _to_ascending(tsr - ssr, lat)
    out["Olr_wm"] = _to_ascending(ttr, lat)
    out["Lwsfc_wm"] = _to_ascending(lw, lat)
    del tsr, ssr, ttr
    out["energy_wm"] = energy[:nt]
    out["Dhdt_wm"] = dhdt[:nt]
    out["Shf_wm"] = out["energy_wm"] - out["Swabs_wm"] - out["Olr_wm"] + out["Dhdt_wm"]
    with netCDF4.Dataset(STF_FMT.format(y=yy, m=mm)) as ds:
        lat = np.asarray(ds["latitude"][:])
        stf = -(np.nan_to_num(np.asarray(ds["sshf"][:nt], dtype=np.float32))
                + np.nan_to_num(np.asarray(ds["slhf"][:nt], dtype=np.float32))) / np.float32(3600.0)
    out["Stf_wm"] = _to_ascending(stf, lat)
    with netCDF4.Dataset(PW_V3_FMT.format(y=yy, m=mm)) as ds:
        lat_pw = np.asarray(ds["lat"][:], dtype=np.float32)
        lon_pw = np.asarray(ds["lon"][:], dtype=np.float32)
        shf = np.asarray(ds["F_Shf_final"][:], dtype=np.float32)
        dh = np.asarray(ds["F_Dhdt_final"][:], dtype=np.float32)
    if lat_pw[0] > lat_pw[-1]:
        lat_pw, shf, dh = lat_pw[::-1], shf[:, ::-1, :], dh[:, ::-1, :]
    assert np.all(np.diff(lon_pw) > 0)
    out["Shf"], out["Dhdt"], out["pw_lat"], out["pw_lon"] = shf, dh, lat_pw, lon_pw
    return out


def patch_wm(arr2d, lat0, lon0):
    ilat = np.clip(np.round((lat0 + LAT_REL + 90.0) * 4).astype(int), 0, 720)
    ilon = (np.round(((lon0 + LON_REL) % 360.0) * 4).astype(int)) % 1440
    return arr2d[ilat[:, None], ilon[None, :]]


def patch_pw(arr2d, lat_vals, lon_vals, lat0, lon0):
    ilat = _nearest_idx(lat_vals, (lat0 + LAT_REL).astype(np.float32))
    ilon = _nearest_idx(lon_vals, ((lon0 + LON_REL) % 360.0).astype(np.float32))
    return arr2d[ilat[:, None], ilon[None, :]]


def read_csv(tag, hemi):
    rows = []
    with open(BASE + f"/cyclone_centered/Composites_{tag}_{hemi}_noleap_center_samples.csv") as f:
        for r in csv.DictReader(f):
            lt = (int(r["day"]) - 1) * 4 + int(r["hour6"]) // 6
            rows.append((int(r["year"]), int(r["month"]), lt, float(r["lat"]), float(r["lon"]),
                         float(r["Shf_center"]), float(r["Dhdt_center"])))
    return rows


def validate_pw_indexing(yy, mm, members):
    """Centre of the v1 F_Shf patch must equal the CSV Shf_center (index convention)."""
    with netCDF4.Dataset(PW_V1_FMT.format(y=yy, m=mm)) as ds:
        lat_pw = np.asarray(ds["lat"][:], dtype=np.float32)
        lon_pw = np.asarray(ds["lon"][:], dtype=np.float32)
        shf = np.asarray(ds["F_Shf_final"][:], dtype=np.float32)
    if lat_pw[0] > lat_pw[-1]:
        lat_pw, shf = lat_pw[::-1], shf[:, ::-1, :]
    diffs = []
    for key, lt, la, lo, shf_c, _ in members:
        if key[1] not in ("NH", "SH") or len(key) != 2:
            continue
        p = patch_pw(shf[min(lt, shf.shape[0] - 1)], lat_pw, lon_pw, la, lo)
        diffs.append(abs(float(p[NY // 2, NX // 2]) - shf_c))
    print(f"PW index validation {yy}-{mm:02d}: {len(diffs)} centres, max |patch centre - CSV| = "
          f"{max(diffs):.3e} PW", flush=True)
    assert max(diffs) < 1e-4, "PW patch index convention does not reproduce the CSV centre values"


def compute():
    lsm_lat, lsm_lon, lsm = load_lsm()
    jobs = {}
    for tag, hemi in CSV_TARGETS:
        rows = read_csv(tag, hemi)
        jobs[(tag, hemi)] = rows
        print(f"{tag}_{hemi}: {len(rows)} snapshots", flush=True)
        if hemi == "NH":
            land = [r for r in rows if center_over_land(r[3], r[4], lsm_lat, lsm_lon, lsm)]
            ocean = [r for r in rows if not center_over_land(r[3], r[4], lsm_lat, lsm_lon, lsm)]
            jobs[(tag, hemi, "Land")], jobs[(tag, hemi, "Ocean")] = land, ocean
            print(f"  -> Land {len(land)}, Ocean {len(ocean)}", flush=True)

    # counts must match the stored composite files before any heavy work
    for key, rows in jobs.items():
        cnt = np.zeros(12, dtype=int)
        for r in rows:
            cnt[r[1] - 1] += 1
        with netCDF4.Dataset(nc_path(key)) as ds:
            stored = np.asarray(ds["count"][:]).astype(int)
        assert np.array_equal(stored, cnt), f"count mismatch for {key}: stored {stored}, replay {cnt}"
    print("all 8 snapshot sets reproduce the stored counts", flush=True)

    by_month = collections.defaultdict(list)
    for key, rows in jobs.items():
        for yy, mm, lt, la, lo, shf_c, dh_c in rows:
            by_month[(yy, mm)].append((key, lt, la, lo, shf_c, dh_c))

    sums = {k: {v: np.zeros((12, NY, NX)) for v in WM2_FIELDS + CHECK_FIELDS + PW_FIELDS} for k in jobs}
    counts = {k: np.zeros(12, dtype=int) for k in jobs}
    for gi, ((yy, mm), members) in enumerate(sorted(by_month.items())):
        t0 = _time.time()
        if gi == 0:
            validate_pw_indexing(yy, mm, members)
        data = load_month(yy, mm)
        nt = min(data[v].shape[0] for v in WM2_FIELDS + PW_FIELDS)
        for key, lt, la, lo, _, _ in members:
            lt = min(lt, nt - 1)
            for v in WM2_FIELDS + CHECK_FIELDS:
                sums[key][v][mm - 1] += patch_wm(data[v][lt], la, lo)
            for v in PW_FIELDS:
                sums[key][v][mm - 1] += patch_pw(data[v][lt], data["pw_lat"], data["pw_lon"], la, lo)
            counts[key][mm - 1] += 1
        del data
        if gi % 12 == 0:
            print(f"[{gi + 1}/{len(by_month)}] {yy}-{mm:02d} ({len(members)} snaps, "
                  f"{_time.time() - t0:.1f}s)", flush=True)

    out = {}
    for key, s in sums.items():
        for v in s:
            out[f"{label(key)}_{v}"] = s[v]
        out[f"{label(key)}_count"] = counts[key]
    np.savez(CKPT, **out)
    print("checkpoint saved", CKPT, flush=True)


def label(key):
    return "_".join(key)


def nc_path(key):
    return BASE + f"/cyclone_centered/Composites_{label(key)}_noleap.nc"


def write():
    ck = np.load(CKPT)
    keys = list(CSV_TARGETS) + [(t, "NH", s) for t in ("Intense", "Weak") for s in ("Land", "Ocean")]
    for key in keys:
        cnt = ck[f"{label(key)}_count"].astype(int)
        path = nc_path(key)
        backup = path + ".bak_v1"
        if not __import__("os").path.exists(backup):
            shutil.copy2(path, backup)
        with netCDF4.Dataset(path, "a") as ds:
            stored = np.asarray(ds["count"][:]).astype(int)
            assert np.array_equal(stored, cnt), f"count mismatch {key}: stored {stored} replay {cnt}"
            denom = np.maximum(cnt, 1)[:, None, None].astype(float)
            for v in WM2_FIELDS + PW_FIELDS:
                name = f"composite_{v}"
                assert ds[name].shape == (12, NY, NX), (name, ds[name].shape)
                ds[name][:] = (ck[f"{label(key)}_{v}"] / denom).astype(np.float32)
            hist = getattr(ds, "history", "")
            ds.history = (hist + f" | {_time.strftime('%Y-%m-%d')}: composite_{{Shf,Dhdt}} and "
                          "composite_{Shf,Dhdt,energy,Swabs,Olr}_wm rebuilt with the corrected storage "
                          "term (tend_*_filtered_3, column-mass change included) by "
                          "regenerate_composites_v3.py")
        print(f"updated {path} (backup {backup})", flush=True)


if __name__ == "__main__":
    if "--write-only" not in sys.argv:
        compute()
    write()
