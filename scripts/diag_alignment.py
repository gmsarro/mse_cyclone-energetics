"""Index alignment of the smoothed products with their raw inputs (one month).

Spatial smoothing must preserve the time index, so corr(smoothed[t], raw[t+k])
must peak sharply at k = 0 for the vint divergences and for both storage
variants. Also compares raw v1 and v2 storage fields to see how they differ.
"""

import netCDF4
import numpy as np

BASE = "/project2/tas1/gmsarro"
ERA5 = "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
yy, mm = 2005, 7
T = [30, 60, 90]           # time indices tested


def to_asc(a, lat):
    return a[::-1, :] if lat[0] > lat[-1] else a


def lagcorr(raw_ds, raw_var, sm_ds, sm_var, label):
    lat_r = np.asarray(raw_ds["latitude"][:])
    lat_s = np.asarray(sm_ds["latitude"][:])
    out = {}
    for k in (-1, 0, 1):
        cs = []
        for t in T:
            s = to_asc(np.asarray(sm_ds[sm_var][t], dtype=np.float64), lat_s)
            r = to_asc(np.asarray(raw_ds[raw_var][t + k], dtype=np.float64), lat_r)
            cs.append(np.corrcoef(s[100:620].ravel(), r[100:620].ravel())[0, 1])
        out[k] = round(float(np.mean(cs)), 4)
    print(f"{label:34s} corr(smoothed[t], raw[t+k]) k=-1,0,+1: {out[-1]}, {out[0]}, {out[1]}")


with netCDF4.Dataset(f"{ERA5}/vint/era5_vint_{yy}_{mm:02d}.6hrly.nc") as raw, \
        netCDF4.Dataset(f"{BASE}/smoothed_vint/era5_vint_{yy}_{mm:02d}_filtered.nc") as sm:
    for v in ("p83.162", "p84.162", "p85.162"):
        lagcorr(raw, v, sm, v + "_filtered", f"vint {v}")

with netCDF4.Dataset(f"{BASE}/dh_dt_data_ERA5/tend_{yy}_{mm:02d}.nc") as raw1, \
        netCDF4.Dataset(f"{BASE}/smoothed_dh_dt_ERA5/tend_{yy}_{mm:02d}_filtered.nc") as sm1:
    v1name = [v for v in raw1.variables if v not in ("time", "latitude", "longitude")][0]
    print("raw v1 variable:", v1name, raw1[v1name].shape, getattr(raw1[v1name], "units", ""))
    lagcorr(raw1, v1name, sm1, "tend_filtered", "storage v1 (tend_YYYY_MM.nc)")
    lat1 = np.asarray(raw1["latitude"][:])
    r1 = to_asc(np.asarray(raw1[v1name][60], dtype=np.float64), lat1)

with netCDF4.Dataset(f"{BASE}/dh_dt_data_ERA5/tend_{yy}_{mm:02d}_2.nc") as raw2, \
        netCDF4.Dataset(f"{BASE}/smoothed_dh_dt_ERA5/tend_{yy}_{mm:02d}_filtered_2.nc") as sm2:
    lagcorr(raw2, "tend", sm2, "tend_filtered", "storage v2 (tend_YYYY_MM_2.nc)")
    lat2 = np.asarray(raw2["latitude"][:])
    r2 = to_asc(np.asarray(raw2["tend"][60], dtype=np.float64), lat2)
    for k in (-1, 0, 1):
        rk = to_asc(np.asarray(raw2["tend"][60 + k], dtype=np.float64), lat2)
        print(f"raw v1[60] vs raw v2[60+{k:+d}]: r={np.corrcoef(r1[100:620].ravel(), rk[100:620].ravel())[0, 1]:.4f}")
    print(f"raw v1 vs v2 at t=60: std v1={r1[100:620].std():.0f}  std v2={r2[100:620].std():.0f}  "
          f"slope v2~a*v1: a={np.polyfit(r1[100:620].ravel(), r2[100:620].ravel(), 1)[0]:.3f}")
