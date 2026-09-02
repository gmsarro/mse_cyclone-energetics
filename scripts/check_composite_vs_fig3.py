"""Consistency check: cyclone-centered composites vs the Fig 3 stormtrack values.

Fig 3 side (WITH_INT sampled file, intense cut = index 5, SH):
  SHF_cycl(m, lat, lon) = tot_energy_final_cycl + F_Dhdt_final_cycl
                          - F_Olr_final_cycl - F_Swabs_final_cycl
  zonal mean -> interpolate in latitude -> mean over the +/-HALF_WIN window
  around the monthly SH stormtrack latitude (replicating final_figures.ipynb).

Composite side (Composites_Intense_SH_noleap.nc):
  composite_Shf(m, y, x): unconditioned composite of the same F_Shf field
  around intense SH cyclone centers (selection is stormtrack-relative,
  +/-5 deg). Compare its center value and small-window means against the
  Fig 3 series: annual means and month-to-month evolution should agree.
"""

import netCDF4
import numpy as np
import scipy.interpolate

BASE = "/project2/tas1/gmsarro"
NCF = f"{BASE}/cyclone_centered/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
NOLEAP = f"{BASE}/cyclone_centered/Composites_Intense_SH_noleap.nc"
ICUT = 5  # intense (>6 CVU)

with netCDF4.Dataset(NCF) as f:
    lat = np.array(f["lat"][:])
    fte_all = np.array(f["F_TE_final"][:])          # (cut, 12, lat, lon)
    tot = np.array(f["tot_energy_final_cycl"][ICUT])
    dh = np.array(f["F_Dhdt_final_cycl"][ICUT])
    olr = np.array(f["F_Olr_final_cycl"][ICUT])
    sw = np.array(f["F_Swabs_final_cycl"][ICUT])

shf = tot + dh - olr - sw                            # (12, lat, lon)
shf_zon = np.nanmean(shf, axis=2)                    # (12, lat)

# stormtrack replication (final_figures.ipynb cell 36)
fte_total = np.sum(fte_all, axis=0)
fte_zonal = np.mean(fte_total, axis=2)
nlat = len(lat)
x_d = np.arange(nlat, dtype=float)
y_d = np.linspace(0, 12, 12)
x3_d = np.linspace(0, nlat - 1, 25600)
Lat = scipy.interpolate.interp1d(x_d, lat)(x3_d)
fte_i = scipy.interpolate.RectBivariateSpline(y_d, x_d, fte_zonal)(y_d, x3_d)
st_sh_idx = np.argmin(fte_i, axis=1)
st_sh_lat = Lat[st_sh_idx]
HALF_WIN = int(25600 / 2 / 9)

shf_i = scipy.interpolate.RectBivariateSpline(y_d, x_d, np.nan_to_num(shf_zon))(y_d, x3_d)
fig3_series = np.array([
    np.mean(shf_i[m, st_sh_idx[m] - HALF_WIN: st_sh_idx[m] + HALF_WIN])
    for m in range(12)
])

with netCDF4.Dataset(NOLEAP) as f:
    comp = np.array(f["composite_Shf"][:])           # (12, y, x)
    yg = np.array(f["y"][:])
    xg = np.array(f["x"][:])

jc = np.argmin(np.abs(yg))
ic = np.argmin(np.abs(xg))
center = comp[:, jc, ic]
win5 = comp[:, np.abs(yg) <= 5][:, :, np.abs(xg) <= 5].mean(axis=(1, 2))
win10 = comp[:, np.abs(yg) <= 10][:, :, np.abs(xg) <= 10].mean(axis=(1, 2))
win15 = comp.mean(axis=(1, 2))


# intense-cyclone coverage frequency at the monthly stormtrack: the sampled
# WITH_INT fields are unconditional means (zero where no cyclone), the
# composite is conditional (cyclone at center), so
#   Fig3_series(m) ~= conditional_composite(m) * coverage_freq(m)
with netCDF4.Dataset(f"{BASE}/track/final/cyclonic_intensity.nc") as f:
    freq = np.array(f["cycl_int_final"][ICUT])       # (12, lat, lon)
freq_zon = np.mean(freq, axis=2)
freq_i = scipy.interpolate.RectBivariateSpline(y_d, x_d, freq_zon)(y_d, x3_d)
freq_series = np.array([
    np.mean(freq_i[m, st_sh_idx[m] - HALF_WIN: st_sh_idx[m] + HALF_WIN])
    for m in range(12)
])


def rep(name, s):
    r = np.corrcoef(s, fig3_series)[0, 1]
    print(f"{name:26s} ann={s.mean():+7.3f} PW  amp={s.max() - s.min():6.3f}  "
          f"r(vs Fig3)={r:+.3f}")


print("monthly SH stormtrack lat:", np.round(st_sh_lat, 1))
print("intense coverage freq (%):", np.round(100 * freq_series, 2))
print()
print(f"{'series':26s} {'annual':>11s} {'seasonal amp':>10s}")
rep("Fig3 stormtrack SHF", fig3_series)
rep("composite center", center)
rep("center * freq", center * freq_series)
rep("win5 * freq", win5 * freq_series)
rep("win10 * freq", win10 * freq_series)
print()
print("month-by-month (PW):")
print("Fig3        :", np.round(fig3_series, 3))
print("center*freq :", np.round(center * freq_series, 3))
print("win5*freq   :", np.round(win5 * freq_series, 3))

# ---------------------------------------------------------------------------
# Same check for the local W/m2 budget-residual SHF: the WATTS sampled file
# (independent pipeline, verified correctly oriented) vs the regenerated
# composite_Shf_wm.  Also compares the regenerated composite_stf_wm against
# the independently computed R6 composite (different window/centering).
# ---------------------------------------------------------------------------
NCW = f"{BASE}/cyclone_centered/WATTS_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
with netCDF4.Dataset(NCW) as f:
    lat_w = np.array(f["lat"][:])                       # 721 rows (WITH_INT has 719)
    shf_w = np.array(f["F_Shf_final_cycl"][ICUT])      # (12, lat, lon), W/m2
shf_w_zon = np.nan_to_num(np.nanmean(shf_w, axis=2))
if lat_w[0] > lat_w[-1]:
    lat_w, shf_w_zon = lat_w[::-1], shf_w_zon[:, ::-1]
shf_w_i = np.stack([np.interp(Lat, lat_w, shf_w_zon[m]) for m in range(12)])
watts_series = np.array([
    np.mean(shf_w_i[m, st_sh_idx[m] - HALF_WIN: st_sh_idx[m] + HALF_WIN])
    for m in range(12)
])

with netCDF4.Dataset(NOLEAP) as f:
    hist = getattr(f, "history", "")
    comp_wm = np.array(f["composite_Shf_wm"][:])
    comp_stf = (np.array(f["composite_stf_wm"][:])
                if "composite_stf_wm" in f.variables else None)
print()
print("noleap history:", hist[-120:] if hist else "(none)")
wm_c = comp_wm[:, jc, ic]
wm_w5 = comp_wm[:, np.abs(yg) <= 5][:, :, np.abs(xg) <= 5].mean(axis=(1, 2))
wm_w10 = comp_wm[:, np.abs(yg) <= 10][:, :, np.abs(xg) <= 10].mean(axis=(1, 2))


def rep_w(name, s):
    r = np.corrcoef(s, watts_series)[0, 1]
    print(f"{name:26s} ann={s.mean():+8.2f} W/m2  amp={s.max() - s.min():7.2f}  "
          f"r(vs WATTS)={r:+.3f}")


print(f"{'W/m2 series':26s} {'annual':>14s} {'seasonal amp':>10s}")
rep_w("WATTS stormtrack SHF", watts_series)
rep_w("Shf_wm center * freq", wm_c * freq_series)
rep_w("Shf_wm win5 * freq", wm_w5 * freq_series)
rep_w("Shf_wm win10 * freq", wm_w10 * freq_series)
print("WATTS  :", np.round(watts_series, 2))
print("win10*f:", np.round(wm_w10 * freq_series, 2))

if comp_stf is not None:
    r6 = np.load(f"{BASE}/mse_cyclone-energetics/figures/reviewer_response/r6_fig5d_v2_data.npz")
    c30 = r6["comp_stf"]                                 # (12, 240, 240) ascending y
    y30, x30 = r6["y"], r6["x"]
    sy = np.abs(y30) <= 15
    sx = np.abs(x30) <= 15
    r6_crop = c30[:, sy][:, :, sx].mean(axis=0)           # (120,120) on +/-14.875 grid
    stf_ann = comp_stf.mean(axis=0)
    rr = np.corrcoef(r6_crop.ravel(), stf_ann.ravel())[0, 1]
    print()
    print(f"composite_stf_wm (regenerated) vs R6 comp_stf (independent): "
          f"pattern r={rr:.4f}, mean {stf_ann.mean():.1f} vs {r6_crop.mean():.1f} W/m2")
