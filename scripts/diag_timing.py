"""Are div (smoothed vint) and the storage term time-aligned?

For propagating synoptic systems the leading balance is div ~ -tend. If the
two fields are on the same time stamps, corr(div[t], -tend[t+k]) peaks at
k = 0; a systematic peak at k != 0 would reveal an index offset in one of the
stored files. Also checks the mass term: corr(div[t], -M[t+k]) should peak
at k = 0 too if p_s and vint are aligned. One month, SH storm track band.
"""

import netCDF4
import numpy as np

import diag_budget_terms as dbt
import diag_mass_term as dmt

yy, mm = 2005, 7
raw = dbt.load_raw(yy, mm)
data, lat = dmt.load_month(yy, mm)
band = (lat <= -30) & (lat >= -70)
div = raw["div"][:, band, :].astype(np.float64)
nt = div.shape[0]


def lagcorr(a, b, k):
    if k >= 0:
        x, y = a[2:nt - 3], b[2 + k:nt - 3 + k]
    else:
        x, y = a[2:nt - 3], b[2 + k:nt - 3 + k]
    return np.corrcoef(x.ravel(), y.ravel())[0, 1]


for name in ("tend1", "tend2"):
    t = raw[name][:, band, :].astype(np.float64)
    print(name, "corr(div[t], -tend[t+k]):",
          {k: round(lagcorr(div, -t, k), 4) for k in (-2, -1, 0, 1, 2)})
    s = div + t
    print(name, "std(div + tend[t+k]) W/m2:",
          {k: int(np.std(div[2:nt - 3] + t[2 + k:nt - 3 + k])) for k in (-2, -1, 0, 1, 2)})
M = data["M"][:, band, :].astype(np.float64)
print("corr(div[t], -M[t+k]):", {k: round(lagcorr(div, -M, k), 4) for k in (-2, -1, 0, 1, 2)})
res = raw["div"] - raw["swabs"] - raw["olr"] + raw["tend1"]
res_b = res[:, band, :].astype(np.float64)
closed = (raw["stf"] + raw["lw_sfc"])[:, band, :].astype(np.float64)
print("pointwise 6-hourly, band 30-70S:")
print("  std(res)=%.0f  std(res + M)=%.0f  std(closed)=%.0f W/m2" % (res_b.std(), (res_b + M).std(), closed.std()))
print("  corr(res, closed)=%.3f  corr(res + M, closed)=%.3f" % (
    np.corrcoef(res_b.ravel(), closed.ravel())[0, 1], np.corrcoef((res_b + M).ravel(), closed.ravel())[0, 1]))
print("  corr(res - closed, -M)=%.3f  slope=%.2f" % (
    np.corrcoef((res_b - closed).ravel(), (-M).ravel())[0, 1],
    np.polyfit((-M).ravel(), (res_b - closed).ravel(), 1)[0]))
