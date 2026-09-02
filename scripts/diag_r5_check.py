"""Diagnostic: per-term effect of monthly vs constant normalization (Fig 3 b vs d).

For each budget term X of the weak-cyclone NH/SH panels:
  B      = DJF-JJA of the raw stormtrack series (panel-b bar)
  C      = B / (cos(mean lat) * f_hat)          (constant normalization)
  D      = DJF-JJA of X(m)/(cos(lat(m)) * f_hat) (exact, as in Fig 3c-f)
  corr   = D - C, split into:
    ann  = Xbar * DJF-JJA of h(m),  h = 1/(cos lat(m) f_hat) - 1/(cos latbar f_hat)
    anom = corr - ann (covariance of seasonal anomalies with h)
Prints everything plus the SHF:RAD ratios in both spaces.
"""
import numpy as np
import xarray as xr
from scipy import interpolate

ROOT = "/project2/tas1/gmsarro"
NC_FLUX = f"{ROOT}/cyclone_centered/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
NC_FLUX = NC_FLUX.replace("/cyclone_centered", "/cyclone_centered")
NC_CYC = f"{ROOT}/track/final/cyclonic_intensity.nc"
HALF_WIN = int(25600 / 2 / 9)


def main():
    with xr.open_dataset(NC_FLUX) as f:
        lat = f["lat"].values

        def _w(name):
            v = f[name].values
            return np.nanmean(v[0] - v[5], axis=2)

        te = _w("F_TE_final_cycl")
        sw = _w("F_Swabs_final_cycl")
        olr = _w("F_Olr_final_cycl")
        dhdt = _w("F_Dhdt_final_cycl")
        tot = _w("tot_energy_final_cycl") + dhdt
        umz = _w("F_TE_z_final_cycl")
        shf = tot - olr - sw
        fte_zon = np.nanmean(f["F_TE_final"][0].values, axis=2)
    with xr.open_dataset(NC_CYC) as f:
        ci = f["cycl_int_final"].values
        foot_zon = np.nanmean(ci[0] - ci[5], axis=2)

    xd = np.arange(lat.size, dtype=float)
    x_hi = np.linspace(0, lat.size - 1, 25600)
    lat_hi = interpolate.interp1d(xd, lat)(x_hi)
    yd = np.linspace(0, 12, 12)

    def _interp(zon):
        return interpolate.RectBivariateSpline(yd, xd, zon)(yd, x_hi)

    def _series(zon, idx12):
        zi = _interp(zon)
        return np.array([np.mean(zi[m, idx12[m] - HALF_WIN: idx12[m] + HALF_WIN])
                         for m in range(12)])

    def _sd(x):
        return np.mean(x[[11, 0, 1]]) - np.mean(x[[5, 6, 7]])

    fte_i = _interp(fte_zon)
    terms = {"TE": te, "SWABS": sw, "OLR": olr, "RAD": sw + olr,
             "SHF": shf, "Dhdt": dhdt, "UMZ": umz}

    for hemi in ("NH", "SH"):
        idx = (np.argmax if hemi == "NH" else np.argmin)(fte_i, axis=1)
        lat12 = lat_hi[idx]
        f_m = _series(foot_zon, idx)
        f_hat = float(np.mean(f_m))
        n_m = np.cos(np.deg2rad(lat12)) * f_hat
        n_bar = np.cos(np.deg2rad(np.mean(lat12))) * f_hat
        h = 1.0 / n_m - 1.0 / n_bar
        print(f"\n=== {hemi} weak cyclones ===")
        print(f"lat12: {np.round(lat12, 1)}")
        print(f"f_hat={f_hat:.4g}  DJF-JJA of h = {_sd(h):+.4g}")
        print(f"{'term':6s} {'B(panelb)':>10s} {'C=B/const':>10s} {'D(exact)':>10s} "
              f"{'corr=D-C':>10s} {'ann=Xbar*dh':>11s} {'anom':>8s} {'Xbar':>10s}")
        out = {}
        for k, zon in terms.items():
            x = _series(zon, idx)
            b = _sd(x)
            c = b / n_bar
            d = _sd(x / n_m)
            xbar = float(np.mean(x))
            ann = xbar * _sd(h)
            print(f"{k:6s} {b:>+10.4g} {c:>+10.4g} {d:>+10.4g} "
                  f"{d - c:>+10.4g} {ann:>+11.4g} {d - c - ann:>+8.4g} {xbar:>+10.4g}")
            out[k] = (b, c, d)
        for pair in (("SHF", "RAD"), ("SHF", "SWABS")):
            b1, _, d1 = out[pair[0]]
            b2, _, d2 = out[pair[1]]
            print(f"ratio {pair[0]}:{pair[1]}  panel-b {b1 / b2:+.3f}  "
                  f"panels c-f {d1 / d2:+.3f}")


if __name__ == "__main__":
    main()
