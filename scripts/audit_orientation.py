"""Per-month latitude-orientation audit of the smoothed vint/tend files.

For every month 2000-2014, recover the implied integrand of the stored
tot_energy_final and F_Dhdt_final (d/dphi of the poleward integral, an
orientation ground truth anchored to the F files' own latitude coordinate)
and correlate it against the directly composed source field in both
orientations:

  r_meta : source data assumed to follow their ascending latitude metadata
           (flip once to the descending F-file frame)
  r_desc : source data assumed stored descending despite metadata (as-is)

If the Hoskins-filter step wrote some batches in the opposite order, r_meta
and r_desc will switch which one is ~1 for those months. Also reports months
where NEITHER correlates (stored F built from misaligned components).
"""

import netCDF4
import numpy as np

A = 6.371e6
LV = 2.5008e6
T = 5
BASE = "/project2/tas1/gmsarro"

lat = None
phi = None
cosl = None


def implied(path, var):
    global lat, phi, cosl
    with netCDF4.Dataset(path) as f:
        if lat is None:
            lat = np.array(f["lat"][:], dtype=np.float64)
            phi = np.deg2rad(lat)
            cosl = np.cos(phi)
        t = min(T, f[var].shape[0] - 1)
        F = np.array(f[var][t], dtype=np.float64) * 1e15
    dF = np.gradient(F, axis=0) / np.gradient(phi)[:, None]
    return dF / (2 * np.pi * A ** 2 * cosl[:, None]), t


def direct(field_721_descframe):
    f = field_721_descframe[1:-1]
    mbar = np.average(f, weights=cosl, axis=0)
    return f - mbar[None, :]


def corr(a, b, inner):
    return np.corrcoef(a[inner].ravel(), b[inner].ravel())[0, 1]


def main():
    bad = []
    print(f"{'month':>8} {'vint_meta':>9} {'vint_desc':>9} "
          f"{'tend_meta':>9} {'tend_desc':>9}")
    for yy in range(2000, 2015):
        for mm in range(1, 13):
            fint = f"{BASE}/cyclone_centered/Integrated_TE/Integrated_Fluxes_{yy}_{mm:02d}_.nc"
            fnew = f"{BASE}/cyclone_centered/Integrated_TE/New_Integrated_Fluxes_{yy}_{mm:02d}_.nc"
            si_tot, t_used = implied(fint, "tot_energy_final")
            si_dh, _ = implied(fnew, "F_Dhdt_final")
            inner = np.abs(lat) < 88

            with netCDF4.Dataset(
                f"{BASE}/smoothed_vint/era5_vint_{yy}_{mm:02d}_filtered.nc"
            ) as ds:
                names = {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered",
                         "vithed": "p83.162_filtered"}
                if names["vigd"] not in ds.variables:
                    names = {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered",
                             "vithed": "vithed_filtered"}
                meta_asc = ds["latitude"][0] < ds["latitude"][-1]
                tot = (np.array(ds[names["vigd"]][t_used], dtype=np.float64)
                       + np.array(ds[names["vimdf"]][t_used], dtype=np.float64) * LV
                       + np.array(ds[names["vithed"]][t_used], dtype=np.float64))
            # to descending frame under each hypothesis
            tot_meta = tot[::-1] if meta_asc else tot
            tot_desc = tot if meta_asc else tot[::-1]
            rv_meta = corr(si_tot, direct(tot_meta), inner)
            rv_desc = corr(si_tot, direct(tot_desc), inner)

            with netCDF4.Dataset(
                f"{BASE}/smoothed_dh_dt_ERA5/tend_{yy}_{mm:02d}_filtered_2.nc"
            ) as ds:
                meta_asc_t = ds["latitude"][0] < ds["latitude"][-1]
                tend = np.nan_to_num(
                    np.array(ds["tend_filtered"][t_used], dtype=np.float64))
            tend_meta = tend[::-1] if meta_asc_t else tend
            tend_desc = tend if meta_asc_t else tend[::-1]
            rt_meta = corr(si_dh, direct(tend_meta), inner)
            rt_desc = corr(si_dh, direct(tend_desc), inner)

            flag = ""
            if rv_meta < 0.99 or rt_meta < 0.99:
                flag = "  <-- CHECK"
                bad.append((yy, mm, rv_meta, rv_desc, rt_meta, rt_desc))
            print(f"{yy}-{mm:02d} {rv_meta:9.4f} {rv_desc:9.4f} "
                  f"{rt_meta:9.4f} {rt_desc:9.4f}{flag}", flush=True)

    print()
    if bad:
        print(f"{len(bad)} month(s) where metadata-consistent reading does "
              "not reproduce the stored transports:")
        for yy, mm, a_, b_, c_, d_ in bad:
            print(f"  {yy}-{mm:02d}: vint meta/desc {a_:.3f}/{b_:.3f}  "
                  f"tend meta/desc {c_:.3f}/{d_:.3f}")
    else:
        print("All 180 months: metadata-consistent orientation reproduces the "
              "stored transports (r >= 0.99). Stored F files and the "
              "regeneration assumption are consistent.")


if __name__ == "__main__":
    main()
