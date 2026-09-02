"""Stage 3 of the corrected-storage chain: rebuild the poleward-integrated
storage and residual surface-flux fields with the production formula
(make_TE_int.py, reproduced to machine precision in diag_budget_terms.py Part A)
but with the corrected, Hoskins-filtered storage term tend_*_filtered_3.nc:

    SHF_res = div_filt - (TSR - SSR) - TTR + tend_filt_v3
    F_x(t, lat, lon) = 2 pi a^2 int_{-90}^{lat} (x - <x>_lat) cos(lat') dlat' / 1e15

Only F_Dhdt_final and F_Shf_final depend on the storage term, so only these two
are written, to Integrated_TE_v3/Integrated_Fluxes_YYYY_MM_.nc; the unchanged
variables (tot_energy_final, F_Swabs_final, F_Olr_final, F_TE_final, F_u/v_mse)
stay in the original Integrated_TE files. Grid, orientation (descending latitude,
interior points lat[1:-1]) and record count follow the original file of the month.

Usage: python integrate_fluxes_v3.py YEAR [MONTH ...]
"""

import logging
import pathlib
import sys
import time as _time

import netCDF4
import numpy as np
import scipy.integrate as integrate

BASE = "/project2/tas1/gmsarro"
ERA5 = "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
VINT_FMT = BASE + "/smoothed_vint/era5_vint_{y:04d}_{m:02d}_filtered.nc"
TEND3_FMT = BASE + "/smoothed_dh_dt_ERA5_v3/tend_{y:04d}_{m:02d}_filtered_3.nc"
RAD_FMT = ERA5 + "/rad/era5_rad_{y:04d}_{m:02d}.6hrly.nc"
INT_FMT = BASE + "/cyclone_centered/Integrated_TE/Integrated_Fluxes_{y:04d}_{m:02d}_.nc"
OUT_DIR = pathlib.Path(BASE + "/cyclone_centered/Integrated_TE_v3")

LV = 2.501e6
A_EARTH = 6.371e6
_VINT_MAPS = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]
_cumtrapz = getattr(integrate, "cumulative_trapezoid", None) or integrate.cumtrapz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
_LOG = logging.getLogger("integrate_v3")


def integrated_prod(x, lat):
    """Verbatim production integration (make_TE_int.py), vectorised: x (lat, ...)."""
    l = np.deg2rad(lat)
    w = np.cos(l)
    x = x - np.average(x, weights=w, axis=0)
    x = x * w.reshape((-1,) + (1,) * (x.ndim - 1))
    int_x = _cumtrapz(x[::-1], l[::-1], axis=0, initial=None)
    int_x_r = _cumtrapz(x, l, axis=0, initial=None)
    return 2 * np.pi * A_EARTH ** 2 * (int_x[::-1][1:] + int_x_r[:-1]) / 2 / 1e15


def integ(field, lat):
    """(t, lat, lon) W/m2 -> (t, lat-2, lon) PW."""
    x = np.moveaxis(field.astype(np.float64), 1, 0)
    res = integrated_prod(x.reshape(x.shape[0], -1), lat)
    return np.moveaxis(res.reshape(-1, x.shape[1], x.shape[2]), 0, 1)


def _f32(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32))


def process_month(yy, mm):
    out_path = OUT_DIR / ("Integrated_Fluxes_%04d_%02d_.nc" % (yy, mm))
    if out_path.exists():
        _LOG.info("exists, skipping %s", out_path)
        return
    t0 = _time.time()
    with netCDF4.Dataset(INT_FMT.format(y=yy, m=mm)) as ds:
        nt = ds["F_Shf_final"].shape[0]
        lat_int = np.asarray(ds["lat"][:])
        lon_int = np.asarray(ds["lon"][:])
    with netCDF4.Dataset(VINT_FMT.format(y=yy, m=mm)) as ds:
        names = next(m for m in _VINT_MAPS if all(v in ds.variables for v in m.values()))
        lat_v = np.asarray(ds["latitude"][:])
        assert lat_v[0] < lat_v[-1], "smoothed vint expected ascending"
        div = (_f32(ds[names["vigd"]][:nt]) + _f32(ds[names["vimdf"]][:nt]) * np.float32(LV)
               + _f32(ds[names["vithed"]][:nt]))[:, ::-1, :]
    with netCDF4.Dataset(TEND3_FMT.format(y=yy, m=mm)) as ds:
        lat_t = np.asarray(ds["latitude"][:])
        tend = _f32(ds["tend_filtered"][:nt])
        if lat_t[0] < lat_t[-1]:
            tend, lat_t = tend[:, ::-1, :], lat_t[::-1]
        assert tend.shape[0] == nt, (tend.shape, nt)
    with netCDF4.Dataset(RAD_FMT.format(y=yy, m=mm)) as ds:
        lat_r = np.asarray(ds["latitude"][:])
        assert lat_r[0] > lat_r[-1], "rad expected descending"
        swabs = (_f32(ds["tsr"][:nt]) - _f32(ds["ssr"][:nt])) / np.float32(3600.0)
        olr = _f32(ds["ttr"][:nt]) / np.float32(3600.0)
    assert np.allclose(lat_r, lat_t) and np.allclose(lat_r, lat_v[::-1]) and np.allclose(lat_r[1:-1], lat_int)

    f_dhdt = integ(tend, lat_r)
    f_shf = integ(div - swabs - olr + tend, lat_r)
    del div, swabs, olr, tend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.nc")
    with netCDF4.Dataset(str(tmp), "w", format="NETCDF4_CLASSIC") as w:
        w.createDimension("lon", lon_int.size)
        w.createDimension("lat", lat_int.size)
        w.createDimension("time", nt)
        v = w.createVariable("lon", "f4", ("lon",)); v[:] = lon_int; v.units = "Degrees East"
        v = w.createVariable("lat", "f4", ("lat",)); v[:] = lat_int; v.units = "Degrees North"
        v = w.createVariable("time", "f4", ("time",)); v[:] = np.arange(nt); v.units = "6-hourly steps"
        v = w.createVariable("F_Dhdt_final", "f4", ("time", "lat", "lon")); v[:] = f_dhdt.astype(np.float32)
        v.units = "PW"; v.long_name = "poleward-integrated MSE storage (corrected, tend_filtered_3)"
        v = w.createVariable("F_Shf_final", "f4", ("time", "lat", "lon")); v[:] = f_shf.astype(np.float32)
        v.units = "PW"; v.long_name = "poleward-integrated residual surface flux with corrected storage"
        w.history = "integrate_fluxes_v3.py (production formula of make_TE_int.py), %s" % _time.strftime("%Y-%m-%d")
    tmp.rename(out_path)
    _LOG.info("saved %s (%.0fs)", out_path, _time.time() - t0)


if __name__ == "__main__":
    yy = int(sys.argv[1])
    months = [int(m) for m in sys.argv[2:]] or list(range(1, 13))
    for mm in months:
        process_month(yy, mm)
