"""Corrected MSE storage term: tendency of the column energy including the
column-mass change.

The previous storage computation (cyclone_energetics.computation.storage, and
the 2021 version behind tend_*_filtered.nc) differenced h = cp*T + Lv*q in
time level by level and integrated over pressure with a beta mask built from
the MONTHLY-MEAN surface pressure, i.e.
    S_old = int_0^{ps_mean} (dh/dt) dp/g .
The tendency of the actual column energy E(t) = int_0^{ps(t)} h dp/g also has
the lower-boundary term h_s (dps/dt)/g, which for moving/deepening cyclones is
O(+/-300 W/m2) (scripts/diag_mass_term.py). Here E(t) is built with the
instantaneous surface pressure in the beta mask and then differenced in time
(centered, forward/backward at the month ends, same stencil as before), so the
boundary term is included by construction:
    S_new[t] = (E[t+1] - E[t-1]) / (2 dt).

The vertical integral is a trapezoid over the layers above the surface plus an
exact partial bottom layer (column_integral), so that dE/dps = h(ps). The
beta-mask/trapezoid scheme of cyclone_energetics.gridded_data gives the
lowest level only half its weight and would retain only ~half of the mass
term; --selfcheck runs analytic tests. Data are read in contiguous time blocks
because the ERA5 files are NETCDF3 [time, level, lat, lon].

Usage: python compute_storage_masschange.py YEAR [MONTH ...] [--selfcheck]
Output: /project2/tas1/gmsarro/dh_dt_data_ERA5_v3/tend_YYYY_MM_3.nc (variable
'tend', W/m2, same grid/time as the ERA5 pressure-level files).
"""

import logging
import pathlib
import sys
import time as _time

import netCDF4
import numpy as np

sys.path.insert(0, "/project2/tas1/gmsarro/mse_cyclone-energetics")
import cyclone_energetics.constants as constants          # noqa: E402
import cyclone_energetics.gridded_data as gridded_data    # noqa: E402

ERA5_DIR = pathlib.Path("/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5")
OUT_DIR = pathlib.Path("/project2/tas1/gmsarro/dh_dt_data_ERA5_v3")
TIME_BLOCK = 8
CPD, LV, G = constants.CPD, constants.LATENT_HEAT_VAPORIZATION, constants.GRAVITY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
_LOG = logging.getLogger("storage_v3")


def column_integral(h: np.ndarray, ps: np.ndarray, plev: np.ndarray) -> np.ndarray:
    """int_0^{ps} h dp for h (nt, nlev, nlat, nlon) on ascending pressure levels
    plev (Pa) and surface pressure ps (nt, nlat, nlon).

    Trapezoid over every layer [p_k, p_{k+1}] that lies entirely above the
    surface, plus the exact integral of the linearly-interpolated h over the
    partial layer [p_K, ps] (K = lowest level above the surface). Below the
    lowest level (ps > p_{-1}) h is extrapolated as constant. This represents
    the column mass exactly, so dE/dps = h(ps) (the old beta/trapezoid scheme
    only gave 0.5*h at the lowest level and thus missed half of the mass term).
    """
    nlev = plev.size
    p4 = plev[None, :, None, None]
    ps4 = ps[:, None, :, :]
    dp = np.diff(plev)[None, :, None, None]
    full = (p4[:, 1:] <= ps4)                                     # layer k fully above the surface
    e = np.sum(0.5 * (h[:, :-1] + h[:, 1:]) * dp * full, axis=1)
    e += h[:, 0] * plev[0]                                        # mass above the top level (constant h)

    k_last = np.sum(plev[None, :, None, None] <= ps4, axis=1) - 1   # lowest level above the surface
    k_last = np.clip(k_last, 0, nlev - 1)
    h_k = np.take_along_axis(h, k_last[:, None], axis=1)[:, 0]
    k_next = np.minimum(k_last + 1, nlev - 1)
    h_next = np.take_along_axis(h, k_next[:, None], axis=1)[:, 0]
    p_k = plev[k_last]
    p_next = plev[k_next]
    dps = ps - p_k                                                # >= 0
    interior = k_last < nlev - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(interior, (h_next - h_k) / (p_next - p_k), 0.0)
    e += h_k * dps + 0.5 * slope * dps * dps
    return e


def column_energy(t: np.ndarray, q: np.ndarray, ps: np.ndarray, plev: np.ndarray) -> np.ndarray:
    """E = int_0^{ps} (cp T + Lv q) dp / g (J/m2) for a block; t, q (nt, nlev, nlat, nlon)."""
    h = CPD * t
    h += LV * q
    if plev[1] < plev[0]:
        h, plev = h[:, ::-1], plev[::-1]
    return column_integral(h, ps, plev) / G


def selfcheck(year: int, month: str) -> None:
    """Analytic checks of column_integral, then a comparison with the package's
    beta/trapezoid scheme on a real slab (differences expected: bottom layer)."""
    import xarray
    rng = np.random.default_rng(0)
    plev_pa = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350,
                        400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950,
                        975, 1000], dtype=np.float64) * 100.0
    ps = rng.uniform(600e2, 1040e2, size=(2, 5, 7))
    a, b = 3.0e5, -0.4
    h_lin = a + b * plev_pa[None, :, None, None] * np.ones((2, 1, 5, 7))
    exact = a * ps + 0.5 * b * ps ** 2
    e_num = column_integral(h_lin, ps, plev_pa)
    e_num[ps > plev_pa[-1]] = np.nan   # constant extrapolation below 1000 hPa is not linear
    exact[ps > plev_pa[-1]] = np.nan
    print("selfcheck: linear h, max rel err = %.2e" % np.nanmax(np.abs(e_num - exact) / np.abs(exact)))
    h_c = 3.0e5 * np.ones_like(h_lin)
    print("selfcheck: const h (incl. ps>1000hPa), max rel err = %.2e" % np.max(
        np.abs(column_integral(h_c, ps, plev_pa) - 3.0e5 * ps) / (3.0e5 * ps)))
    eps = 1.0
    dEdps = (column_integral(h_lin, ps + eps, plev_pa) - column_integral(h_lin, ps - eps, plev_pa)) / (2 * eps)
    h_at_ps = np.where(ps > plev_pa[-1], a + b * plev_pa[-1], a + b * ps)
    print("selfcheck: dE/dps vs h(ps), max rel err = %.2e" % np.max(np.abs(dEdps - h_at_ps) / np.abs(h_at_ps)))
    kw = dict(data_directory=ERA5_DIR, year=year, month=month)
    t_path = gridded_data.resolve_path(field="temperature", **kw)
    q_path = gridded_data.resolve_path(field="specific_humidity", **kw)
    ps_path = gridded_data.resolve_path(field="surface_pressure", **kw)
    sl, ts = slice(300, 312), slice(0, 3)
    plev_pa = gridded_data.read_pressure_levels(q_path)
    plev = xarray.DataArray(plev_pa, dims=["level"], coords={"level": plev_pa})
    ps = gridded_data.open_field(ps_path, variable="sp", latitude_slice=sl, time_slice=ts)
    tt = gridded_data.open_field(t_path, variable="t", latitude_slice=sl, time_slice=ts).assign_coords(level=plev_pa)
    qq = gridded_data.open_field(q_path, variable="q", latitude_slice=sl, time_slice=ts).assign_coords(level=plev_pa)
    beta = gridded_data.compute_beta_mask(pressure_levels=plev, surface_pressure=ps)
    e_x = ((CPD * tt + LV * qq) * beta).integrate("level").transpose("time", "latitude", "longitude").values / G
    e_n = column_energy(tt.values, qq.values, ps.values, plev_pa)
    print("selfcheck: real slab, mean E = %.4e J/m2; package beta/trapezoid scheme differs by "
          "mean %.2e, max %.2e J/m2 (expected: bottom-layer treatment)" % (
              e_n.mean(), (e_x - e_n).mean(), np.abs(e_x - e_n).max()))


def process_month(year: int, month: str) -> pathlib.Path:
    out_path = OUT_DIR / ("tend_%d_%s_3.nc" % (year, month))
    if out_path.exists():
        _LOG.info("exists, skipping: %s", out_path)
        return out_path
    kw = dict(data_directory=ERA5_DIR, year=year, month=month)
    t_path = gridded_data.resolve_path(field="temperature", **kw)
    q_path = gridded_data.resolve_path(field="specific_humidity", **kw)
    ps_path = gridded_data.resolve_path(field="surface_pressure", **kw)
    t0 = _time.time()

    with netCDF4.Dataset(str(t_path)) as dt_, netCDF4.Dataset(str(q_path)) as dq_, \
            netCDF4.Dataset(str(ps_path)) as dp_:
        for ds in (dt_, dq_, dp_):
            ds.set_auto_mask(False)  # keep scale/offset decoding, no masked arrays
        lat = np.asarray(dt_["latitude"][:], dtype=np.float64)
        lon = np.asarray(dt_["longitude"][:], dtype=np.float64)
        plev = np.asarray(dt_["level"][:], dtype=np.float64) * 100.0
        assert np.allclose(plev, np.asarray(dq_["level"][:], dtype=np.float64) * 100.0)
        time_var = dt_["time"]
        time_vals = np.asarray(time_var[:], dtype=np.float64)
        time_attrs = {k: time_var.getncattr(k) for k in time_var.ncattrs()}
        assert np.allclose(time_vals, np.asarray(dp_["time"][:], dtype=np.float64)), "time mismatch t/ps"
        n_time = time_vals.size
        dt = float(time_vals[1] - time_vals[0]) * 3600.0
        ps_all = np.asarray(dp_["sp"][:], dtype=np.float64)

        energy = np.zeros((n_time, lat.size, lon.size), dtype=np.float64)
        for b0 in range(0, n_time, TIME_BLOCK):
            b1 = min(b0 + TIME_BLOCK, n_time)
            tb = np.asarray(dt_["t"][b0:b1], dtype=np.float64)
            qb = np.asarray(dq_["q"][b0:b1], dtype=np.float64)
            energy[b0:b1] = column_energy(tb, qb, ps_all[b0:b1], plev)
            del tb, qb
        _LOG.info("  %d-%s column energy done (%.0fs)", year, month, _time.time() - t0)

    tend = np.empty_like(energy)
    tend[1:-1] = (energy[2:] - energy[:-2]) / (2.0 * dt)
    tend[0] = (energy[1] - energy[0]) / dt
    tend[-1] = (energy[-1] - energy[-2]) / dt
    del energy

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.nc")
    with netCDF4.Dataset(str(tmp), "w", format="NETCDF4") as out:
        out.createDimension("time", n_time)
        out.createDimension("latitude", lat.size)
        out.createDimension("longitude", lon.size)
        v = out.createVariable("time", "f8", ("time",))
        v[:] = time_vals
        v.setncatts(time_attrs)
        v = out.createVariable("latitude", "f8", ("latitude",))
        v[:] = lat
        v.units = "degrees_north"
        v = out.createVariable("longitude", "f8", ("longitude",))
        v[:] = lon
        v.units = "degrees_east"
        v = out.createVariable("tend", "f4", ("time", "latitude", "longitude"), zlib=True, complevel=1)
        v[:] = tend.astype(np.float32)
        v.units = "W m-2"
        v.long_name = ("time tendency of the column moist static energy (cp T + Lv q) "
                       "integrated to the instantaneous surface pressure")
        v.method = ("E(t) = int_0^{ps(t)} h(t) dp/g on ERA5 pressure levels (trapezoid above the "
                    "surface + exact linear partial bottom layer); centered time difference "
                    "(forward/backward at month ends); includes the column-mass change term "
                    "h_s dps/dt / g omitted by the fixed-level tendency (tend_*_2.nc)")
        out.history = "compute_storage_masschange.py, %s" % _time.strftime("%Y-%m-%d")
    tmp.rename(out_path)
    _LOG.info("saved %s (%.0fs)", out_path, _time.time() - t0)
    return out_path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selfcheck"]
    year = int(args[0])
    months = args[1:] or constants.MONTH_STRINGS
    if "--selfcheck" in sys.argv:
        selfcheck(year, "%02d" % int(months[0]))
    for m in months:
        process_month(year, "%02d" % int(m))
