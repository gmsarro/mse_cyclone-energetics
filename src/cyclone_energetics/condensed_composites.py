import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray as xr

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)


def _read_monthly_3d(
    *,
    dataset: netCDF4.Dataset,
    name: str,
) -> npt.NDArray:
    arr = np.array(dataset[name][:])
    if arr.ndim != 3 or arr.shape[0] != 12:
        raise ValueError(
            "%s must be monthly (12, y, x). Got shape %s" % (name, arr.shape)
        )
    return arr


def _read_count_12(
    *,
    dataset: netCDF4.Dataset,
) -> npt.NDArray:
    if "count" in dataset.variables:
        c = np.array(dataset["count"][:], float)
    elif "number" in dataset.variables:
        c = np.array(dataset["number"][:], float)
    else:
        raise KeyError("No 'count' or 'number' found for monthly counts.")
    if c.shape[0] != 12:
        raise ValueError("count/number must be length-12. Got shape %s" % str(c.shape))
    return c


def _compute_stormtrack_position(
    *,
    stormtrack_nc: pathlib.Path,
) -> npt.NDArray:
    with netCDF4.Dataset(str(stormtrack_nc)) as ds:
        lat_st = ds["lat"][:]
        f_te_final = ds["F_TE_final"][0, :, :, :]
        f_te_zon = np.mean(f_te_final, axis=2)

    nlat = lat_st.shape[0]
    x_d = np.linspace(0, nlat - 1, nlat)
    x3_d = np.linspace(0, nlat - 1, 25600)
    lat_hi = scipy.interpolate.interp1d(x_d, lat_st)(x3_d)

    y_d = np.linspace(0, 12, 12)
    y3_d = np.linspace(0, 12, 12)
    f_te_zon_int = scipy.interpolate.interp2d(
        x_d, y_d, f_te_zon, kind="cubic"
    )(x3_d, y3_d)

    stormtrack_sh = np.argmin(f_te_zon_int, axis=1)
    return lat_hi[stormtrack_sh]


def _build_weight_cube(
    *,
    lat_row_base: npt.NDArray,
    max_position_12: npt.NDArray,
    nx: int,
) -> npt.NDArray:
    lat_row_base = np.asarray(lat_row_base, float)
    max_position_12 = np.asarray(max_position_12, float).reshape(12)
    ny = lat_row_base.shape[0]
    weight = np.empty((12, ny, nx), float)
    for i in range(12):
        row_deg = lat_row_base + max_position_12[i]
        lat2d = np.tile(row_deg[:, None], (1, nx))
        weight[i] = (
            constants.EARTH_RADIUS
            * np.cos(np.deg2rad(lat2d))
            * 2.0
            * np.pi
            * constants.PW_TO_WM2_FACTOR
        )
    return weight


def create_condensed_composites(
    *,
    intense_composite_path: pathlib.Path,
    regular_composite_path: pathlib.Path,
    intense_full_path: pathlib.Path,
    all_full_path: pathlib.Path,
    stormtrack_path: pathlib.Path,
    output_path: pathlib.Path,
    compress_level: int = 4,
) -> None:
    max_te_sh_position = _compute_stormtrack_position(
        stormtrack_nc=stormtrack_path
    )

    ds_int = xr.open_dataset(str(intense_composite_path))
    ds_wk = xr.open_dataset(str(regular_composite_path))

    te_int_pw = ds_int["composite_TE"].values
    shf_int_pw = ds_int["composite_Shf"].values
    cnt_int_track = ds_int["count"].values.astype(float)
    x_int = ds_int["x"].values
    y_int = ds_int["y"].values

    te_wk_pw = ds_wk["composite_TE"].values
    shf_wk_pw = ds_wk["composite_Shf"].values
    cnt_wk_track = ds_wk["count"].values.astype(float)

    with netCDF4.Dataset(str(all_full_path)) as dsa:
        lat_ref = dsa["lat"][:]
        lon_ref = dsa["lon"][:]
        z_all = _read_monthly_3d(dataset=dsa, name="composite_Z")
        vo_all = _read_monthly_3d(dataset=dsa, name="composite_VO")
        shf_all = _read_monthly_3d(dataset=dsa, name="composite_Shf")
        n_all_full = _read_count_12(dataset=dsa)

    with netCDF4.Dataset(str(intense_full_path)) as dsi:
        z_int = _read_monthly_3d(dataset=dsi, name="composite_Z")
        vo_int = _read_monthly_3d(dataset=dsi, name="composite_VO")
        shf_int = _read_monthly_3d(dataset=dsi, name="composite_Shf")
        n_int_full = _read_count_12(dataset=dsi)

    n_wk_full = n_all_full - n_int_full
    z_wk = (
        z_all * (n_all_full[:, None, None] / n_wk_full[:, None, None])
        - z_int * (n_int_full[:, None, None] / n_wk_full[:, None, None])
    )
    vo_wk = (
        vo_all * (n_all_full[:, None, None] / n_wk_full[:, None, None])
        - vo_int * (n_int_full[:, None, None] / n_wk_full[:, None, None])
    )
    shf_wk = (
        shf_all * (n_all_full[:, None, None] / n_wk_full[:, None, None])
        - shf_int * (n_int_full[:, None, None] / n_wk_full[:, None, None])
    )

    wgt = _build_weight_cube(
        lat_row_base=y_int, max_position_12=max_te_sh_position, nx=te_int_pw.shape[-1]
    )

    i_l_int = te_int_pw / wgt
    i_shf_int = shf_int_pw / wgt
    i_l_wk = te_wk_pw / wgt
    i_shf_wk = shf_wk_pw / wgt

    # SHF is computed as a residual from the MSE budget:
    # SHF = column_MSE - Swabs - OLR + dh/dt
    # Already in W/m^2 from the compositing step.
    shf_local_int = shf_int
    shf_local_wk = shf_wk

    i_l = np.stack([i_l_int, i_l_wk], axis=0).astype(np.float32)
    i_shf = np.stack([i_shf_int, i_shf_wk], axis=0).astype(np.float32)
    shf_local = np.stack([shf_local_int, shf_local_wk], axis=0).astype(np.float32)
    z_arr = np.stack([z_int, z_wk], axis=0).astype(np.float32)
    vo_arr = np.stack([vo_int, vo_wk], axis=0).astype(np.float32)
    count_track = np.stack([cnt_int_track, cnt_wk_track], axis=0).astype(np.float64)
    count_full = np.stack([n_int_full, n_wk_full], axis=0).astype(np.float64)

    months = np.arange(1, 13, dtype=np.int32)
    category_names = np.array(["intense", "weak"], dtype=object)
    month_names = np.array(constants.MONTH_NAMES, dtype=object)

    ds_out = xr.Dataset(
        data_vars={
            "I_L": (
                ("category", "month", "y", "x"),
                i_l,
                {"units": "W m-1", "long_name": "I_L (from TE composite)"},
            ),
            "I_SHF": (
                ("category", "month", "y", "x"),
                i_shf,
                {"units": "W m-1", "long_name": "I_SHF (from SHF composite)"},
            ),
            "SHF": (
                ("category", "month", "lat", "lon"),
                shf_local,
                {"units": "W m-2", "long_name": "Local SHF"},
            ),
            "Z": (
                ("category", "month", "lat", "lon"),
                z_arr,
                {"long_name": "Geopotential height composite"},
            ),
            "VO": (
                ("category", "month", "lat", "lon"),
                vo_arr,
                {"long_name": "Vorticity composite"},
            ),
            "count_track": (
                ("category", "month"),
                count_track,
                {"long_name": "Monthly cyclone counts (track composites)"},
            ),
            "count_full": (
                ("category", "month"),
                count_full,
                {"long_name": "Monthly cyclone counts (full-field composites)"},
            ),
            "count_all_full": (
                ("month",),
                n_all_full.astype(np.float64),
                {"long_name": "Monthly cyclone counts for all cyclones"},
            ),
        },
        coords={
            "category": (("category",), np.arange(2, dtype=np.int32)),
            "category_name": (("category",), category_names),
            "month": (("month",), months),
            "month_name": (("month",), month_names),
            "x": (("x",), x_int.astype(np.float32)),
            "y": (("y",), y_int.astype(np.float32)),
            "lon": (("lon",), lon_ref.astype(np.float32)),
            "lat": (("lat",), lat_ref.astype(np.float32)),
        },
        attrs={
            "title": "Monthly condensed cyclone-centered composites",
        },
    )

    comp = {"zlib": True, "complevel": compress_level}
    encoding = {}
    for v in ds_out.data_vars:
        if ds_out[v].dtype.kind in ("f", "i"):
            encoding[v] = {**comp}

    ds_out.to_netcdf(
        str(output_path), engine="netcdf4", format="NETCDF4", encoding=encoding
    )

    ds_int.close()
    ds_wk.close()
    ds_out.close()

    _LOG.info("Wrote condensed file: %s", output_path)
