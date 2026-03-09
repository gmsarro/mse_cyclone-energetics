from __future__ import annotations

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray

import cyclone_energetics.constants as constants
import cyclone_energetics.gridded_data as gridded_data

_LOG = logging.getLogger(__name__)

_STORMTRACK_INTERP_NPTS: int = 25600


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
    with netCDF4.Dataset(str(stormtrack_nc)) as stormtrack_dataset:
        lat_dim = gridded_data.resolve_dimension_name(
            stormtrack_dataset, standard_name="latitude"
        )
        lat_st = stormtrack_dataset[lat_dim][:]
        f_te_final = np.array(stormtrack_dataset["F_TE_final"][0])
        f_te_zon = np.mean(f_te_final, axis=2)

    nlat = lat_st.shape[0]
    x_d = np.linspace(0, nlat - 1, nlat)
    x3_d = np.linspace(0, nlat - 1, _STORMTRACK_INTERP_NPTS)
    lat_hi = scipy.interpolate.interp1d(x_d, lat_st)(x3_d)

    y_d = np.linspace(0, 12, 12)
    y3_d = np.linspace(0, 12, 12)
    f_te_zon_int = scipy.interpolate.RectBivariateSpline(
        y_d, x_d, f_te_zon
    )(y3_d, x3_d)

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
    intense_full_path: typing.Optional[pathlib.Path],
    all_full_path: typing.Optional[pathlib.Path],
    stormtrack_path: pathlib.Path,
    output_path: pathlib.Path,
    compress_level: int = 4,
) -> None:
    max_te_sh_position = _compute_stormtrack_position(
        stormtrack_nc=stormtrack_path
    )

    with xarray.open_dataset(str(intense_composite_path)) as intense_composite, \
         xarray.open_dataset(str(regular_composite_path)) as regular_composite:
        te_int_pw = intense_composite["composite_TE"].values
        shf_int_pw = intense_composite["composite_Shf"].values
        cnt_int_track = intense_composite["count"].values.astype(float)
        x_int = intense_composite["x"].values
        y_int = intense_composite["y"].values

        te_wk_pw = regular_composite["composite_TE"].values
        shf_wk_pw = regular_composite["composite_Shf"].values
        cnt_wk_track = regular_composite["count"].values.astype(float)

        if "composite_Z" in intense_composite and "composite_Shf_wm" in intense_composite:
            z_int = intense_composite["composite_Z"].values
            vo_int = intense_composite["composite_VO"].values
            shf_int = intense_composite["composite_Shf_wm"].values
            t_int = intense_composite["composite_T"].values
            q_int = intense_composite["composite_Q"].values
            n_int_full = cnt_int_track

            z_wk = regular_composite["composite_Z"].values
            vo_wk = regular_composite["composite_VO"].values
            shf_wk = regular_composite["composite_Shf_wm"].values
            t_wk = regular_composite["composite_T"].values
            q_wk = regular_composite["composite_Q"].values
            n_wk_full = cnt_wk_track
            n_all_full = cnt_int_track + cnt_wk_track

            lat_ref = y_int
            lon_ref = x_int
        elif intense_full_path is not None and all_full_path is not None:
            with netCDF4.Dataset(str(all_full_path)) as all_composites_dataset:
                lat_dim = gridded_data.resolve_dimension_name(
                    all_composites_dataset, standard_name="latitude"
                )
                lon_dim = gridded_data.resolve_dimension_name(
                    all_composites_dataset, standard_name="longitude"
                )
                lat_ref = all_composites_dataset[lat_dim][:]
                lon_ref = all_composites_dataset[lon_dim][:]
                z_all = _read_monthly_3d(dataset=all_composites_dataset, name="composite_Z")
                vo_all = _read_monthly_3d(dataset=all_composites_dataset, name="composite_VO")
                shf_all = _read_monthly_3d(dataset=all_composites_dataset, name="composite_Shf")
                t_all = _read_monthly_3d(dataset=all_composites_dataset, name="composite_T")
                q_all = _read_monthly_3d(dataset=all_composites_dataset, name="composite_Q")
                n_all_full = _read_count_12(dataset=all_composites_dataset)

            with netCDF4.Dataset(str(intense_full_path)) as intense_dataset:
                z_int = _read_monthly_3d(dataset=intense_dataset, name="composite_Z")
                vo_int = _read_monthly_3d(dataset=intense_dataset, name="composite_VO")
                shf_int = _read_monthly_3d(dataset=intense_dataset, name="composite_Shf")
                t_int = _read_monthly_3d(dataset=intense_dataset, name="composite_T")
                q_int = _read_monthly_3d(dataset=intense_dataset, name="composite_Q")
                n_int_full = _read_count_12(dataset=intense_dataset)

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
            t_wk = (
                t_all * (n_all_full[:, None, None] / n_wk_full[:, None, None])
                - t_int * (n_int_full[:, None, None] / n_wk_full[:, None, None])
            )
            q_wk = (
                q_all * (n_all_full[:, None, None] / n_wk_full[:, None, None])
                - q_int * (n_int_full[:, None, None] / n_wk_full[:, None, None])
            )
        else:
            raise ValueError(
                "Either unified files with W/m² fields or separate "
                "intense_full_path and all_full_path required"
            )

    wgt = _build_weight_cube(
        lat_row_base=y_int, max_position_12=max_te_sh_position, nx=te_int_pw.shape[-1]
    )

    i_l_int = te_int_pw / wgt
    i_shf_int = shf_int_pw / wgt
    i_l_wk = te_wk_pw / wgt
    i_shf_wk = shf_wk_pw / wgt

    shf_local_int = shf_int
    shf_local_wk = shf_wk

    i_l = np.stack([i_l_int, i_l_wk], axis=0).astype(np.float32)
    i_shf = np.stack([i_shf_int, i_shf_wk], axis=0).astype(np.float32)
    shf_local = np.stack([shf_local_int, shf_local_wk], axis=0).astype(np.float32)
    z_arr = np.stack([z_int, z_wk], axis=0).astype(np.float32)
    vo_arr = np.stack([vo_int, vo_wk], axis=0).astype(np.float32)
    t_arr = np.stack([t_int, t_wk], axis=0).astype(np.float32)
    q_arr = np.stack([q_int, q_wk], axis=0).astype(np.float32)
    count_track = np.stack([cnt_int_track, cnt_wk_track], axis=0).astype(np.float64)
    count_full = np.stack([n_int_full, n_wk_full], axis=0).astype(np.float64)

    months = np.arange(1, 13, dtype=np.int32)
    category_names = np.array(["intense", "weak"], dtype=object)
    month_names = np.array(constants.MONTH_NAMES, dtype=object)

    output_dataset = xarray.Dataset(
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
            "T": (
                ("category", "month", "lat", "lon"),
                t_arr,
                {"units": "K", "long_name": "2-m temperature composite"},
            ),
            "Q": (
                ("category", "month", "lat", "lon"),
                q_arr,
                {"units": "kg kg-1", "long_name": "Specific humidity at 850 hPa composite"},
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
            "lon": (("lon",), np.asarray(lon_ref).astype(np.float32)),
            "lat": (("lat",), np.asarray(lat_ref).astype(np.float32)),
        },
        attrs={
            "title": "Monthly condensed cyclone-centered composites",
        },
    )

    comp = {"zlib": True, "complevel": compress_level}
    encoding: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    for v in output_dataset.data_vars:
        if output_dataset[v].dtype.kind in ("f", "i"):
            encoding[v] = {**comp}

    output_dataset.to_netcdf(
        str(output_path), engine="netcdf4", format="NETCDF4", encoding=encoding
    )

    _LOG.info("Wrote condensed file: %s", output_path)
