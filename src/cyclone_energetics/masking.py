from __future__ import annotations

import logging
import pathlib

import matplotlib
import matplotlib.path
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import cyclone_energetics.constants as constants
import cyclone_energetics.geometry as geometry

matplotlib.use("agg")
_LOG = logging.getLogger(__name__)

_POLAR_GRID_SPACING: float = 1.0
_TRACK_GRID_SPACING: float = 1.5


def _load_track_npz(
    *,
    track_directory: pathlib.Path,
    track_type: str,
    hemisphere: str,
) -> dict[str, npt.NDArray]:
    fname = "%s_VO_anom_T42_ERA5_1979_2018_all%s_long.npz" % (
        track_type,
        hemisphere,
    )
    npz = np.load(str(track_directory / fname), allow_pickle=True)
    return dict(npz)


def _build_grids(
    *,
    hemisphere: str,
    y1: int,
    y2: int,
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    lats = np.arange(-90, 90 + _TRACK_GRID_SPACING, _TRACK_GRID_SPACING)
    latt = lats[y1:y2]
    lons = np.arange(0, 360, _TRACK_GRID_SPACING)
    xx_ll, yy_ll = np.meshgrid(lons, latt)
    p_xx, p_yy = geometry.lonlat_to_polar(
        longitude=xx_ll, latitude=yy_ll, hemisphere=hemisphere
    )
    p_points = np.vstack((p_xx.ravel(), p_yy.ravel())).T
    xx_polar, yy_polar = geometry.build_polar_mesh(grid_spacing=_POLAR_GRID_SPACING)
    mesh_points = np.vstack((xx_polar.ravel(), yy_polar.ravel())).T
    return latt, lons, p_xx, p_points, xx_polar, yy_polar, mesh_points, p_yy


def _process_single_timestep(
    *,
    vorticity: npt.NDArray,
    vorticity_threshold: float,
    track_data_c: dict[str, npt.NDArray],
    track_data_a: dict[str, npt.NDArray],
    sec_c: npt.NDArray,
    sec_a: npt.NDArray,
    p_points: npt.NDArray,
    mesh_points: npt.NDArray,
    xx_polar: npt.NDArray,
    yy_polar: npt.NDArray,
    hemisphere: str,
    p_xx_shape: tuple[int, ...],
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    new_vor = scipy.interpolate.griddata(
        p_points, vorticity.ravel(), mesh_points
    ).reshape(np.shape(xx_polar))

    results = {}
    for k, label in [("C", "cyclone"), ("A", "anticyclone")]:
        if k == "C":
            cvu = vorticity_threshold
            over_mask = new_vor.ravel() >= vorticity_threshold
            sec = sec_c
            td = track_data_c
        else:
            cvu = -vorticity_threshold
            over_mask = new_vor.ravel() <= -vorticity_threshold
            sec = sec_a
            td = track_data_a

        mask_field = np.zeros(np.shape(new_vor))
        flag_field = np.zeros(np.shape(new_vor))
        intensity_field = np.zeros(np.shape(new_vor))
        intensity_change_field = np.zeros(np.shape(new_vor))

        x_k, y_k = geometry.lonlat_to_polar(
            longitude=td["lonc"][sec],
            latitude=td["latc"][sec],
            hemisphere=hemisphere,
        )
        pos_k = np.vstack((x_k, y_k))
        ti_k = td["eventarray"][sec]
        rev_k = td["Revc"][sec]
        grc_k = td["grc"][sec]

        contour_set = plt.contour(
            xx_polar, yy_polar, new_vor, [cvu], colors="None"
        )
        contour_lines = contour_set.allsegs[0]
        n_contours = len(contour_lines)

        relationship_matrix = np.zeros((len(ti_k), n_contours))
        for j in range(len(ti_k)):
            xy = pos_k[:, j]
            for i in range(n_contours):
                line = matplotlib.path.Path(contour_lines[i], closed=True)
                if line.contains_point(xy, radius=2 * _POLAR_GRID_SPACING):
                    relationship_matrix[j, i] = 1

        offset = int(90 / _POLAR_GRID_SPACING)
        for i in range(n_contours):
            relate = np.where(relationship_matrix[:, i] == 1)[0]
            if len(relate) == 0:
                continue

            line = matplotlib.path.Path(contour_lines[i], closed=True)
            inner = line.contains_points(mesh_points)
            point_inner = mesh_points[inner & over_mask]

            if len(relate) == 1:
                x0, y0 = float(x_k[relate[0]]), float(y_k[relate[0]])
                dists = geometry.polar_mesh_to_km_distance(
                    x0=x0, y0=y0, grid_points=point_inner, hemisphere=hemisphere
                )
                hit = np.where(dists <= constants.DISTANCE_THRESHOLD)[0]
                xyhit = (point_inner[hit, :] / _POLAR_GRID_SPACING).astype(int)
                flag_field[xyhit[:, 1] + offset, xyhit[:, 0] + offset] = 1
                mask_field[xyhit[:, 1] + offset, xyhit[:, 0] + offset] = ti_k[
                    relate[0]
                ]
                intensity_field[xyhit[:, 1] + offset, xyhit[:, 0] + offset] = rev_k[
                    relate[0]
                ]
                intensity_change_field[
                    xyhit[:, 1] + offset, xyhit[:, 0] + offset
                ] = grc_k[relate[0]]
            else:
                dists_all = np.zeros((point_inner.shape[0], len(relate)))
                for j in range(len(relate)):
                    x0, y0 = float(x_k[relate[j]]), float(y_k[relate[j]])
                    dists_all[:, j] = geometry.polar_mesh_to_km_distance(
                        x0=x0,
                        y0=y0,
                        grid_points=point_inner,
                        hemisphere=hemisphere,
                    )
                for j in range(len(relate)):
                    hit = np.where(
                        (np.min(dists_all, axis=1) <= constants.DISTANCE_THRESHOLD)
                        & (np.argmin(dists_all, axis=1) == j)
                    )[0]
                    xyhit = (point_inner[hit, :] / _POLAR_GRID_SPACING).astype(int)
                    flag_field[xyhit[:, 1] + offset, xyhit[:, 0] + offset] = 1
                    mask_field[
                        xyhit[:, 1] + offset, xyhit[:, 0] + offset
                    ] = ti_k[relate[j]]
                    intensity_field[
                        xyhit[:, 1] + offset, xyhit[:, 0] + offset
                    ] = rev_k[relate[j]]
                    intensity_change_field[
                        xyhit[:, 1] + offset, xyhit[:, 0] + offset
                    ] = grc_k[relate[j]]

        plt.close("all")
        results[k] = {
            "mask": mask_field,
            "flag": flag_field,
            "intensity": intensity_field,
            "intensity_change": intensity_change_field,
        }

    mask_c = scipy.interpolate.griddata(
        mesh_points, results["C"]["mask"].ravel(), p_points, method="nearest"
    ).reshape(p_xx_shape)
    mask_a = scipy.interpolate.griddata(
        mesh_points, results["A"]["mask"].ravel(), p_points, method="nearest"
    ).reshape(p_xx_shape)
    flag_c = scipy.interpolate.griddata(
        mesh_points, results["C"]["flag"].ravel(), p_points, method="nearest"
    ).reshape(p_xx_shape)
    flag_a = scipy.interpolate.griddata(
        mesh_points, results["A"]["flag"].ravel(), p_points, method="nearest"
    ).reshape(p_xx_shape)
    int_c = scipy.interpolate.griddata(
        mesh_points, results["C"]["intensity"].ravel(), p_points, method="nearest"
    ).reshape(p_xx_shape)
    int_a = scipy.interpolate.griddata(
        mesh_points, results["A"]["intensity"].ravel(), p_points, method="nearest"
    ).reshape(p_xx_shape)
    int_del_c = scipy.interpolate.griddata(
        mesh_points,
        results["C"]["intensity_change"].ravel(),
        p_points,
        method="nearest",
    ).reshape(p_xx_shape)
    int_del_a = scipy.interpolate.griddata(
        mesh_points,
        results["A"]["intensity_change"].ravel(),
        p_points,
        method="nearest",
    ).reshape(p_xx_shape)

    return mask_c, mask_a, flag_c, flag_a, int_c, int_a, int_del_c, int_del_a


def create_cyclone_masks(
    *,
    hemisphere: str,
    year_start: int,
    year_end: int,
    vorticity_directory: pathlib.Path,
    track_directory: pathlib.Path,
    output_directory: pathlib.Path,
    vorticity_threshold: float = constants.VORTICITY_THRESHOLD,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    is_nh = hemisphere == "NH"
    y1, y2 = (60, 121) if is_nh else (0, 61)
    vfac = 1 if is_nh else -1

    cyclone_data = _load_track_npz(
        track_directory=track_directory,
        track_type="TRACK",
        hemisphere=hemisphere,
    )
    anticyclone_data = _load_track_npz(
        track_directory=track_directory,
        track_type="ANTIC",
        hemisphere=hemisphere,
    )

    (
        latt,
        lons,
        p_xx,
        p_points,
        xx_polar,
        yy_polar,
        mesh_points,
        _p_yy,
    ) = _build_grids(hemisphere=hemisphere, y1=y1, y2=y2)

    long_diy_c = cyclone_data["timec"] % constants.TIMESTEPS_PER_YEAR
    long_diy_a = anticyclone_data["timec"] % constants.TIMESTEPS_PER_YEAR

    years = np.arange(year_start, year_end)
    for year in years:
        _LOG.info("Creating masks: hemisphere=%s year=%s", hemisphere, year)
        with netCDF4.Dataset(
            str(vorticity_directory / ("VO.anom.T42.%d.v146.nc" % year))
        ) as ds_vo:
            vo_data = ds_vo["VO"][:, y1:y2, :].data * vfac

        n_time = vo_data.shape[0]
        all_mask_c = np.zeros(vo_data.shape)
        all_mask_a = np.zeros(vo_data.shape)
        all_flag_c = np.zeros(vo_data.shape)
        all_flag_a = np.zeros(vo_data.shape)
        all_int_c = np.zeros(vo_data.shape)
        all_int_a = np.zeros(vo_data.shape)
        all_int_del_c = np.zeros(vo_data.shape)
        all_int_del_a = np.zeros(vo_data.shape)

        longyear_c = cyclone_data["longyear"]
        longyear_a = anticyclone_data["longyear"]

        for t in range(n_time):
            sec_c = np.where(
                (longyear_c == year) & (long_diy_c == t)
            )[0]
            sec_a = np.where(
                (longyear_a == year) & (long_diy_a == t)
            )[0]

            (
                mask_c,
                mask_a,
                flag_c,
                flag_a,
                int_c,
                int_a,
                int_del_c,
                int_del_a,
            ) = _process_single_timestep(
                vorticity=vo_data[t, :, :],
                vorticity_threshold=vorticity_threshold,
                track_data_c=cyclone_data,
                track_data_a=anticyclone_data,
                sec_c=sec_c,
                sec_a=sec_a,
                p_points=p_points,
                mesh_points=mesh_points,
                xx_polar=xx_polar,
                yy_polar=yy_polar,
                hemisphere=hemisphere,
                p_xx_shape=p_xx.shape,
            )

            all_mask_c[t, :, :] = mask_c
            all_mask_a[t, :, :] = mask_a
            all_flag_c[t, :, :] = flag_c
            all_flag_a[t, :, :] = flag_a
            all_int_c[t, :, :] = int_c
            all_int_a[t, :, :] = int_a
            all_int_del_c[t, :, :] = int_del_c
            all_int_del_a[t, :, :] = int_del_a

        _save_mask_file(
            output_path=output_directory / ("MASK_%s_%d.nc" % (hemisphere, year)),
            latt=latt,
            lons=lons,
            mask_c=all_mask_c,
            mask_a=all_mask_a,
            flag_c=all_flag_c,
            flag_a=all_flag_a,
            int_c=all_int_c,
            int_a=all_int_a,
            int_del_c=all_int_del_c,
            int_del_a=all_int_del_a,
        )
        _LOG.info("Saved mask file for year %s", year)


def _save_mask_file(
    *,
    output_path: pathlib.Path,
    latt: npt.NDArray,
    lons: npt.NDArray,
    mask_c: npt.NDArray,
    mask_a: npt.NDArray,
    flag_c: npt.NDArray,
    flag_a: npt.NDArray,
    int_c: npt.NDArray,
    int_a: npt.NDArray,
    int_del_c: npt.NDArray,
    int_del_a: npt.NDArray,
) -> None:
    with netCDF4.Dataset(str(output_path), "w", format="NETCDF3_CLASSIC") as wfile:
        wfile.createDimension("lon", len(lons))
        wfile.createDimension("lat", len(latt))
        wfile.createDimension("time", None)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))

        mc = wfile.createVariable("mask_C", "f4", ("time", "lat", "lon"))
        ma = wfile.createVariable("mask_A", "f4", ("time", "lat", "lon"))
        ic = wfile.createVariable("intensity_C", "f4", ("time", "lat", "lon"))
        ia = wfile.createVariable("intensity_A", "f4", ("time", "lat", "lon"))
        ic_del = wfile.createVariable("intensity_change_C", "f4", ("time", "lat", "lon"))
        ia_del = wfile.createVariable("intensity_change_A", "f4", ("time", "lat", "lon"))
        fc = wfile.createVariable("flag_C", "f4", ("time", "lat", "lon"))
        fa = wfile.createVariable("flag_A", "f4", ("time", "lat", "lon"))

        lat_var.units = "Degrees North"
        lat_var.axis = "Y"
        lon_var.units = "Degrees East"
        lon_var.axis = "X"
        time_var.units = "6 hours from 0101 00 UTC"
        time_var.axis = "T"

        lat_var[:] = latt
        lon_var[:] = lons
        time_var[:] = np.arange(constants.TIMESTEPS_PER_YEAR)

        mc[:, :, :] = mask_c
        ma[:, :, :] = mask_a
        ic[:, :, :] = int_c
        ia[:, :, :] = int_a
        ic_del[:, :, :] = int_del_c
        ia_del[:, :, :] = int_del_a
        fc[:, :, :] = flag_c
        fa[:, :, :] = flag_a
