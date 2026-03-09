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
    track_filename_pattern: str = "{track_type}_VO_anom_T42_ERA5_{hemisphere}_long.npz",
) -> dict[str, npt.NDArray[np.float64]]:
    fname = track_filename_pattern.format(
        track_type=track_type,
        hemisphere=hemisphere,
    )
    npz = np.load(str(track_directory / fname), allow_pickle=True)
    return dict(npz)


def _build_grids(
    *,
    hemisphere: str,
    y1: int,
    y2: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    latitude_full = np.arange(-90, 90 + _TRACK_GRID_SPACING, _TRACK_GRID_SPACING)
    latitude_subset = latitude_full[y1:y2]
    longitude_full = np.arange(0, 360, _TRACK_GRID_SPACING)
    lon_mesh, lat_mesh = np.meshgrid(longitude_full, latitude_subset)
    polar_x, polar_y = geometry.lonlat_to_polar(
        longitude=lon_mesh, latitude=lat_mesh, hemisphere=hemisphere
    )
    polar_points = np.vstack((polar_x.ravel(), polar_y.ravel())).T
    polar_mesh_x, polar_mesh_y = geometry.build_polar_mesh(grid_spacing=_POLAR_GRID_SPACING)
    mesh_points = np.vstack((polar_mesh_x.ravel(), polar_mesh_y.ravel())).T
    return latitude_subset, longitude_full, polar_x, polar_points, polar_mesh_x, polar_mesh_y, mesh_points, polar_y


def _process_single_timestep(
    *,
    vorticity: npt.NDArray[np.float64],
    vorticity_threshold: float,
    track_data_cyclone: dict[str, npt.NDArray[np.float64]],
    track_data_anticyclone: dict[str, npt.NDArray[np.float64]],
    section_cyclone: npt.NDArray[np.intp],
    section_anticyclone: npt.NDArray[np.intp],
    polar_points: npt.NDArray[np.float64],
    mesh_points: npt.NDArray[np.float64],
    polar_mesh_x: npt.NDArray[np.float64],
    polar_mesh_y: npt.NDArray[np.float64],
    hemisphere: str,
    polar_x_shape: tuple[int, ...],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    interpolated_vorticity = scipy.interpolate.griddata(
        polar_points, vorticity.ravel(), mesh_points
    ).reshape(np.shape(polar_mesh_x))

    results: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
    for category_key, _ in [("C", "cyclone"), ("A", "anticyclone")]:
        if category_key == "C":
            contour_value = vorticity_threshold
            over_mask = interpolated_vorticity.ravel() >= vorticity_threshold
            section = section_cyclone
            track_data = track_data_cyclone
        else:
            contour_value = -vorticity_threshold
            over_mask = interpolated_vorticity.ravel() <= -vorticity_threshold
            section = section_anticyclone
            track_data = track_data_anticyclone

        mask_field = np.zeros(np.shape(interpolated_vorticity))
        flag_field = np.zeros(np.shape(interpolated_vorticity))
        intensity_field = np.zeros(np.shape(interpolated_vorticity))
        intensity_change_field = np.zeros(np.shape(interpolated_vorticity))

        track_polar_x, track_polar_y = geometry.lonlat_to_polar(
            longitude=track_data["lonc"][section],
            latitude=track_data["latc"][section],
            hemisphere=hemisphere,
        )
        track_positions = np.vstack((track_polar_x, track_polar_y))
        track_ids = track_data["eventarray"][section]
        track_intensities = track_data["Revc"][section]
        track_growth_rates = track_data["grc"][section]

        contour_set = plt.contour(
            polar_mesh_x, polar_mesh_y, interpolated_vorticity, [contour_value], colors="None"
        )
        contour_lines = contour_set.allsegs[0]
        n_contours = len(contour_lines)

        relationship_matrix = np.zeros((len(track_ids), n_contours))
        for j in range(len(track_ids)):
            xy = track_positions[:, j]
            for i in range(n_contours):
                line = matplotlib.path.Path(contour_lines[i], closed=True)
                if line.contains_point(xy, radius=2 * _POLAR_GRID_SPACING):
                    relationship_matrix[j, i] = 1

        offset = int(90 / _POLAR_GRID_SPACING)
        for i in range(n_contours):
            related_tracks = np.where(relationship_matrix[:, i] == 1)[0]
            if len(related_tracks) == 0:
                continue

            line = matplotlib.path.Path(contour_lines[i], closed=True)
            inner = line.contains_points(mesh_points)
            points_inside = mesh_points[inner & over_mask]

            if len(related_tracks) == 1:
                x0 = float(track_polar_x[related_tracks[0]])
                y0 = float(track_polar_y[related_tracks[0]])
                distances = geometry.polar_mesh_to_km_distance(
                    x0=x0, y0=y0, grid_points=points_inside, hemisphere=hemisphere
                )
                within_threshold = np.where(distances <= constants.DISTANCE_THRESHOLD)[0]
                grid_indices = (points_inside[within_threshold, :] / _POLAR_GRID_SPACING).astype(int)
                flag_field[grid_indices[:, 1] + offset, grid_indices[:, 0] + offset] = 1
                mask_field[grid_indices[:, 1] + offset, grid_indices[:, 0] + offset] = track_ids[
                    related_tracks[0]
                ]
                intensity_field[grid_indices[:, 1] + offset, grid_indices[:, 0] + offset] = track_intensities[
                    related_tracks[0]
                ]
                intensity_change_field[
                    grid_indices[:, 1] + offset, grid_indices[:, 0] + offset
                ] = track_growth_rates[related_tracks[0]]
            else:
                all_distances = np.zeros((points_inside.shape[0], len(related_tracks)))
                for j in range(len(related_tracks)):
                    x0 = float(track_polar_x[related_tracks[j]])
                    y0 = float(track_polar_y[related_tracks[j]])
                    all_distances[:, j] = geometry.polar_mesh_to_km_distance(
                        x0=x0,
                        y0=y0,
                        grid_points=points_inside,
                        hemisphere=hemisphere,
                    )
                for j in range(len(related_tracks)):
                    within_threshold = np.where(
                        (np.min(all_distances, axis=1) <= constants.DISTANCE_THRESHOLD)
                        & (np.argmin(all_distances, axis=1) == j)
                    )[0]
                    grid_indices = (points_inside[within_threshold, :] / _POLAR_GRID_SPACING).astype(int)
                    flag_field[grid_indices[:, 1] + offset, grid_indices[:, 0] + offset] = 1
                    mask_field[
                        grid_indices[:, 1] + offset, grid_indices[:, 0] + offset
                    ] = track_ids[related_tracks[j]]
                    intensity_field[
                        grid_indices[:, 1] + offset, grid_indices[:, 0] + offset
                    ] = track_intensities[related_tracks[j]]
                    intensity_change_field[
                        grid_indices[:, 1] + offset, grid_indices[:, 0] + offset
                    ] = track_growth_rates[related_tracks[j]]

        plt.close("all")
        results[category_key] = {
            "mask": mask_field,
            "flag": flag_field,
            "intensity": intensity_field,
            "intensity_change": intensity_change_field,
        }

    field_names = ["mask", "flag", "intensity", "intensity_change"]
    interpolated_fields: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    for category in ("C", "A"):
        for field_name in field_names:
            key = (category, field_name)
            interpolated_fields[key] = scipy.interpolate.griddata(
                mesh_points,
                results[category][field_name].ravel(),
                polar_points,
                method="nearest",
            ).reshape(polar_x_shape)

    return (
        interpolated_fields[("C", "mask")],
        interpolated_fields[("A", "mask")],
        interpolated_fields[("C", "flag")],
        interpolated_fields[("A", "flag")],
        interpolated_fields[("C", "intensity")],
        interpolated_fields[("A", "intensity")],
        interpolated_fields[("C", "intensity_change")],
        interpolated_fields[("A", "intensity_change")],
    )


def create_cyclone_masks(
    *,
    hemisphere: str,
    year_start: int,
    year_end: int,
    vorticity_directory: pathlib.Path,
    track_directory: pathlib.Path,
    output_directory: pathlib.Path,
    vorticity_threshold: float = constants.VORTICITY_THRESHOLD,
    track_filename_pattern: str = "{track_type}_VO_anom_T42_ERA5_{hemisphere}_long.npz",
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    is_northern = hemisphere == "NH"
    vorticity_sign = 1 if is_northern else -1

    latitude_full = np.arange(-90, 90 + _TRACK_GRID_SPACING, _TRACK_GRID_SPACING)
    equator_index = np.argmin(np.abs(latitude_full))
    y1 = equator_index if is_northern else 0
    y2 = len(latitude_full) if is_northern else equator_index + 1

    cyclone_data = _load_track_npz(
        track_directory=track_directory,
        track_type="TRACK",
        hemisphere=hemisphere,
        track_filename_pattern=track_filename_pattern,
    )
    anticyclone_data = _load_track_npz(
        track_directory=track_directory,
        track_type="ANTIC",
        hemisphere=hemisphere,
        track_filename_pattern=track_filename_pattern,
    )

    (
        latitude_subset,
        longitude_grid,
        polar_x,
        polar_points,
        polar_mesh_x,
        polar_mesh_y,
        mesh_points,
        _polar_y,
    ) = _build_grids(hemisphere=hemisphere, y1=y1, y2=y2)

    day_in_year_cyclone = cyclone_data["timec"] % constants.TIMESTEPS_PER_YEAR
    day_in_year_anticyclone = anticyclone_data["timec"] % constants.TIMESTEPS_PER_YEAR

    years = np.arange(year_start, year_end)
    for year in years:
        _LOG.info("Creating masks: hemisphere=%s year=%s", hemisphere, year)
        with netCDF4.Dataset(
            str(vorticity_directory / ("VO.anom.T42.%d.v146.nc" % year))
        ) as vorticity_dataset:
            vorticity_data = vorticity_dataset["VO"][:, y1:y2, :].data * vorticity_sign

        n_timesteps = vorticity_data.shape[0]
        all_mask_cyclone = np.zeros(vorticity_data.shape)
        all_mask_anticyclone = np.zeros(vorticity_data.shape)
        all_flag_cyclone = np.zeros(vorticity_data.shape)
        all_flag_anticyclone = np.zeros(vorticity_data.shape)
        all_intensity_cyclone = np.zeros(vorticity_data.shape)
        all_intensity_anticyclone = np.zeros(vorticity_data.shape)
        all_intensity_change_cyclone = np.zeros(vorticity_data.shape)
        all_intensity_change_anticyclone = np.zeros(vorticity_data.shape)

        year_cyclone = cyclone_data["longyear"]
        year_anticyclone = anticyclone_data["longyear"]

        for timestep in range(n_timesteps):
            section_cyclone = np.where(
                (year_cyclone == year) & (day_in_year_cyclone == timestep)
            )[0]
            section_anticyclone = np.where(
                (year_anticyclone == year) & (day_in_year_anticyclone == timestep)
            )[0]

            (
                mask_cyclone,
                mask_anticyclone,
                flag_cyclone,
                flag_anticyclone,
                intensity_cyclone,
                intensity_anticyclone,
                intensity_change_cyclone,
                intensity_change_anticyclone,
            ) = _process_single_timestep(
                vorticity=vorticity_data[timestep],
                vorticity_threshold=vorticity_threshold,
                track_data_cyclone=cyclone_data,
                track_data_anticyclone=anticyclone_data,
                section_cyclone=section_cyclone,
                section_anticyclone=section_anticyclone,
                polar_points=polar_points,
                mesh_points=mesh_points,
                polar_mesh_x=polar_mesh_x,
                polar_mesh_y=polar_mesh_y,
                hemisphere=hemisphere,
                polar_x_shape=polar_x.shape,
            )

            all_mask_cyclone[timestep] = mask_cyclone
            all_mask_anticyclone[timestep] = mask_anticyclone
            all_flag_cyclone[timestep] = flag_cyclone
            all_flag_anticyclone[timestep] = flag_anticyclone
            all_intensity_cyclone[timestep] = intensity_cyclone
            all_intensity_anticyclone[timestep] = intensity_anticyclone
            all_intensity_change_cyclone[timestep] = intensity_change_cyclone
            all_intensity_change_anticyclone[timestep] = intensity_change_anticyclone

        _save_mask_file(
            output_path=output_directory / ("MASK_%s_%d.nc" % (hemisphere, year)),
            latitude=latitude_subset,
            longitude=longitude_grid,
            mask_cyclone=all_mask_cyclone,
            mask_anticyclone=all_mask_anticyclone,
            flag_cyclone=all_flag_cyclone,
            flag_anticyclone=all_flag_anticyclone,
            intensity_cyclone=all_intensity_cyclone,
            intensity_anticyclone=all_intensity_anticyclone,
            intensity_change_cyclone=all_intensity_change_cyclone,
            intensity_change_anticyclone=all_intensity_change_anticyclone,
        )
        _LOG.info("Saved mask file for year %s", year)


def _save_mask_file(
    *,
    output_path: pathlib.Path,
    latitude: npt.NDArray[np.float64],
    longitude: npt.NDArray[np.float64],
    mask_cyclone: npt.NDArray[np.float64],
    mask_anticyclone: npt.NDArray[np.float64],
    flag_cyclone: npt.NDArray[np.float64],
    flag_anticyclone: npt.NDArray[np.float64],
    intensity_cyclone: npt.NDArray[np.float64],
    intensity_anticyclone: npt.NDArray[np.float64],
    intensity_change_cyclone: npt.NDArray[np.float64],
    intensity_change_anticyclone: npt.NDArray[np.float64],
) -> None:
    with netCDF4.Dataset(str(output_path), "w", format="NETCDF3_CLASSIC") as wfile:
        wfile.createDimension("lon", len(longitude))
        wfile.createDimension("lat", len(latitude))
        wfile.createDimension("time", None)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))

        mask_c_var = wfile.createVariable("mask_C", "f4", ("time", "lat", "lon"))
        mask_a_var = wfile.createVariable("mask_A", "f4", ("time", "lat", "lon"))
        intensity_c_var = wfile.createVariable("intensity_C", "f4", ("time", "lat", "lon"))
        intensity_a_var = wfile.createVariable("intensity_A", "f4", ("time", "lat", "lon"))
        intensity_change_c_var = wfile.createVariable("intensity_change_C", "f4", ("time", "lat", "lon"))
        intensity_change_a_var = wfile.createVariable("intensity_change_A", "f4", ("time", "lat", "lon"))
        flag_c_var = wfile.createVariable("flag_C", "f4", ("time", "lat", "lon"))
        flag_a_var = wfile.createVariable("flag_A", "f4", ("time", "lat", "lon"))

        lat_var.units = "degrees_north"
        lat_var.axis = "Y"
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"
        lon_var.units = "degrees_east"
        lon_var.axis = "X"
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"
        time_var.units = "6 hours from 0101 00 UTC"
        time_var.axis = "T"
        time_var.long_name = "time"

        mask_c_var.units = "1"
        mask_c_var.long_name = "cyclone track identifier mask"
        mask_a_var.units = "1"
        mask_a_var.long_name = "anticyclone track identifier mask"
        flag_c_var.units = "1"
        flag_c_var.long_name = "cyclone binary presence flag"
        flag_a_var.units = "1"
        flag_a_var.long_name = "anticyclone binary presence flag"
        intensity_c_var.units = "CVU"
        intensity_c_var.long_name = "cyclone vorticity intensity"
        intensity_a_var.units = "CVU"
        intensity_a_var.long_name = "anticyclone vorticity intensity"
        intensity_change_c_var.units = "CVU (6h)-1"
        intensity_change_c_var.long_name = "cyclone intensity tendency"
        intensity_change_a_var.units = "CVU (6h)-1"
        intensity_change_a_var.long_name = "anticyclone intensity tendency"

        lat_var[:] = latitude
        lon_var[:] = longitude
        time_var[:] = np.arange(constants.TIMESTEPS_PER_YEAR)

        mask_c_var[:] = mask_cyclone
        mask_a_var[:] = mask_anticyclone
        intensity_c_var[:] = intensity_cyclone
        intensity_a_var[:] = intensity_anticyclone
        intensity_change_c_var[:] = intensity_change_cyclone
        intensity_change_a_var[:] = intensity_change_anticyclone
        flag_c_var[:] = flag_cyclone
        flag_a_var[:] = flag_anticyclone
