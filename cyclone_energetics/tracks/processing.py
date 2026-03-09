from __future__ import annotations

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)


def _filter_short_tracks(
    *,
    num_points: npt.NDArray[np.integer],
    first_point: npt.NDArray[np.integer],
    track_id: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.floating],
]:
    valid = np.where(num_points > 0)[0]
    return num_points[valid], first_point[valid], track_id[valid]


def _compute_event_array(
    *,
    track_id: npt.NDArray[np.floating],
    first_point: npt.NDArray[np.integer],
    num_points: npt.NDArray[np.integer],
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    total_points: int,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    n_events = len(track_id)
    event_array = np.zeros(total_points, dtype="float32")
    start_longitude = np.zeros(n_events, dtype="float32")
    start_latitude = np.zeros(n_events, dtype="float32")
    for i in range(n_events):
        fp = first_point[i]
        npt_i = num_points[i]
        event_array[fp : fp + npt_i] = track_id[i]
        start_longitude[i] = longitude[fp]
        start_latitude[i] = latitude[fp]
    return event_array, start_longitude, start_latitude


def _compute_time_indices(
    *,
    time_array: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
]:
    total_points = len(time_array)
    long_year = (time_array // constants.TIMESTEPS_PER_YEAR) + constants.ERA5_BASE_YEAR
    long_day_in_year = time_array % constants.TIMESTEPS_PER_YEAR
    long_month = np.zeros(total_points, dtype="int32")
    for month_index in range(12):
        month_mask = (long_day_in_year >= constants.MONTH_BOUNDARIES[month_index]) & (
            long_day_in_year < constants.MONTH_BOUNDARIES[month_index + 1]
        )
        long_month[month_mask] = month_index + 1
    return long_year.astype("int32"), long_month, long_day_in_year.astype("int32")


def _compute_travel_distances(
    *,
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    event_array: npt.NDArray[np.floating],
    track_id: npt.NDArray[np.floating],
    intensity: npt.NDArray[np.floating],
    genesis_locations: npt.NDArray[np.integer],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    total_points = len(longitude)
    n_events = len(track_id)
    earth_radius = constants.EARTH_RADIUS
    deg_to_rad = np.pi / 180.0
    longitude_rad = longitude * deg_to_rad
    latitude_rad = latitude * deg_to_rad

    stepwise_distance = np.zeros(total_points, dtype="float32")
    stepwise_distance[1:] = (
        2.0
        * earth_radius
        * np.arcsin(
            np.sqrt(
                np.sin((latitude_rad[1:] - latitude_rad[:-1]) / 2.0) ** 2
                + np.cos(latitude_rad[1:])
                * np.cos(latitude_rad[:-1])
                * np.sin((longitude_rad[1:] - longitude_rad[:-1]) / 2.0) ** 2
            )
        )
    )
    stepwise_distance[genesis_locations] = 0.0

    travel_distance = np.zeros(n_events, dtype="float32")
    max_intensity_location = np.zeros(n_events)
    for i in range(n_events):
        section = np.where(event_array == track_id[i])[0]
        max_intensity_location[i] = section[0] + np.where(
            intensity[section] == np.max(intensity[section])
        )[0][0]
        travel_distance[i] = np.sum(stepwise_distance[section])
    return stepwise_distance, travel_distance, max_intensity_location


def _categorize_tracks(
    *,
    travel_distance: npt.NDArray[np.floating],
    num_points: npt.NDArray[np.integer],
) -> npt.NDArray[np.integer]:
    n_events = len(travel_distance)
    category = np.zeros(n_events, dtype=int)
    category[(travel_distance >= 1e6) & (num_points >= 8)] = 0
    category[(travel_distance >= 1e6) & (num_points < 8)] = 1
    category[(travel_distance < 1e6) & (num_points >= 8)] = 2
    category[(travel_distance < 1e6) & (num_points < 8)] = 3
    return category


def process_track_data(
    *,
    track_directory: pathlib.Path,
    output_directory: pathlib.Path,
    hemispheres: list[str] | None = None,
    track_types: list[str] | None = None,
    track_filename_pattern: str = "{track_type}_VO_anom_T42_ERA5_{hemisphere}.nc",
) -> None:
    if hemispheres is None:
        hemispheres = ["SH", "NH"]
    if track_types is None:
        track_types = ["TRACK", "ANTIC"]

    output_directory.mkdir(parents=True, exist_ok=True)

    for hemisphere in hemispheres:
        for track_type in track_types:
            _LOG.info("Processing tracks: %s %s", track_type, hemisphere)
            filename = track_filename_pattern.format(
                track_type=track_type,
                hemisphere=hemisphere,
            )
            input_path = track_directory / filename
            output_name = filename.replace(".nc", "_long.npz")
            output_path = output_directory / output_name

            with netCDF4.Dataset(str(input_path)) as dataset:
                time_values = dataset["time"][:].data - 1
                latitude_values = dataset["latitude"][:].data
                longitude_values = dataset["longitude"][:].data
                intensity_values = dataset["intensity"][:].data
                raw_track_id = dataset["TRACK_ID"][:].data + 1.0
                raw_first_point = dataset["FIRST_PT"][:].data
                raw_num_points = dataset["NUM_PTS"][:].data

            num_points, first_point, track_id = _filter_short_tracks(
                num_points=raw_num_points,
                first_point=raw_first_point,
                track_id=raw_track_id,
            )

            total_points = len(time_values)
            event_array, start_longitude, start_latitude = _compute_event_array(
                track_id=track_id,
                first_point=first_point,
                num_points=num_points,
                longitude=longitude_values,
                latitude=latitude_values,
                total_points=total_points,
            )

            long_year, long_month, long_day_in_year = _compute_time_indices(
                time_array=time_values,
            )
            long_day = long_day_in_year - constants.MONTH_BOUNDARIES[
                (long_month - 1).astype("int32")
            ]

            genesis_locations = np.unique(event_array, return_index=True)[1].astype("int")

            growth_rate = np.zeros(total_points, dtype="float32")
            growth_rate[1:-1] = (intensity_values[2:] - intensity_values[:-2]) * 2
            growth_rate[genesis_locations] = -999.0
            growth_rate[genesis_locations[1:] - 1] = -999.0

            stepwise_distance, travel_distance, max_intensity_location = _compute_travel_distances(
                longitude=longitude_values,
                latitude=latitude_values,
                event_array=event_array,
                track_id=track_id,
                intensity=intensity_values,
                genesis_locations=genesis_locations,
            )

            category = _categorize_tracks(
                travel_distance=travel_distance,
                num_points=num_points,
            )

            valid_indices = np.where(event_array != 0)[0]
            event_array = event_array[valid_indices]
            long_year = long_year[valid_indices]
            long_month = long_month[valid_indices]
            long_day = long_day[valid_indices]
            growth_rate = growth_rate[valid_indices]
            time_values = time_values[valid_indices]
            intensity_values = intensity_values[valid_indices]
            latitude_values = latitude_values[valid_indices]
            longitude_values = longitude_values[valid_indices]
            stepwise_distance = stepwise_distance[valid_indices]

            long_category = np.zeros(len(event_array))
            for i in range(len(track_id)):
                long_category[np.where(event_array == track_id[i])[0]] = category[i]

            np.savez(
                str(output_path),
                fp=first_point,
                ti=track_id,
                eventarray=event_array,
                longyear=long_year,
                longmon=long_month,
                longday=long_day,
                grc=growth_rate,
                timec=time_values,
                Revc=intensity_values,
                latc=latitude_values,
                lonc=longitude_values,
                stlat=start_latitude,
                stlon=start_longitude,
                npt=num_points,
                travdist=travel_distance,
                alldist=stepwise_distance,
                genloc=genesis_locations,
                maxloc=max_intensity_location,
                tcategory=category,
                longcat=long_category,
            )
            _LOG.info("Saved processed tracks: %s", output_path)
