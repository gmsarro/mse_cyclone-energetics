from __future__ import annotations

"""Process cyclone and anticyclone tracks from the TRACK algorithm.

The TRACK algorithm identifies and follows vorticity features on the
filtered T42 vorticity field.  See:

    Hodges, K. I. (1994). A general method for tracking analysis and its
    application to meteorological data. Monthly Weather Review, 122,
    2573–2586.

    Hodges, K. I. (1995). Feature tracking on the unit sphere. Monthly
    Weather Review, 123, 3458–3465.

    Hoskins, B., & Hodges, K. I. (2019). The annual cycle of Northern
    Hemisphere storm tracks. Part I: Seasons. Journal of Climate, 32,
    1743–1760. doi:10.1175/JCLI-D-17-0870.

The same algorithm also produces the filtered vorticity fields used by
the area-assignment step (:mod:`cyclone_energetics.masking`).
"""

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
    start_lon = np.zeros(n_events, dtype="float32")
    start_lat = np.zeros(n_events, dtype="float32")
    for i in range(n_events):
        fp = first_point[i]
        npt_i = num_points[i]
        event_array[fp : fp + npt_i] = track_id[i]
        start_lon[i] = longitude[fp]
        start_lat[i] = latitude[fp]
    return event_array, start_lon, start_lat


def _compute_time_indices(
    *,
    time_array: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
]:
    total_points = len(time_array)
    long_year = (time_array // constants.TIMESTEPS_PER_YEAR) + 1979
    long_diy = time_array % constants.TIMESTEPS_PER_YEAR
    long_mon = np.zeros(total_points, dtype="int32")
    for m in range(12):
        month_mask = (long_diy >= constants.MONTH_BOUNDARIES[m]) & (
            long_diy < constants.MONTH_BOUNDARIES[m + 1]
        )
        long_mon[month_mask] = m + 1
    return long_year.astype("int32"), long_mon, long_diy.astype("int32")


def _compute_travel_distances(
    *,
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    event_array: npt.NDArray[np.floating],
    track_id: npt.NDArray[np.floating],
    intensity: npt.NDArray[np.floating],
    gen_loc: npt.NDArray[np.integer],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    total_points = len(longitude)
    n_events = len(track_id)
    re = 6.37e6
    conv = np.pi / 180.0
    lont = longitude * conv
    latt = latitude * conv

    all_dist = np.zeros(total_points, dtype="float32")
    all_dist[1:] = (
        2.0
        * re
        * np.arcsin(
            np.sqrt(
                np.sin((latt[1:] - latt[:-1]) / 2.0) ** 2
                + np.cos(latt[1:])
                * np.cos(latt[:-1])
                * np.sin((lont[1:] - lont[:-1]) / 2.0) ** 2
            )
        )
    )
    all_dist[gen_loc] = 0.0

    trav_dist = np.zeros(n_events, dtype="float32")
    max_loc = np.zeros(n_events)
    for i in range(n_events):
        sec = np.where(event_array == track_id[i])[0]
        max_loc[i] = sec[0] + np.where(intensity[sec] == np.max(intensity[sec]))[0][0]
        trav_dist[i] = np.sum(all_dist[sec])
    return all_dist, trav_dist, max_loc


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
) -> None:
    if hemispheres is None:
        hemispheres = ["SH", "NH"]
    if track_types is None:
        track_types = ["TRACK", "ANTIC"]

    output_directory.mkdir(parents=True, exist_ok=True)

    for hemisphere in hemispheres:
        for track_type in track_types:
            _LOG.info("Processing tracks: %s %s", track_type, hemisphere)
            filename = "%s_VO_anom_T42_ERA5_1979_2024_%s.nc" % (
                track_type,
                hemisphere,
            )
            input_path = track_directory / filename
            output_name = filename.replace(".nc", "_long.npz")
            output_path = output_directory / output_name

            with netCDF4.Dataset(str(input_path)) as ds:
                time_c = ds["time"][:].data - 1
                lat_c = ds["latitude"][:].data
                lon_c = ds["longitude"][:].data
                intensity_c = ds["intensity"][:].data
                raw_track_id = ds["TRACK_ID"][:].data + 1.0
                raw_first_pt = ds["FIRST_PT"][:].data
                raw_num_pts = ds["NUM_PTS"][:].data

            num_pts, first_pt, track_id = _filter_short_tracks(
                num_points=raw_num_pts,
                first_point=raw_first_pt,
                track_id=raw_track_id,
            )

            total_points = len(time_c)
            event_array, start_lon, start_lat = _compute_event_array(
                track_id=track_id,
                first_point=first_pt,
                num_points=num_pts,
                longitude=lon_c,
                latitude=lat_c,
                total_points=total_points,
            )

            long_year, long_mon, long_diy = _compute_time_indices(
                time_array=time_c,
            )
            long_day = long_diy - constants.MONTH_BOUNDARIES[
                (long_mon - 1).astype("int32")
            ]

            gen_loc = np.unique(event_array, return_index=True)[1].astype("int")

            growth_rate = np.zeros(total_points, dtype="float32")
            growth_rate[1:-1] = (intensity_c[2:] - intensity_c[:-2]) * 2
            growth_rate[gen_loc] = -999.0
            growth_rate[gen_loc[1:] - 1] = -999.0

            all_dist, trav_dist, max_loc = _compute_travel_distances(
                longitude=lon_c,
                latitude=lat_c,
                event_array=event_array,
                track_id=track_id,
                intensity=intensity_c,
                gen_loc=gen_loc,
            )

            category = _categorize_tracks(
                travel_distance=trav_dist,
                num_points=num_pts,
            )

            take = np.where(event_array != 0)[0]
            event_array = event_array[take]
            long_year = long_year[take]
            long_mon = long_mon[take]
            long_day = long_day[take]
            growth_rate = growth_rate[take]
            time_c = time_c[take]
            intensity_c = intensity_c[take]
            lat_c = lat_c[take]
            lon_c = lon_c[take]
            all_dist = all_dist[take]

            long_cat = np.zeros(len(event_array))
            for i in range(len(track_id)):
                long_cat[np.where(event_array == track_id[i])[0]] = category[i]

            np.savez(
                str(output_path),
                fp=first_pt,
                ti=track_id,
                eventarray=event_array,
                longyear=long_year,
                longmon=long_mon,
                longday=long_day,
                grc=growth_rate,
                timec=time_c,
                Revc=intensity_c,
                latc=lat_c,
                lonc=lon_c,
                stlat=start_lat,
                stlon=start_lon,
                npt=num_pts,
                travdist=trav_dist,
                alldist=all_dist,
                genloc=gen_loc,
                maxloc=max_loc,
                tcategory=category,
                longcat=long_cat,
            )
            _LOG.info("Saved processed tracks: %s", output_path)
