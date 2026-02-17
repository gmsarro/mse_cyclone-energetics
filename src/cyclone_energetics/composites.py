import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_COMPOSITE_BOX_DEGREES: float = 15.0
_COMPOSITE_GRID_SIZE: int = 120
_AREA_GRID_SIZE: int = 20
_INTENSITY_THRESHOLD_STRONG: int = 6


def build_cyclone_composites(
    *,
    year_start: int,
    year_end: int,
    hemispheres: list[str],
    track_types: list[str],
    track_directory: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    mask_directory_sh: pathlib.Path,
    mask_directory_nh: pathlib.Path,
    output_directory: pathlib.Path,
    intensity_threshold: int = _INTENSITY_THRESHOLD_STRONG,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(
        str(
            integrated_flux_directory / "Integrated_Fluxes_%d_01_.nc" % year_start
        )
    ) as ds:
        flux_lat = ds["lat"][:]
        flux_lon = ds["lon"][:]

    with netCDF4.Dataset(
        str(mask_directory_nh / ("MASK_NH_%d.nc" % year_start))
    ) as ds:
        mask_lat_nh = ds["lat"][:]
        mask_lon_nh = ds["lon"][:]

    with netCDF4.Dataset(
        str(mask_directory_sh / ("MASK_SH_%d.nc" % year_start))
    ) as ds:
        mask_lat_sh = ds["lat"][:]
        mask_lon_sh = ds["lon"][:]

    latitude_t = np.concatenate(
        (mask_lat_sh[:-1], mask_lat_nh), axis=0
    )[::-1]
    longitude_t = mask_lon_nh

    for hemisphere in hemispheres:
        for rotation in track_types:
            _LOG.info(
                "Building composites: hemisphere=%s rotation=%s",
                hemisphere,
                rotation,
            )
            _build_single_composite(
                hemisphere=hemisphere,
                rotation=rotation,
                year_start=year_start,
                year_end=year_end,
                track_directory=track_directory,
                integrated_flux_directory=integrated_flux_directory,
                mask_directory_sh=mask_directory_sh,
                mask_directory_nh=mask_directory_nh,
                output_directory=output_directory,
                flux_lat=flux_lat,
                flux_lon=flux_lon,
                latitude_t=latitude_t,
                longitude_t=longitude_t,
                mask_lat_nh=mask_lat_nh,
                intensity_threshold=intensity_threshold,
            )


def _build_single_composite(
    *,
    hemisphere: str,
    rotation: str,
    year_start: int,
    year_end: int,
    track_directory: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    mask_directory_sh: pathlib.Path,
    mask_directory_nh: pathlib.Path,
    output_directory: pathlib.Path,
    flux_lat: npt.NDArray,
    flux_lon: npt.NDArray,
    latitude_t: npt.NDArray,
    longitude_t: npt.NDArray,
    mask_lat_nh: npt.NDArray,
    intensity_threshold: int,
) -> None:
    with netCDF4.Dataset(
        str(
            track_directory
            / ("%s_VO_anom_T42_ERA5_1979_2018_all%s.nc" % (rotation, hemisphere))
        )
    ) as ds:
        track_time = ds["time"][:].data - 1
        track_lon = ds["longitude"][:].data
        track_lat = ds["latitude"][:].data
        track_int = ds["intensity"][:].data

    composite_te = np.zeros((12, _COMPOSITE_GRID_SIZE, _COMPOSITE_GRID_SIZE), float)
    composite_energy = np.zeros(
        (12, _COMPOSITE_GRID_SIZE, _COMPOSITE_GRID_SIZE), float
    )
    composite_shf = np.zeros(
        (12, _COMPOSITE_GRID_SIZE, _COMPOSITE_GRID_SIZE), float
    )
    composite_dhdt = np.zeros(
        (12, _COMPOSITE_GRID_SIZE, _COMPOSITE_GRID_SIZE), float
    )
    composite_swabs = np.zeros(
        (12, _COMPOSITE_GRID_SIZE, _COMPOSITE_GRID_SIZE), float
    )
    composite_olr = np.zeros(
        (12, _COMPOSITE_GRID_SIZE, _COMPOSITE_GRID_SIZE), float
    )
    number_of_cyclones = np.zeros(12, float)
    composite_lat_saved = None
    composite_lon_saved = None

    hemi_adj = 1 if hemisphere != "SH" else -1

    for time_step in range(len(track_time)):
        if track_int[time_step] <= intensity_threshold:
            continue
        if not (20 < track_lat[time_step] * hemi_adj < 75):
            continue

        year = int(track_time[time_step] / (365 * 4) + 1979)
        if year < year_start or year >= year_end:
            continue

        day_of_year = int(track_time[time_step] - (year - 1979) * (365 * 4))
        month_str, pythonic_month, day_of_month = _resolve_month(
            day_of_year=day_of_year
        )

        lat_storm = track_lat[time_step]
        lon_storm = track_lon[time_step]

        lat_max = (int(lat_storm * 4)) / 4.0 + _COMPOSITE_BOX_DEGREES
        lat_min = (int(lat_storm * 4)) / 4.0 - _COMPOSITE_BOX_DEGREES
        lon_max = (int(lon_storm * 4)) / 4.0 + _COMPOSITE_BOX_DEGREES
        lon_min = (int(lon_storm * 4)) / 4.0 - _COMPOSITE_BOX_DEGREES

        if lon_max >= 360 or lon_min < 0:
            continue

        a_idx = np.where(flux_lat == lat_max)
        b_idx = np.where(flux_lat == lat_min)
        c_idx = np.where(flux_lon == lon_min)
        d_idx = np.where(flux_lon == lon_max)

        if any(
            len(idx[0]) == 0 for idx in [a_idx, b_idx, c_idx, d_idx]
        ):
            continue

        a_i = int(a_idx[0][0])
        b_i = int(b_idx[0][0])
        c_i = int(c_idx[0][0])
        d_i = int(d_idx[0][0])

        with netCDF4.Dataset(
            str(
                integrated_flux_directory
                / ("Integrated_Fluxes_%d_%s_.nc" % (year, month_str))
            )
        ) as ds:
            composite_te[pythonic_month, :, :] += ds["F_TE_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]
            composite_energy[pythonic_month, :, :] += ds["tot_energy_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]
            composite_shf[pythonic_month, :, :] += ds["F_Shf_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]
            composite_swabs[pythonic_month, :, :] += ds["F_Swabs_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]
            composite_olr[pythonic_month, :, :] += ds["F_Olr_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]
            composite_lat_saved = ds["lat"][a_i:b_i]
            composite_lon_saved = ds["lon"][c_i:d_i]

        with netCDF4.Dataset(
            str(
                integrated_flux_directory
                / ("New_Integrated_Fluxes_%d_%s_.nc" % (year, month_str))
            )
        ) as ds:
            composite_dhdt[pythonic_month, :, :] += ds["F_Dhdt_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]

        number_of_cyclones[pythonic_month] += 1

    if composite_lat_saved is None or composite_lon_saved is None:
        _LOG.info(
            "No composites built for %s %s - no qualifying cyclones",
            hemisphere,
            rotation,
        )
        return

    _save_composite_file(
        output_path=output_directory
        / ("New_Intense_Composites_%s_%s_.nc" % (rotation, hemisphere)),
        composite_lat=composite_lat_saved,
        composite_lon=composite_lon_saved,
        composite_te=composite_te,
        composite_energy=composite_energy,
        composite_shf=composite_shf,
        composite_dhdt=composite_dhdt,
        composite_swabs=composite_swabs,
        composite_olr=composite_olr,
        number_of_cyclones=number_of_cyclones,
    )


def _resolve_month(
    *,
    day_of_year: int,
) -> tuple[str, int, int]:
    for ee in range(12):
        if (
            constants.MONTH_BOUNDARIES[ee]
            <= day_of_year
            < constants.MONTH_BOUNDARIES[ee + 1]
        ):
            return (
                constants.MONTH_STRINGS[ee],
                ee,
                day_of_year - constants.MONTH_BOUNDARIES[ee],
            )
    raise ValueError("Invalid day_of_year: %d" % day_of_year)


def _save_composite_file(
    *,
    output_path: pathlib.Path,
    composite_lat: npt.NDArray,
    composite_lon: npt.NDArray,
    composite_te: npt.NDArray,
    composite_energy: npt.NDArray,
    composite_shf: npt.NDArray,
    composite_dhdt: npt.NDArray,
    composite_swabs: npt.NDArray,
    composite_olr: npt.NDArray,
    number_of_cyclones: npt.NDArray,
) -> None:
    with netCDF4.Dataset(
        str(output_path), "w", format="NETCDF3_CLASSIC"
    ) as wfile:
        wfile.createDimension("lon", len(composite_lon))
        wfile.createDimension("lat", len(composite_lat))
        wfile.createDimension("month", 12)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))

        te_var = wfile.createVariable("composite_TE", "f4", ("month", "lat", "lon"))
        energy_var = wfile.createVariable(
            "composite_energy", "f4", ("month", "lat", "lon")
        )
        shf_var = wfile.createVariable(
            "composite_Shf", "f4", ("month", "lat", "lon")
        )
        dhdt_var = wfile.createVariable(
            "composite_Dhdt", "f4", ("month", "lat", "lon")
        )
        swabs_var = wfile.createVariable(
            "composite_Swabs", "f4", ("month", "lat", "lon")
        )
        olr_var = wfile.createVariable(
            "composite_Olr", "f4", ("month", "lat", "lon")
        )
        number_var = wfile.createVariable("number", "f4", ("month",))

        lat_var.units = "Degrees North"
        lat_var.axis = "Y"
        lon_var.units = "Degrees East"
        lon_var.axis = "X"

        lat_var[:] = composite_lat - np.mean(composite_lat)
        lon_var[:] = composite_lon - np.mean(composite_lon)
        number_var[:] = number_of_cyclones

        for month_idx in range(12):
            n = number_of_cyclones[month_idx]
            if n > 0:
                te_var[month_idx, :, :] = composite_te[month_idx, :, :] / n
                energy_var[month_idx, :, :] = composite_energy[month_idx, :, :] / n
                shf_var[month_idx, :, :] = composite_shf[month_idx, :, :] / n
                dhdt_var[month_idx, :, :] = composite_dhdt[month_idx, :, :] / n
                swabs_var[month_idx, :, :] = composite_swabs[month_idx, :, :] / n
                olr_var[month_idx, :, :] = composite_olr[month_idx, :, :] / n

    _LOG.info("Saved composites: %s", output_path)
