import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import cyclone_energetics.constants as constants
import cyclone_energetics.integration as integration

_LOG = logging.getLogger(__name__)

_FLUX_NAMES: list[str] = [
    "tot_energy",
    "F_Swabs",
    "F_Olr",
    "F_Dhdt",
    "F_Shf",
    "F_TE",
]

_N_LAT_ERA5: int = 719
_N_LON_ERA5: int = 1440
_N_LAT_TRACK: int = 121
_N_LON_TRACK: int = 240


def _interpolate_masks_to_era5(
    *,
    mask_data: npt.NDArray,
    latitude_track: npt.NDArray,
    longitude_track: npt.NDArray,
    n_time: int,
) -> npt.NDArray:
    x = np.linspace(0, _N_LON_TRACK, _N_LON_TRACK)
    y = np.linspace(0, _N_LAT_TRACK, _N_LAT_TRACK)
    xi = np.linspace(0, _N_LON_TRACK, _N_LAT_ERA5 + 2)
    yi = np.linspace(0, _N_LAT_TRACK, _N_LAT_ERA5 + 2)

    mask_interp = np.zeros((n_time, _N_LAT_ERA5 + 2, _N_LAT_ERA5 + 2), float)
    for t in range(n_time):
        spline = scipy.interpolate.RectBivariateSpline(
            y, x, mask_data[t, :, :], kx=1, ky=1,
        )
        mask_interp[t, :, :] = spline(yi, xi)
    return mask_interp


def _apply_mask(
    *,
    flux_data: npt.NDArray,
    mask: npt.NDArray,
    threshold: float = 0.5,
) -> npt.NDArray:
    masked = np.copy(flux_data)
    idx_below = mask[:, 1:-1, :] < threshold
    masked[idx_below] = 0.0
    return masked


def assign_fluxes_with_intensity(
    *,
    year_start: int,
    year_end: int,
    integrated_flux_directory: pathlib.Path,
    mask_directory_nh: pathlib.Path,
    mask_directory_sh: pathlib.Path,
    output_directory: pathlib.Path,
    area_cut: str = "0.225",
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    n_cuts = len(constants.INTENSITY_CUTS)

    flux_shape = (n_cuts, 12, _N_LAT_ERA5, _N_LON_ERA5)
    flux_arrays = {
        name: {
            suffix: np.zeros(flux_shape, float)
            for suffix in ["total", "cycl", "ant"]
        }
        for name in _FLUX_NAMES
    }

    number_steps = 0
    for year in range(year_start, year_end):
        for month_idx, month in enumerate(constants.MONTH_STRINGS):
            _LOG.info(
                "Assigning fluxes: year=%s month=%s area_cut=%s",
                year,
                month,
                area_cut,
            )
            max_day = int(constants.TIMESTEPS_PER_MONTH[month_idx])
            month_start = constants.MONTH_BOUNDARIES[month_idx]
            month_end = constants.MONTH_BOUNDARIES[month_idx + 1]

            fluxes = _load_monthly_fluxes(
                integrated_flux_directory=integrated_flux_directory,
                year=year,
                month=month,
                max_day=max_day,
            )

            mask_sh = _load_mask_file(
                mask_path=mask_directory_sh / ("MASK_SH_%d.nc" % year),
                month_start=month_start,
                month_end=month_end,
            )
            mask_nh = _load_mask_file(
                mask_path=mask_directory_nh / ("MASK_NH_%d.nc" % year),
                month_start=month_start,
                month_end=month_end,
            )

            for cut_idx in range(n_cuts):
                cut_value = constants.INTENSITY_CUTS[cut_idx]

                cyc_combined, ant_combined = _combine_hemisphere_masks(
                    mask_sh=mask_sh,
                    mask_nh=mask_nh,
                    intensity_cut=cut_value,
                    n_time=max_day,
                )

                for name in _FLUX_NAMES:
                    flux_total = fluxes[name]
                    flux_cycl = _apply_mask(
                        flux_data=flux_total, mask=cyc_combined
                    )
                    flux_ant = _apply_mask(
                        flux_data=flux_total, mask=ant_combined
                    )

                    flux_arrays[name]["total"][cut_idx, month_idx, :, :] += np.mean(
                        flux_total, axis=0
                    )
                    flux_arrays[name]["cycl"][cut_idx, month_idx, :, :] += np.mean(
                        flux_cycl, axis=0
                    )
                    flux_arrays[name]["ant"][cut_idx, month_idx, :, :] += np.mean(
                        flux_ant, axis=0
                    )

            _LOG.info("Monthly mean calculated for year=%s month=%s", year, month)
        number_steps += 1

    for name in _FLUX_NAMES:
        for suffix in ["total", "cycl", "ant"]:
            flux_arrays[name][suffix] /= number_steps

    _save_assigned_fluxes(
        output_path=output_directory
        / ("WITH_INT_Cyclones_Sampled_Poleward_Fluxes_%s.nc" % area_cut),
        flux_arrays=flux_arrays,
        composite_lat=fluxes["_lat"],
        composite_lon=fluxes["_lon"],
    )


def _load_monthly_fluxes(
    *,
    integrated_flux_directory: pathlib.Path,
    year: int,
    month: str,
    max_day: int,
) -> dict[str, npt.NDArray]:
    result: dict[str, npt.NDArray] = {}
    with netCDF4.Dataset(
        str(
            integrated_flux_directory
            / ("Integrated_Fluxes_%d_%s_.nc" % (year, month))
        )
    ) as ds:
        result["F_TE"] = ds["F_TE_final"][:max_day, :, :]
        result["tot_energy"] = ds["tot_energy_final"][:max_day, :, :]
        result["F_Shf"] = ds["F_Shf_final"][:max_day, :, :]
        result["F_Swabs"] = ds["F_Swabs_final"][:max_day, :, :]
        result["F_Olr"] = ds["F_Olr_final"][:max_day, :, :]
        result["_lat"] = ds["lat"][:]
        result["_lon"] = ds["lon"][:]

    with netCDF4.Dataset(
        str(
            integrated_flux_directory
            / ("New_Integrated_Fluxes_%d_%s_.nc" % (year, month))
        )
    ) as ds:
        result["F_Dhdt"] = ds["F_Dhdt_final"][:max_day, :, :]

    return result


def _load_mask_file(
    *,
    mask_path: pathlib.Path,
    month_start: int,
    month_end: int,
) -> dict[str, npt.NDArray]:
    with netCDF4.Dataset(str(mask_path)) as ds:
        return {
            "lat": ds["lat"][:],
            "lon": ds["lon"][:],
            "flag_A": ds["flag_A"][month_start:month_end, :, :],
            "flag_C": ds["flag_C"][month_start:month_end, :, :],
            "intensity_A": ds["intensity_A"][month_start:month_end, :, :],
            "intensity_C": ds["intensity_C"][month_start:month_end, :, :],
        }


def _combine_hemisphere_masks(
    *,
    mask_sh: dict[str, npt.NDArray],
    mask_nh: dict[str, npt.NDArray],
    intensity_cut: int,
    n_time: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    cyc_sh = np.copy(mask_sh["flag_C"])
    ant_sh = np.copy(mask_sh["flag_A"])
    cyc_nh = np.copy(mask_nh["flag_C"])
    ant_nh = np.copy(mask_nh["flag_A"])

    cyc_sh[mask_sh["intensity_C"] < intensity_cut] = 0
    ant_sh[mask_sh["intensity_A"] < intensity_cut] = 0
    cyc_nh[mask_nh["intensity_C"] < intensity_cut] = 0
    ant_nh[mask_nh["intensity_A"] < intensity_cut] = 0

    ant_combined = np.concatenate(
        (ant_sh[:, :-1, :], ant_nh), axis=1
    )[:, ::-1, :]
    cyc_combined = np.concatenate(
        (cyc_sh[:, :-1, :], cyc_nh), axis=1
    )[:, ::-1, :]

    latitude_t = np.concatenate(
        (mask_sh["lat"][:-1], mask_nh["lat"]), axis=0
    )[::-1]
    longitude_t = mask_nh["lon"]

    x = np.linspace(0, 240, 240)
    y = np.linspace(0, 121, 121)
    xi = np.linspace(0, 240, _N_LON_ERA5)
    yi = np.linspace(0, 121, _N_LAT_ERA5 + 2)

    ant_interp = np.zeros((n_time, _N_LAT_ERA5 + 2, _N_LON_ERA5), float)
    cyc_interp = np.zeros((n_time, _N_LAT_ERA5 + 2, _N_LON_ERA5), float)

    for t in range(n_time):
        spline_ant = scipy.interpolate.RectBivariateSpline(
            y, x, ant_combined[t, :, :], kx=1, ky=1,
        )
        ant_interp[t, :, :] = spline_ant(yi, xi)
        spline_cyc = scipy.interpolate.RectBivariateSpline(
            y, x, cyc_combined[t, :, :], kx=1, ky=1,
        )
        cyc_interp[t, :, :] = spline_cyc(yi, xi)

    return cyc_interp, ant_interp


def _save_assigned_fluxes(
    *,
    output_path: pathlib.Path,
    flux_arrays: dict[str, dict[str, npt.NDArray]],
    composite_lat: npt.NDArray,
    composite_lon: npt.NDArray,
) -> None:
    n_cuts = len(constants.INTENSITY_CUTS)
    with netCDF4.Dataset(
        str(output_path), "w", format="NETCDF4_CLASSIC"
    ) as wfile:
        wfile.createDimension("lon", len(composite_lon))
        wfile.createDimension("lat", len(composite_lat))
        wfile.createDimension("time", 12)
        wfile.createDimension("intensity", n_cuts)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))
        int_var = wfile.createVariable("intensity", "f4", ("intensity",))

        lat_var.units = "Degrees North"
        lat_var.axis = "Y"
        lon_var.units = "Degrees East"
        lon_var.axis = "X"
        time_var.units = "Months"
        time_var.axis = "T"
        int_var.units = "CVU"

        lat_var[:] = composite_lat
        lon_var[:] = composite_lon
        time_var[:] = np.arange(12)
        int_var[:] = constants.INTENSITY_CUTS

        for name in _FLUX_NAMES:
            for suffix, nc_suffix in [
                ("total", ""),
                ("cycl", "_cycl"),
                ("ant", "_ant"),
            ]:
                var_name = "%s_final%s" % (name, nc_suffix)
                var = wfile.createVariable(
                    var_name, "f4", ("intensity", "time", "lat", "lon")
                )
                var[:, :, :, :] = flux_arrays[name][suffix]

    _LOG.info("Saved assigned fluxes: %s", output_path)
