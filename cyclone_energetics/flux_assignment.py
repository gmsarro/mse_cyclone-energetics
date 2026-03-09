from __future__ import annotations

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_FLUX_NAMES: list[str] = [
    "tot_energy",
    "F_Swabs",
    "F_Olr",
    "F_Dhdt",
    "F_Shf",
    "F_TE",
    "F_u_mse",
    "F_v_mse",
]


def _interpolate_masks_to_era5(
    *,
    mask_data: npt.NDArray[np.floating],
    n_lat_target: int,
    n_lon_target: int,
    n_time: int,
) -> npt.NDArray[np.floating]:
    n_lat_src, n_lon_src = mask_data.shape[-2], mask_data.shape[-1]
    x = np.linspace(0, n_lon_src, n_lon_src)
    y = np.linspace(0, n_lat_src, n_lat_src)
    xi = np.linspace(0, n_lon_src, n_lon_target)
    yi = np.linspace(0, n_lat_src, n_lat_target)

    mask_interp = np.zeros((n_time, n_lat_target, n_lon_target), float)
    for t in range(n_time):
        spline = scipy.interpolate.RectBivariateSpline(
            y, x, mask_data[t], kx=1, ky=1,
        )
        mask_interp[t] = spline(yi, xi)
    return mask_interp


def _apply_mask(
    *,
    flux_data: npt.NDArray[np.floating],
    mask: npt.NDArray[np.floating],
    threshold: float = 0.5,
) -> npt.NDArray[np.floating]:
    masked = np.copy(flux_data)
    idx_below = mask[:, 1:-1] < threshold
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

    first_year = year_start
    first_month = constants.MONTH_STRINGS[0]
    with netCDF4.Dataset(
        str(integrated_flux_directory / ("Integrated_Fluxes_%d_%s_.nc" % (first_year, first_month)))
    ) as ds:
        n_lat_era5 = len(ds["lat"][:])
        n_lon_era5 = len(ds["lon"][:])

    flux_shape = (n_cuts, 12, n_lat_era5, n_lon_era5)
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
                    n_lat_target=n_lat_era5,
                    n_lon_target=n_lon_era5,
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
) -> dict[str, npt.NDArray[np.floating]]:
    result: dict[str, npt.NDArray[np.floating]] = {}
    with netCDF4.Dataset(
        str(
            integrated_flux_directory
            / ("Integrated_Fluxes_%d_%s_.nc" % (year, month))
        )
    ) as ds:
        result["F_TE"] = np.array(ds["F_TE_final"][:max_day])
        result["tot_energy"] = np.array(ds["tot_energy_final"][:max_day])
        result["F_Shf"] = np.array(ds["F_Shf_final"][:max_day])
        result["F_Swabs"] = np.array(ds["F_Swabs_final"][:max_day])
        result["F_Olr"] = np.array(ds["F_Olr_final"][:max_day])
        result["_lat"] = np.array(ds["lat"][:])
        result["_lon"] = np.array(ds["lon"][:])

    with netCDF4.Dataset(
        str(
            integrated_flux_directory
            / ("New_Integrated_Fluxes_%d_%s_.nc" % (year, month))
        )
    ) as ds:
        result["F_Dhdt"] = np.array(ds["F_Dhdt_final"][:max_day])
        if "F_u_mse_final" in ds.variables:
            result["F_u_mse"] = np.array(ds["F_u_mse_final"][:max_day])
        else:
            result["F_u_mse"] = np.zeros_like(result["F_Dhdt"])
        if "F_v_mse_final" in ds.variables:
            result["F_v_mse"] = np.array(ds["F_v_mse_final"][:max_day])
        else:
            result["F_v_mse"] = np.zeros_like(result["F_Dhdt"])

    return result


def _load_mask_file(
    *,
    mask_path: pathlib.Path,
    month_start: int,
    month_end: int,
) -> dict[str, npt.NDArray[np.floating]]:
    with netCDF4.Dataset(str(mask_path)) as ds:
        return {
            "lat": np.array(ds["lat"][:]),
            "lon": np.array(ds["lon"][:]),
            "flag_A": np.array(ds["flag_A"][month_start:month_end]),
            "flag_C": np.array(ds["flag_C"][month_start:month_end]),
            "intensity_A": np.array(ds["intensity_A"][month_start:month_end]),
            "intensity_C": np.array(ds["intensity_C"][month_start:month_end]),
        }


def _combine_hemisphere_masks(
    *,
    mask_sh: dict[str, npt.NDArray[np.floating]],
    mask_nh: dict[str, npt.NDArray[np.floating]],
    intensity_cut: int,
    n_time: int,
    n_lat_target: int,
    n_lon_target: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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

    n_lat_src = ant_combined.shape[-2]
    n_lon_src = ant_combined.shape[-1]

    x = np.linspace(0, n_lon_src, n_lon_src)
    y = np.linspace(0, n_lat_src, n_lat_src)
    xi = np.linspace(0, n_lon_src, n_lon_target)
    yi = np.linspace(0, n_lat_src, n_lat_target + 2)

    ant_interp = np.zeros((n_time, n_lat_target + 2, n_lon_target), float)
    cyc_interp = np.zeros((n_time, n_lat_target + 2, n_lon_target), float)

    for t in range(n_time):
        spline_ant = scipy.interpolate.RectBivariateSpline(
            y, x, ant_combined[t], kx=1, ky=1,
        )
        ant_interp[t] = spline_ant(yi, xi)
        spline_cyc = scipy.interpolate.RectBivariateSpline(
            y, x, cyc_combined[t], kx=1, ky=1,
        )
        cyc_interp[t] = spline_cyc(yi, xi)

    return cyc_interp, ant_interp


def _save_assigned_fluxes(
    *,
    output_path: pathlib.Path,
    flux_arrays: dict[str, dict[str, npt.NDArray[np.floating]]],
    composite_lat: npt.NDArray[np.floating],
    composite_lon: npt.NDArray[np.floating],
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
                var[:] = flux_arrays[name][suffix]

    _LOG.info("Saved assigned fluxes: %s", output_path)
