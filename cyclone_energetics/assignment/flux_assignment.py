from __future__ import annotations

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import cyclone_energetics.constants as constants
import cyclone_energetics.gridded_data as gridded_data

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

_FLUX_METADATA: dict[str, dict[str, str]] = {
    "tot_energy": {"units": "PW", "long_name": "poleward total energy flux"},
    "F_Swabs": {"units": "PW", "long_name": "poleward absorbed shortwave radiation flux"},
    "F_Olr": {"units": "PW", "long_name": "poleward outgoing longwave radiation flux"},
    "F_Dhdt": {"units": "PW", "long_name": "poleward MSE storage tendency flux"},
    "F_Shf": {"units": "PW", "long_name": "poleward surface heat flux"},
    "F_TE": {"units": "PW", "long_name": "poleward transient-eddy MSE flux"},
    "F_u_mse": {"units": "PW", "long_name": "poleward zonal MSE advection flux"},
    "F_v_mse": {"units": "PW", "long_name": "poleward meridional MSE advection flux"},
}


def _interpolate_masks_to_era5(
    *,
    mask_data: npt.NDArray[np.floating],
    n_latitude_target: int,
    n_longitude_target: int,
    n_timesteps: int,
) -> npt.NDArray[np.floating]:
    n_latitude_source = mask_data.shape[-2]
    n_longitude_source = mask_data.shape[-1]
    x = np.linspace(0, n_longitude_source, n_longitude_source)
    y = np.linspace(0, n_latitude_source, n_latitude_source)
    xi = np.linspace(0, n_longitude_source, n_longitude_target)
    yi = np.linspace(0, n_latitude_source, n_latitude_target)

    mask_interpolated = np.zeros((n_timesteps, n_latitude_target, n_longitude_target), float)
    for timestep in range(n_timesteps):
        spline = scipy.interpolate.RectBivariateSpline(
            y, x, mask_data[timestep], kx=1, ky=1,
        )
        mask_interpolated[timestep] = spline(yi, xi)
    return mask_interpolated


def _apply_mask(
    *,
    flux_data: npt.NDArray[np.floating],
    mask: npt.NDArray[np.floating],
    threshold: float = 0.5,
) -> npt.NDArray[np.floating]:
    masked = np.copy(flux_data)
    below_threshold = mask[:, 1:-1] < threshold
    masked[below_threshold] = 0.0
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
    n_intensity_cuts = len(constants.INTENSITY_CUTS)

    first_year = year_start
    first_month = constants.MONTH_STRINGS[0]
    with netCDF4.Dataset(
        str(integrated_flux_directory / ("Integrated_Fluxes_%d_%s_.nc" % (first_year, first_month)))
    ) as dataset:
        n_latitude_era5 = len(dataset["lat"][:])
        n_longitude_era5 = len(dataset["lon"][:])

    flux_shape = (n_intensity_cuts, 12, n_latitude_era5, n_longitude_era5)
    flux_arrays = {
        name: {
            suffix: np.zeros(flux_shape, float)
            for suffix in ["total", "cycl", "ant"]
        }
        for name in _FLUX_NAMES
    }

    n_years_processed = 0
    for year in range(year_start, year_end):
        for month_index, month in enumerate(constants.MONTH_STRINGS):
            _LOG.info(
                "Assigning fluxes: year=%s month=%s area_cut=%s",
                year,
                month,
                area_cut,
            )
            max_timesteps = int(constants.TIMESTEPS_PER_MONTH[month_index])
            month_start = constants.MONTH_BOUNDARIES[month_index]
            month_end = constants.MONTH_BOUNDARIES[month_index + 1]

            fluxes = _load_monthly_fluxes(
                integrated_flux_directory=integrated_flux_directory,
                year=year,
                month=month,
                max_timesteps=max_timesteps,
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

            for cut_index in range(n_intensity_cuts):
                intensity_cut_value = constants.INTENSITY_CUTS[cut_index]

                cyclone_mask_combined, anticyclone_mask_combined = _combine_hemisphere_masks(
                    mask_sh=mask_sh,
                    mask_nh=mask_nh,
                    intensity_cut=intensity_cut_value,
                    n_timesteps=max_timesteps,
                    n_latitude_target=n_latitude_era5,
                    n_longitude_target=n_longitude_era5,
                )

                for name in _FLUX_NAMES:
                    flux_total = fluxes[name]
                    flux_cyclone = _apply_mask(
                        flux_data=flux_total, mask=cyclone_mask_combined
                    )
                    flux_anticyclone = _apply_mask(
                        flux_data=flux_total, mask=anticyclone_mask_combined
                    )

                    flux_arrays[name]["total"][cut_index, month_index, :, :] += np.mean(
                        flux_total, axis=0
                    )
                    flux_arrays[name]["cycl"][cut_index, month_index, :, :] += np.mean(
                        flux_cyclone, axis=0
                    )
                    flux_arrays[name]["ant"][cut_index, month_index, :, :] += np.mean(
                        flux_anticyclone, axis=0
                    )

            _LOG.info("Monthly mean calculated for year=%s month=%s", year, month)
        n_years_processed += 1

    for name in _FLUX_NAMES:
        for suffix in ["total", "cycl", "ant"]:
            flux_arrays[name][suffix] /= n_years_processed

    _save_assigned_fluxes(
        output_path=output_directory
        / ("WITH_INT_Cyclones_Sampled_Poleward_Fluxes_%s.nc" % area_cut),
        flux_arrays=flux_arrays,
        composite_latitude=fluxes["_lat"],
        composite_longitude=fluxes["_lon"],
    )


def _load_monthly_fluxes(
    *,
    integrated_flux_directory: pathlib.Path,
    year: int,
    month: str,
    max_timesteps: int,
) -> dict[str, npt.NDArray[np.floating]]:
    result: dict[str, npt.NDArray[np.floating]] = {}
    with netCDF4.Dataset(
        str(
            integrated_flux_directory
            / ("Integrated_Fluxes_%d_%s_.nc" % (year, month))
        )
    ) as dataset:
        result["F_TE"] = np.array(dataset["F_TE_final"][:max_timesteps])
        result["tot_energy"] = np.array(dataset["tot_energy_final"][:max_timesteps])
        result["F_Shf"] = np.array(dataset["F_Shf_final"][:max_timesteps])
        result["F_Swabs"] = np.array(dataset["F_Swabs_final"][:max_timesteps])
        result["F_Olr"] = np.array(dataset["F_Olr_final"][:max_timesteps])
        result["_lat"] = np.array(dataset["lat"][:])
        result["_lon"] = np.array(dataset["lon"][:])

    with netCDF4.Dataset(
        str(
            integrated_flux_directory
            / ("New_Integrated_Fluxes_%d_%s_.nc" % (year, month))
        )
    ) as dataset:
        result["F_Dhdt"] = np.array(dataset["F_Dhdt_final"][:max_timesteps])
        if "F_u_mse_final" in dataset.variables:
            result["F_u_mse"] = np.array(dataset["F_u_mse_final"][:max_timesteps])
        else:
            result["F_u_mse"] = np.zeros_like(result["F_Dhdt"])
        if "F_v_mse_final" in dataset.variables:
            result["F_v_mse"] = np.array(dataset["F_v_mse_final"][:max_timesteps])
        else:
            result["F_v_mse"] = np.zeros_like(result["F_Dhdt"])

    return result


def _load_mask_file(
    *,
    mask_path: pathlib.Path,
    month_start: int,
    month_end: int,
) -> dict[str, npt.NDArray[np.floating]]:
    with netCDF4.Dataset(str(mask_path)) as dataset:
        return {
            "lat": np.array(dataset["lat"][:]),
            "lon": np.array(dataset["lon"][:]),
            "flag_A": np.array(dataset["flag_A"][month_start:month_end]),
            "flag_C": np.array(dataset["flag_C"][month_start:month_end]),
            "intensity_A": np.array(dataset["intensity_A"][month_start:month_end]),
            "intensity_C": np.array(dataset["intensity_C"][month_start:month_end]),
        }


def _combine_hemisphere_masks(
    *,
    mask_sh: dict[str, npt.NDArray[np.floating]],
    mask_nh: dict[str, npt.NDArray[np.floating]],
    intensity_cut: int,
    n_timesteps: int,
    n_latitude_target: int,
    n_longitude_target: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    cyclone_sh = np.copy(mask_sh["flag_C"])
    anticyclone_sh = np.copy(mask_sh["flag_A"])
    cyclone_nh = np.copy(mask_nh["flag_C"])
    anticyclone_nh = np.copy(mask_nh["flag_A"])

    cyclone_sh[mask_sh["intensity_C"] < intensity_cut] = 0
    anticyclone_sh[mask_sh["intensity_A"] < intensity_cut] = 0
    cyclone_nh[mask_nh["intensity_C"] < intensity_cut] = 0
    anticyclone_nh[mask_nh["intensity_A"] < intensity_cut] = 0

    anticyclone_combined = np.concatenate(
        (anticyclone_sh[:, :-1, :], anticyclone_nh), axis=1
    )[:, ::-1, :]
    cyclone_combined = np.concatenate(
        (cyclone_sh[:, :-1, :], cyclone_nh), axis=1
    )[:, ::-1, :]

    n_latitude_source = anticyclone_combined.shape[-2]
    n_longitude_source = anticyclone_combined.shape[-1]

    x = np.linspace(0, n_longitude_source, n_longitude_source)
    y = np.linspace(0, n_latitude_source, n_latitude_source)
    xi = np.linspace(0, n_longitude_source, n_longitude_target)
    yi = np.linspace(0, n_latitude_source, n_latitude_target + 2)

    anticyclone_interpolated = np.zeros((n_timesteps, n_latitude_target + 2, n_longitude_target), float)
    cyclone_interpolated = np.zeros((n_timesteps, n_latitude_target + 2, n_longitude_target), float)

    for timestep in range(n_timesteps):
        spline_anticyclone = scipy.interpolate.RectBivariateSpline(
            y, x, anticyclone_combined[timestep], kx=1, ky=1,
        )
        anticyclone_interpolated[timestep] = spline_anticyclone(yi, xi)
        spline_cyclone = scipy.interpolate.RectBivariateSpline(
            y, x, cyclone_combined[timestep], kx=1, ky=1,
        )
        cyclone_interpolated[timestep] = spline_cyclone(yi, xi)

    return cyclone_interpolated, anticyclone_interpolated


def _save_assigned_fluxes(
    *,
    output_path: pathlib.Path,
    flux_arrays: dict[str, dict[str, npt.NDArray[np.floating]]],
    composite_latitude: npt.NDArray[np.floating],
    composite_longitude: npt.NDArray[np.floating],
) -> None:
    n_intensity_cuts = len(constants.INTENSITY_CUTS)
    with netCDF4.Dataset(
        str(output_path), "w", format="NETCDF4_CLASSIC"
    ) as wfile:
        wfile.createDimension("lon", len(composite_longitude))
        wfile.createDimension("lat", len(composite_latitude))
        wfile.createDimension("time", 12)
        wfile.createDimension("intensity", n_intensity_cuts)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))
        int_var = wfile.createVariable("intensity", "f4", ("intensity",))

        lat_var.units = "degrees_north"
        lat_var.axis = "Y"
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"
        lon_var.units = "degrees_east"
        lon_var.axis = "X"
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"
        time_var.units = "months"
        time_var.axis = "T"
        time_var.long_name = "month of year"
        int_var.units = "CVU"
        int_var.long_name = "vorticity intensity threshold"

        lat_var[:] = composite_latitude
        lon_var[:] = composite_longitude
        time_var[:] = np.arange(12)
        int_var[:] = constants.INTENSITY_CUTS

        suffix_labels = {"total": "", "cycl": " (cyclone-masked)", "ant": " (anticyclone-masked)"}

        for name in _FLUX_NAMES:
            meta = _FLUX_METADATA.get(name, {"units": "PW", "long_name": name})
            for suffix, nc_suffix in [
                ("total", ""),
                ("cycl", "_cycl"),
                ("ant", "_ant"),
            ]:
                var_name = "%s_final%s" % (name, nc_suffix)
                var = wfile.createVariable(
                    var_name, "f4", ("intensity", "time", "lat", "lon")
                )
                var.units = meta["units"]
                var.long_name = meta["long_name"] + suffix_labels[suffix]
                var[:] = flux_arrays[name][suffix]

    _LOG.info("Saved assigned fluxes: %s", output_path)
