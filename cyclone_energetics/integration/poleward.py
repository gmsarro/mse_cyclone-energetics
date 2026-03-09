from __future__ import annotations

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.integrate

import cyclone_energetics.constants as constants
import cyclone_energetics.gridded_data as gridded_data

_LOG = logging.getLogger(__name__)

_VINT_NAME_MAPS: list[dict[str, str]] = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]

ERA5_RADIATION_ACCUMULATION_SECONDS: float = 3600.0


def _resolve_vint_names(*, dataset: netCDF4.Dataset) -> dict[str, str]:
    for mapping in _VINT_NAME_MAPS:
        if all(v in dataset.variables for v in mapping.values()):
            return mapping
    available = list(dataset.variables.keys())
    raise KeyError("Cannot find vint variables in file.  Available: %s" % available)


def _poleward_integrate_core(
    field: npt.NDArray[np.floating],
    *,
    latitude: npt.NDArray[np.floating],
    reference_field: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    latitude_rad = np.deg2rad(latitude)
    cos_latitude = np.cos(latitude_rad)

    reference = field if reference_field is None else reference_field
    global_mean = np.average(reference, weights=cos_latitude, axis=0)

    if field.ndim > 1:
        field_anomaly = field - global_mean[np.newaxis, ...]
        cos_broadcast = cos_latitude.reshape((-1,) + (1,) * (field.ndim - 1))
        field_weighted = field_anomaly * cos_broadcast
    else:
        field_anomaly = field - global_mean
        field_weighted = field_anomaly * cos_latitude

    integral_south = scipy.integrate.cumulative_trapezoid(
        field_weighted[::-1], latitude_rad[::-1], axis=0, initial=None
    )
    integral_north = scipy.integrate.cumulative_trapezoid(
        field_weighted, latitude_rad, axis=0, initial=None
    )
    averaged_integral = (
        2.0
        * np.pi
        * constants.EARTH_RADIUS**2
        * (integral_south[::-1][1:] + integral_north[:-1])
        / 2.0
    )
    return averaged_integral / 1e15


def poleward_integration(
    field: npt.NDArray[np.floating],
    *,
    latitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return _poleward_integrate_core(field, latitude=latitude)


def poleward_integration_individual(
    field: npt.NDArray[np.floating],
    *,
    reference_field: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return _poleward_integrate_core(
        field, latitude=latitude, reference_field=reference_field
    )


def _poleward_integrate_batch(
    *,
    fields: dict[str, npt.NDArray[np.floating]],
    latitude: npt.NDArray[np.floating],
) -> dict[str, npt.NDArray[np.floating]]:
    latitude_rad = np.deg2rad(latitude)
    cos_latitude = np.cos(latitude_rad)
    cos_3d = cos_latitude[np.newaxis, :, np.newaxis]

    results: dict[str, npt.NDArray[np.floating]] = {}
    for name, field in fields.items():
        global_mean = np.average(field, weights=cos_latitude, axis=1)
        field_anomaly = field - global_mean[:, np.newaxis, :]
        field_weighted = field_anomaly * cos_3d

        integral_south = scipy.integrate.cumulative_trapezoid(
            field_weighted[:, ::-1, :], latitude_rad[::-1], axis=1, initial=None
        )
        integral_north = scipy.integrate.cumulative_trapezoid(
            field_weighted, latitude_rad, axis=1, initial=None
        )
        averaged_integral = (
            2.0
            * np.pi
            * constants.EARTH_RADIUS**2
            * (integral_south[:, ::-1, :][:, 1:, :] + integral_north[:, :-1, :])
            / 2.0
        )
        results[name] = averaged_integral / 1e15
    return results


def integrate_fluxes_poleward(
    *,
    year_start: int,
    year_end: int,
    te_directory: pathlib.Path,
    dhdt_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    radiation_directory: pathlib.Path,
    output_directory: pathlib.Path,
    adv_mse_directory: pathlib.Path | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for year in range(year_start, year_end):
        for month_index, month in enumerate(constants.MONTH_STRINGS):
            _LOG.info("Integrating fluxes: year=%s month=%s", year, month)
            max_timesteps = int(constants.TIMESTEPS_PER_MONTH[month_index])

            te_path = te_directory / ("TE_%d_%s.nc" % (year, month))
            with netCDF4.Dataset(str(te_path)) as te_dataset:
                latitude_name = gridded_data.resolve_dimension_name(te_dataset, standard_name="latitude")
                longitude_name = gridded_data.resolve_dimension_name(te_dataset, standard_name="longitude")
                time_values = te_dataset["time"][:max_timesteps]
                latitude_values = te_dataset[latitude_name][:]
                longitude_values = te_dataset[longitude_name][:]
                transient_eddy = np.array(te_dataset["TE"][:max_timesteps])

            dhdt_path = dhdt_directory / ("tend_%d_%s_filtered_2.nc" % (year, month))
            with netCDF4.Dataset(str(dhdt_path)) as dhdt_dataset:
                storage_tendency = np.array(dhdt_dataset["tend_filtered"][:max_timesteps, ::-1])

            vint_path = vint_directory / ("era5_vint_%d_%s_filtered.nc" % (year, month))
            with netCDF4.Dataset(str(vint_path)) as vint_dataset:
                vint_names = _resolve_vint_names(dataset=vint_dataset)
                dry_static_energy_flux = np.array(vint_dataset[vint_names["vigd"]][:max_timesteps, ::-1])
                moisture_flux = np.array(vint_dataset[vint_names["vimdf"]][:max_timesteps, ::-1])
                thermal_energy_flux = np.array(vint_dataset[vint_names["vithed"]][:max_timesteps, ::-1])

            latitude_array = np.copy(np.asarray(latitude_values))

            total_energy = (
                dry_static_energy_flux
                + moisture_flux * constants.LATENT_HEAT_VAPORIZATION
                + thermal_energy_flux
            )

            radiation_path = radiation_directory / ("era5_rad_%d_%s.6hrly.nc" % (year, month))
            with netCDF4.Dataset(str(radiation_path)) as radiation_dataset:
                top_solar_radiation = np.nan_to_num(
                    np.array(radiation_dataset["tsr"][:max_timesteps]) / ERA5_RADIATION_ACCUMULATION_SECONDS,
                    nan=0.0,
                )
                surface_solar_radiation = np.nan_to_num(
                    np.array(radiation_dataset["ssr"][:max_timesteps]) / ERA5_RADIATION_ACCUMULATION_SECONDS,
                    nan=0.0,
                )
                top_thermal_radiation = np.nan_to_num(
                    np.array(radiation_dataset["ttr"][:max_timesteps]) / ERA5_RADIATION_ACCUMULATION_SECONDS,
                    nan=0.0,
                )

            storage_tendency_clean = np.nan_to_num(storage_tendency, nan=0.0)
            surface_heat_flux = (
                total_energy
                - (top_solar_radiation - surface_solar_radiation)
                - top_thermal_radiation
                + storage_tendency_clean
            )
            absorbed_shortwave = top_solar_radiation - surface_solar_radiation
            outgoing_longwave = top_thermal_radiation

            batch = _poleward_integrate_batch(
                fields={
                    "te": transient_eddy,
                    "tot_energy": total_energy,
                    "swabs": absorbed_shortwave,
                    "olr": outgoing_longwave,
                    "shf": surface_heat_flux,
                    "dhdt": storage_tendency_clean,
                },
                latitude=latitude_array,
            )

            _LOG.info("Integration completed for year=%s month=%s", year, month)

            _save_integrated_fluxes(
                output_path=output_directory
                / ("Integrated_Fluxes_%d_%s_.nc" % (year, month)),
                latitude=latitude_values[1:-1],
                longitude=longitude_values,
                time_values=time_values,
                total_energy=batch["tot_energy"],
                absorbed_shortwave=batch["swabs"],
                outgoing_longwave=batch["olr"],
                surface_heat_flux=batch["shf"],
                transient_eddy=batch["te"],
            )

            integrated_zonal_advection: npt.NDArray[np.floating] | None = None
            integrated_meridional_advection: npt.NDArray[np.floating] | None = None
            if adv_mse_directory is not None:
                adv_path = adv_mse_directory / (
                    "Adv_%d_%s_filtered.nc" % (year, month)
                )
                if adv_path.exists():
                    with netCDF4.Dataset(str(adv_path)) as adv_dataset:
                        adv_latitude_name = gridded_data.resolve_dimension_name(
                            adv_dataset, standard_name="latitude"
                        )
                        adv_longitude_name = gridded_data.resolve_dimension_name(
                            adv_dataset, standard_name="longitude"
                        )
                        advection_latitude = np.array(
                            adv_dataset[adv_latitude_name][::-1], dtype=np.float64
                        )
                        advection_longitude = np.array(
                            adv_dataset[adv_longitude_name][:], dtype=np.float64
                        )
                        zonal_advection_raw = np.nan_to_num(
                            np.array(
                                adv_dataset["u_mse_filtered"][:max_timesteps, ::-1],
                                dtype=np.float64,
                            ),
                            nan=0.0,
                        )
                        meridional_advection_raw = np.nan_to_num(
                            np.array(
                                adv_dataset["v_mse_filtered"][:max_timesteps, ::-1],
                                dtype=np.float64,
                            ),
                            nan=0.0,
                        )

                    adv_batch = _poleward_integrate_batch(
                        fields={"u_mse": zonal_advection_raw, "v_mse": meridional_advection_raw},
                        latitude=advection_latitude,
                    )
                    integrated_zonal_advection = adv_batch["u_mse"]
                    integrated_meridional_advection = adv_batch["v_mse"]
                    _LOG.info(
                        "Advective MSE integration done for year=%s month=%s",
                        year,
                        month,
                    )

            _save_new_integrated_fluxes(
                output_path=output_directory
                / ("New_Integrated_Fluxes_%d_%s_.nc" % (year, month)),
                latitude=latitude_values[1:-1],
                longitude=longitude_values,
                time_values=time_values,
                storage_tendency=batch["dhdt"],
                zonal_advection=integrated_zonal_advection,
                meridional_advection=integrated_meridional_advection,
            )


def _save_integrated_fluxes(
    *,
    output_path: pathlib.Path,
    latitude: npt.NDArray[np.floating],
    longitude: npt.NDArray[np.floating],
    time_values: npt.NDArray[np.floating],
    total_energy: npt.NDArray[np.floating],
    absorbed_shortwave: npt.NDArray[np.floating],
    outgoing_longwave: npt.NDArray[np.floating],
    surface_heat_flux: npt.NDArray[np.floating],
    transient_eddy: npt.NDArray[np.floating],
) -> None:
    with netCDF4.Dataset(
        str(output_path), "w", format="NETCDF3_CLASSIC"
    ) as wfile:
        wfile.createDimension("lon", len(longitude))
        wfile.createDimension("lat", len(latitude))
        wfile.createDimension("time", None)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))

        te_var = wfile.createVariable("F_TE_final", "f4", ("time", "lat", "lon"))
        tot_var = wfile.createVariable("tot_energy_final", "f4", ("time", "lat", "lon"))
        swabs_var = wfile.createVariable("F_Swabs_final", "f4", ("time", "lat", "lon"))
        olr_var = wfile.createVariable("F_Olr_final", "f4", ("time", "lat", "lon"))
        shf_var = wfile.createVariable("F_Shf_final", "f4", ("time", "lat", "lon"))

        lat_var.units = "degrees_north"
        lat_var.axis = "Y"
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"
        lon_var.units = "degrees_east"
        lon_var.axis = "X"
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"
        time_var.units = "days"
        time_var.axis = "T"
        time_var.long_name = "time"

        te_var.units = "PW"
        te_var.long_name = "poleward transient-eddy MSE flux"
        tot_var.units = "PW"
        tot_var.long_name = "poleward total energy flux"
        swabs_var.units = "PW"
        swabs_var.long_name = "poleward absorbed shortwave radiation flux"
        olr_var.units = "PW"
        olr_var.long_name = "poleward outgoing longwave radiation flux"
        shf_var.units = "PW"
        shf_var.long_name = "poleward surface heat flux"

        lat_var[:] = latitude
        lon_var[:] = longitude
        time_var[:] = time_values
        te_var[:] = transient_eddy
        tot_var[:] = total_energy
        swabs_var[:] = absorbed_shortwave
        olr_var[:] = outgoing_longwave
        shf_var[:] = surface_heat_flux
    _LOG.info("Saved integrated fluxes: %s", output_path)


def _save_new_integrated_fluxes(
    *,
    output_path: pathlib.Path,
    latitude: npt.NDArray[np.floating],
    longitude: npt.NDArray[np.floating],
    time_values: npt.NDArray[np.floating],
    storage_tendency: npt.NDArray[np.floating],
    zonal_advection: npt.NDArray[np.floating] | None = None,
    meridional_advection: npt.NDArray[np.floating] | None = None,
) -> None:
    with netCDF4.Dataset(
        str(output_path), "w", format="NETCDF3_CLASSIC"
    ) as wfile:
        wfile.createDimension("lon", len(longitude))
        wfile.createDimension("lat", len(latitude))
        wfile.createDimension("time", None)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))

        dhdt_var = wfile.createVariable("F_Dhdt_final", "f4", ("time", "lat", "lon"))

        lat_var.units = "degrees_north"
        lat_var.axis = "Y"
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"
        lon_var.units = "degrees_east"
        lon_var.axis = "X"
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"
        time_var.units = "days"
        time_var.axis = "T"
        time_var.long_name = "time"

        dhdt_var.units = "PW"
        dhdt_var.long_name = "poleward MSE storage tendency flux"

        lat_var[:] = latitude
        lon_var[:] = longitude
        time_var[:] = time_values
        dhdt_var[:] = storage_tendency

        if zonal_advection is not None:
            u_mse_var = wfile.createVariable(
                "F_u_mse_final", "f4", ("time", "lat", "lon")
            )
            u_mse_var.units = "PW"
            u_mse_var.long_name = "poleward zonal MSE advection flux"
            u_mse_var[:] = zonal_advection

        if meridional_advection is not None:
            v_mse_var = wfile.createVariable(
                "F_v_mse_final", "f4", ("time", "lat", "lon")
            )
            v_mse_var.units = "PW"
            v_mse_var.long_name = "poleward meridional MSE advection flux"
            v_mse_var[:] = meridional_advection

    _LOG.info("Saved new integrated fluxes: %s", output_path)
