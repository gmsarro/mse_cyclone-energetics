from __future__ import annotations

import logging
import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.integrate

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

# ERA5 CDS variable names changed ~2022.  Old data uses the GRIB parameter
# IDs (p85.162, p84.162, p83.162), while newer data uses the short names
# (vigd, vimdf, vithed).  We check for both.
_VINT_NAME_MAPS: list = [
    {"vigd": "vigd_filtered", "vimdf": "vimdf_filtered", "vithed": "vithed_filtered"},
    {"vigd": "p85.162_filtered", "vimdf": "p84.162_filtered", "vithed": "p83.162_filtered"},
]


def _resolve_vint_names(
    *,
    ds: netCDF4.Dataset,
) -> dict:
    """Return the correct variable-name mapping for a smoothed vint file."""
    for mapping in _VINT_NAME_MAPS:
        if all(v in ds.variables for v in mapping.values()):
            return mapping
    available = list(ds.variables.keys())
    raise KeyError(
        "Cannot find vint variables in file.  Available: %s" % available
    )


def poleward_integration(
    field: npt.NDArray[np.floating],
    *,
    latitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Poleward-integrate a 2-D+ field along the latitude axis (axis 0).

    Accepts shape ``(n_lat, ...)`` and returns ``(n_lat - 2, ...)``.
    The global mean is removed before integration and the result is expressed
    in PW (divided by 1e15).

    The integration respects spherical geometry: the field is weighted by
    ``cos(lat)`` and integrated over ``d(lat_rad)``, then multiplied by
    ``2 * pi * a^2`` to recover the full zonal-mean poleward transport.

    Both north-to-south and south-to-north integrations are performed, and
    the average is returned to reduce accumulation bias.
    """
    lat_rad = np.deg2rad(latitude)
    cos_lat = np.cos(lat_rad)

    weights = cos_lat
    if field.ndim > 1:
        extra_axes = tuple(range(1, field.ndim))
        mean_val = np.average(field, weights=weights, axis=0)
        field_anom = field - mean_val[np.newaxis, ...]
        cos_broad = cos_lat.reshape((-1,) + (1,) * (field.ndim - 1))
        field_weighted = field_anom * cos_broad
    else:
        mean_val = np.average(field, weights=weights, axis=0)
        field_anom = field - mean_val
        field_weighted = field_anom * cos_lat

    integral_south = scipy.integrate.cumulative_trapezoid(
        field_weighted[::-1], lat_rad[::-1], axis=0, initial=None
    )
    integral_north = scipy.integrate.cumulative_trapezoid(
        field_weighted, lat_rad, axis=0, initial=None
    )
    avg_integral = (
        2.0
        * np.pi
        * constants.EARTH_RADIUS ** 2
        * (integral_south[::-1][1:] + integral_north[:-1])
        / 2.0
    )
    return avg_integral / 1e15


def poleward_integration_individual(
    field: npt.NDArray[np.floating],
    *,
    reference_field: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Like :func:`poleward_integration` but subtracts the global mean of a
    *separate* reference field instead of the field itself."""
    lat_rad = np.deg2rad(latitude)
    cos_lat = np.cos(lat_rad)

    mean_val = np.average(reference_field, weights=cos_lat, axis=0)
    if field.ndim > 1:
        field_anom = field - mean_val[np.newaxis, ...]
        cos_broad = cos_lat.reshape((-1,) + (1,) * (field.ndim - 1))
        field_weighted = field_anom * cos_broad
    else:
        field_anom = field - mean_val
        field_weighted = field_anom * cos_lat

    integral_south = scipy.integrate.cumulative_trapezoid(
        field_weighted[::-1], lat_rad[::-1], axis=0, initial=None
    )
    integral_north = scipy.integrate.cumulative_trapezoid(
        field_weighted, lat_rad, axis=0, initial=None
    )
    avg_integral = (
        2.0
        * np.pi
        * constants.EARTH_RADIUS ** 2
        * (integral_south[::-1][1:] + integral_north[:-1])
        / 2.0
    )
    return avg_integral / 1e15


def _poleward_integrate_batch(
    *,
    fields: dict[str, npt.NDArray[np.floating]],
    latitude: npt.NDArray[np.floating],
) -> dict[str, npt.NDArray[np.floating]]:
    """Integrate all flux fields in one vectorised pass.

    Each value in *fields* must have shape ``(n_time, n_lat, n_lon)``.
    Returns arrays of shape ``(n_time, n_lat - 2, n_lon)``.
    """
    lat_rad = np.deg2rad(latitude)
    cos_lat = np.cos(lat_rad)
    cos_3d = cos_lat[np.newaxis, :, np.newaxis]

    results: dict[str, npt.NDArray[np.floating]] = {}
    for name, field in fields.items():
        global_mean = np.average(field, weights=cos_lat, axis=1)
        field_anom = field - global_mean[:, np.newaxis, :]
        field_weighted = field_anom * cos_3d

        integral_south = scipy.integrate.cumulative_trapezoid(
            field_weighted[:, ::-1, :], lat_rad[::-1], axis=1, initial=None
        )
        integral_north = scipy.integrate.cumulative_trapezoid(
            field_weighted, lat_rad, axis=1, initial=None
        )
        avg_integral = (
            2.0
            * np.pi
            * constants.EARTH_RADIUS ** 2
            * (integral_south[:, ::-1, :][:, 1:, :] + integral_north[:, :-1, :])
            / 2.0
        )
        results[name] = avg_integral / 1e15
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
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for year in range(year_start, year_end):
        for month_idx, month in enumerate(constants.MONTH_STRINGS):
            _LOG.info("Integrating fluxes: year=%s month=%s", year, month)
            max_day = int(constants.TIMESTEPS_PER_MONTH[month_idx])

            with netCDF4.Dataset(
                str(te_directory / ("TE_%d_%s.nc" % (year, month)))
            ) as ds_te:
                time_vals = ds_te["time"][:max_day]
                lat = ds_te["latitude"][:]
                lon = ds_te["longitude"][:]
                te_n = np.array(ds_te["TE"][:max_day, :, :])

            with netCDF4.Dataset(
                str(
                    dhdt_directory
                    / ("tend_%d_%s_filtered_2.nc" % (year, month))
                )
            ) as ds_dhdt:
                dhdt = np.array(ds_dhdt["tend_filtered"][:max_day, ::-1, :])

            with netCDF4.Dataset(
                str(
                    vint_directory
                    / ("era5_vint_%d_%s_filtered.nc" % (year, month))
                )
            ) as ds_vint:
                vn = _resolve_vint_names(ds=ds_vint)
                vigd = np.array(ds_vint[vn["vigd"]][:max_day, ::-1, :])
                vimdf = np.array(ds_vint[vn["vimdf"]][:max_day, ::-1, :])
                vithed = np.array(ds_vint[vn["vithed"]][:max_day, ::-1, :])

            # Use latitude from the TE file for all integrations, matching
            # the original code (make_TE_int_2025.py line: latitude = np.copy(lat))
            latitude = np.copy(np.asarray(lat))

            tot_energy = (
                vigd + vimdf * constants.LATENT_HEAT_VAPORIZATION + vithed
            )

            with netCDF4.Dataset(
                str(
                    radiation_directory
                    / ("era5_rad_%d_%s.6hrly.nc" % (year, month))
                )
            ) as ds_rad:
                tsr = np.nan_to_num(
                    np.array(ds_rad["tsr"][:max_day, :, :]) / 3600.0, nan=0.0
                )
                ssr = np.nan_to_num(
                    np.array(ds_rad["ssr"][:max_day, :, :]) / 3600.0, nan=0.0
                )
                ttr = np.nan_to_num(
                    np.array(ds_rad["ttr"][:max_day, :, :]) / 3600.0, nan=0.0
                )

            f_dhdt = np.nan_to_num(dhdt, nan=0.0)
            f_shf = tot_energy - (tsr - ssr) - ttr + f_dhdt
            f_swabs = tsr - ssr
            f_olr = ttr

            batch = _poleward_integrate_batch(
                fields={
                    "te": te_n,
                    "tot_energy": tot_energy,
                    "swabs": f_swabs,
                    "olr": f_olr,
                    "shf": f_shf,
                    "dhdt": f_dhdt,
                },
                latitude=latitude,
            )

            _LOG.info("Integration completed for year=%s month=%s", year, month)

            _save_integrated_fluxes(
                output_path=output_directory
                / ("Integrated_Fluxes_%d_%s_.nc" % (year, month)),
                lat=lat[1:-1],
                lon=lon,
                time_vals=time_vals,
                tot_energy=batch["tot_energy"],
                f_swabs=batch["swabs"],
                f_olr=batch["olr"],
                f_shf=batch["shf"],
                f_te=batch["te"],
            )

            _save_new_integrated_fluxes(
                output_path=output_directory
                / ("New_Integrated_Fluxes_%d_%s_.nc" % (year, month)),
                lat=lat[1:-1],
                lon=lon,
                time_vals=time_vals,
                f_dhdt=batch["dhdt"],
            )


def _save_integrated_fluxes(
    *,
    output_path: pathlib.Path,
    lat: npt.NDArray,
    lon: npt.NDArray,
    time_vals: npt.NDArray,
    tot_energy: npt.NDArray,
    f_swabs: npt.NDArray,
    f_olr: npt.NDArray,
    f_shf: npt.NDArray,
    f_te: npt.NDArray,
) -> None:
    with netCDF4.Dataset(str(output_path), "w", format="NETCDF3_CLASSIC") as wfile:
        wfile.createDimension("lon", len(lon))
        wfile.createDimension("lat", len(lat))
        wfile.createDimension("time", None)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))

        te_var = wfile.createVariable("F_TE_final", "f4", ("time", "lat", "lon"))
        tot_var = wfile.createVariable(
            "tot_energy_final", "f4", ("time", "lat", "lon")
        )
        swabs_var = wfile.createVariable(
            "F_Swabs_final", "f4", ("time", "lat", "lon")
        )
        olr_var = wfile.createVariable("F_Olr_final", "f4", ("time", "lat", "lon"))
        shf_var = wfile.createVariable("F_Shf_final", "f4", ("time", "lat", "lon"))

        lat_var.units = "Degrees North"
        lat_var.axis = "Y"
        lon_var.units = "Degrees East"
        lon_var.axis = "X"
        time_var.units = "days"
        time_var.axis = "T"

        lat_var[:] = lat
        lon_var[:] = lon
        time_var[:] = time_vals
        te_var[:, :, :] = f_te
        tot_var[:, :, :] = tot_energy
        swabs_var[:, :, :] = f_swabs
        olr_var[:, :, :] = f_olr
        shf_var[:, :, :] = f_shf
    _LOG.info("Saved integrated fluxes: %s", output_path)


def _save_new_integrated_fluxes(
    *,
    output_path: pathlib.Path,
    lat: npt.NDArray,
    lon: npt.NDArray,
    time_vals: npt.NDArray,
    f_dhdt: npt.NDArray,
) -> None:
    with netCDF4.Dataset(str(output_path), "w", format="NETCDF3_CLASSIC") as wfile:
        wfile.createDimension("lon", len(lon))
        wfile.createDimension("lat", len(lat))
        wfile.createDimension("time", None)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))
        time_var = wfile.createVariable("time", "f4", ("time",))

        dhdt_var = wfile.createVariable(
            "F_Dhdt_final", "f4", ("time", "lat", "lon")
        )

        lat_var.units = "Degrees North"
        lat_var.axis = "Y"
        lon_var.units = "Degrees East"
        lon_var.axis = "X"
        time_var.units = "days"
        time_var.axis = "T"

        lat_var[:] = lat
        lon_var[:] = lon
        time_var[:] = time_vals
        dhdt_var[:, :, :] = f_dhdt
    _LOG.info("Saved new integrated fluxes: %s", output_path)
