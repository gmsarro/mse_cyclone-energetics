"""Build cyclone-centred composites of *local* (W m⁻²) energy-budget terms.

This module creates the "W_M" composite files used for the cyclone-centred
spatial maps (e.g. Figure 7).  Unlike the standard composites module which
composites *meridionally integrated* fluxes, this module reads the raw
(non-integrated) ERA5 fields:

    composite_energy = vigd + vimdf × L_v + vithed          (column MSE)
    composite_Shf    = energy - (TSR - SSR) - TTR + Dhdt    (residual SHF, W m⁻²)

The geopotential height (Z), 2-m temperature (T), 850-hPa specific humidity
(Q), and filtered vorticity (VO) are also composited to provide the spatial
structure around each cyclone centre.
"""

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
_VO_GRID_SIZE: int = 20
_GEOPOTENTIAL_LEVEL_INDEX: int = 30
_Q_LEVEL_INDEX: int = 35  # 850 hPa


def build_wm_composites(
    *,
    year_start: int,
    year_end: int,
    hemispheres: list,
    track_types: list,
    track_directory: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    dhdt_directory: pathlib.Path,
    radiation_directory: pathlib.Path,
    z_directory: pathlib.Path,
    t2m_directory: pathlib.Path,
    q_directory: pathlib.Path,
    vorticity_directory: pathlib.Path,
    mask_directory_sh: pathlib.Path,
    mask_directory_nh: pathlib.Path,
    output_directory: pathlib.Path,
    intensity_threshold: int = 0,
) -> None:
    """Build W/m² composite files for each hemisphere × track type.

    Parameters
    ----------
    intensity_threshold : int
        Only include timesteps where track intensity > this value.
        Use 0 for all cyclones, 6 for intense only.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    # Read reference grids once
    with netCDF4.Dataset(
        str(integrated_flux_directory / ("Integrated_Fluxes_%d_01_.nc" % year_start))
    ) as ds:
        flux_lat = ds["lat"][:]
        flux_lon = ds["lon"][:]

    with netCDF4.Dataset(
        str(vint_directory / ("era5_vint_%d_01_filtered.nc" % year_start))
    ) as ds:
        vint_lat = ds["latitude"][::-1]

    with netCDF4.Dataset(
        str(mask_directory_nh / ("MASK_NH_%d.nc" % year_start))
    ) as ds:
        mask_lat_nh = ds["lat"][:]

    with netCDF4.Dataset(
        str(mask_directory_sh / ("MASK_SH_%d.nc" % year_start))
    ) as ds:
        mask_lat_sh = ds["lat"][:]

    latitude_t = np.concatenate((mask_lat_sh[:-1], mask_lat_nh), axis=0)[::-1]
    longitude_t = mask_lat_nh  # mask lon == mask lat in the track grid

    with netCDF4.Dataset(
        str(mask_directory_nh / ("MASK_NH_%d.nc" % year_start))
    ) as ds:
        longitude_t = ds["lon"][:]

    # Read vorticity grid
    vo_sample = str(vorticity_directory / ("VO.anom.T42.%d.v146.nc" % year_start))
    with netCDF4.Dataset(vo_sample) as ds:
        vo_lats = ds["lat"][:]
        vo_lons = ds["lon"][:]

    for hemisphere in hemispheres:
        for rotation in track_types:
            _LOG.info(
                "Building W/m² composites: hemisphere=%s rotation=%s threshold=%s",
                hemisphere,
                rotation,
                intensity_threshold,
            )
            _build_single_wm_composite(
                hemisphere=hemisphere,
                rotation=rotation,
                year_start=year_start,
                year_end=year_end,
                intensity_threshold=intensity_threshold,
                track_directory=track_directory,
                integrated_flux_directory=integrated_flux_directory,
                vint_directory=vint_directory,
                dhdt_directory=dhdt_directory,
                radiation_directory=radiation_directory,
                z_directory=z_directory,
                t2m_directory=t2m_directory,
                q_directory=q_directory,
                vorticity_directory=vorticity_directory,
                output_directory=output_directory,
                flux_lat=flux_lat,
                flux_lon=flux_lon,
                vint_lat=vint_lat,
                vo_lats=vo_lats,
                vo_lons=vo_lons,
                latitude_t=latitude_t,
                longitude_t=longitude_t,
            )


def _build_single_wm_composite(
    *,
    hemisphere: str,
    rotation: str,
    year_start: int,
    year_end: int,
    intensity_threshold: int,
    track_directory: pathlib.Path,
    integrated_flux_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    dhdt_directory: pathlib.Path,
    radiation_directory: pathlib.Path,
    z_directory: pathlib.Path,
    t2m_directory: pathlib.Path,
    q_directory: pathlib.Path,
    vorticity_directory: pathlib.Path,
    output_directory: pathlib.Path,
    flux_lat: npt.NDArray,
    flux_lon: npt.NDArray,
    vint_lat: npt.NDArray,
    vo_lats: npt.NDArray,
    vo_lons: npt.NDArray,
    latitude_t: npt.NDArray,
    longitude_t: npt.NDArray,
) -> None:
    track_path = (
        track_directory
        / ("%s_VO_anom_T42_ERA5_1979_2018_all%s.nc" % (rotation, hemisphere))
    )
    with netCDF4.Dataset(str(track_path)) as ds:
        track_time = ds["time"][:].data - 1
        track_lon = ds["longitude"][:].data
        track_lat = ds["latitude"][:].data
        track_int = ds["intensity"][:].data

    n_box = _COMPOSITE_GRID_SIZE
    n_vo = _VO_GRID_SIZE
    composite_te = np.zeros((12, n_box, n_box), float)
    composite_energy = np.zeros((12, n_box, n_box), float)
    composite_shf = np.zeros((12, n_box, n_box), float)
    composite_dhdt = np.zeros((12, n_box, n_box), float)
    composite_swabs = np.zeros((12, n_box, n_box), float)
    composite_olr = np.zeros((12, n_box, n_box), float)
    composite_z = np.zeros((12, n_box, n_box), float)
    composite_t = np.zeros((12, n_box, n_box), float)
    composite_q = np.zeros((12, n_box, n_box), float)
    composite_vo = np.zeros((12, n_vo, n_vo), float)
    number_of_cyclones = np.zeros(12, float)
    composite_lat_saved = None
    composite_lon_saved = None

    hemi_adj = 1 if hemisphere != "SH" else -1

    for time_step in range(len(track_time)):
        if intensity_threshold > 0 and track_int[time_step] <= intensity_threshold:
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

        # High-res box (0.25° grid)
        lat_max = (int(lat_storm * 4)) / 4.0 + _COMPOSITE_BOX_DEGREES
        lat_min = (int(lat_storm * 4)) / 4.0 - _COMPOSITE_BOX_DEGREES
        lon_max = (int(lon_storm * 4)) / 4.0 + _COMPOSITE_BOX_DEGREES
        lon_min = (int(lon_storm * 4)) / 4.0 - _COMPOSITE_BOX_DEGREES

        # Coarse box (1.5° grid) for vorticity
        lat_max_small = (int(lat_storm * 2 / 3)) * 3 / 2 + 15
        lat_min_small = (int(lat_storm * 2 / 3)) * 3 / 2 - 15
        lon_max_small = (int(lon_storm * 2 / 3)) * 3 / 2 + 15
        lon_min_small = (int(lon_storm * 2 / 3)) * 3 / 2 - 15

        if lon_max >= 360 or lon_min < 0:
            continue

        a_idx = np.where(flux_lat == lat_max)
        b_idx = np.where(flux_lat == lat_min)
        c_idx = np.where(flux_lon == lon_min)
        d_idx = np.where(flux_lon == lon_max)

        if any(len(idx[0]) == 0 for idx in [a_idx, b_idx, c_idx, d_idx]):
            continue

        a_i = int(a_idx[0][0])
        b_i = int(b_idx[0][0])
        c_i = int(c_idx[0][0])
        d_i = int(d_idx[0][0])

        # Full-latitude indices (vint grid, flipped)
        a_flip = np.where(vint_lat == lat_max)
        b_flip = np.where(vint_lat == lat_min)
        if len(a_flip[0]) == 0 or len(b_flip[0]) == 0:
            continue
        a_fi = int(a_flip[0][0])
        b_fi = int(b_flip[0][0])

        # TE from integrated file
        with netCDF4.Dataset(
            str(
                integrated_flux_directory
                / ("Integrated_Fluxes_%d_%s_.nc" % (year, month_str))
            )
        ) as ds:
            composite_te[pythonic_month, :, :] += ds["F_TE_final"][
                day_of_month, a_i:b_i, c_i:d_i
            ]
            composite_lat_saved = ds["lat"][a_i:b_i]
            composite_lon_saved = ds["lon"][c_i:d_i]

        # Vint fields (smoothed) — note: latitude is flipped
        with netCDF4.Dataset(
            str(
                vint_directory
                / ("era5_vint_%d_%s_filtered.nc" % (year, month_str))
            )
        ) as ds:
            first = ds["p85.162_filtered"][
                day_of_month, a_fi:b_fi, c_i:d_i
            ]
            second = ds["p84.162_filtered"][
                day_of_month, a_fi:b_fi, c_i:d_i
            ]
            third = ds["p83.162_filtered"][
                day_of_month, a_fi:b_fi, c_i:d_i
            ]

        energy_local = (
            first[::-1, :] + second[::-1, :] * constants.LATENT_HEAT_VAPORIZATION + third[::-1, :]
        )
        composite_energy[pythonic_month, :, :] += energy_local

        # dh/dt
        with netCDF4.Dataset(
            str(
                dhdt_directory
                / ("tend_%d_%s_filtered.nc" % (year, month_str))
            )
        ) as ds:
            dhdt_local = ds["tend_filtered"][
                day_of_month, a_fi:b_fi, c_i:d_i
            ]
        composite_dhdt[pythonic_month, :, :] += dhdt_local[::-1, :]

        # Radiation
        with netCDF4.Dataset(
            str(
                radiation_directory
                / ("era5_rad_%d_%s.6hrly.nc" % (year, month_str))
            )
        ) as ds:
            # Full latitude grid for rad — same as flux_lat (not flipped)
            a_all = np.where(ds["latitude"][:] == lat_max)
            b_all = np.where(ds["latitude"][:] == lat_min)
            if len(a_all[0]) == 0 or len(b_all[0]) == 0:
                continue
            a_ai = int(a_all[0][0])
            b_ai = int(b_all[0][0])

            tsr_raw = ds["tsr"][day_of_month, a_ai:b_ai, c_i:d_i]
            ssr_raw = ds["ssr"][day_of_month, a_ai:b_ai, c_i:d_i]
            ttr_raw = ds["ttr"][day_of_month, a_ai:b_ai, c_i:d_i]

        tsr = np.nan_to_num(tsr_raw / 3600.0, nan=0.0)
        ssr = np.nan_to_num(ssr_raw / 3600.0, nan=0.0)
        ttr = np.nan_to_num(ttr_raw / 3600.0, nan=0.0)

        composite_swabs[pythonic_month, :, :] += tsr - ssr
        composite_olr[pythonic_month, :, :] += ttr
        composite_shf[pythonic_month, :, :] += (
            energy_local - (tsr - ssr) - ttr + dhdt_local[::-1, :]
        )

        # Geopotential height (level 30 ≈ 500 hPa)
        with netCDF4.Dataset(
            str(
                z_directory
                / ("era5_z_%d_%s.6hrly.nc" % (year, month_str))
            )
        ) as ds:
            a_z = np.where(ds["latitude"][:] == lat_max)
            b_z = np.where(ds["latitude"][:] == lat_min)
            if len(a_z[0]) == 0 or len(b_z[0]) == 0:
                continue
            composite_z[pythonic_month, :, :] += (
                ds["z"][
                    day_of_month,
                    _GEOPOTENTIAL_LEVEL_INDEX,
                    int(a_z[0][0]) : int(b_z[0][0]),
                    c_i:d_i,
                ]
                / constants.GRAVITY
            )

        # 2-m temperature
        with netCDF4.Dataset(
            str(
                t2m_directory
                / ("era5_t2m_%d_%s.6hrly.nc" % (year, month_str))
            )
        ) as ds:
            a_t = np.where(ds["latitude"][:] == lat_max)
            b_t = np.where(ds["latitude"][:] == lat_min)
            if len(a_t[0]) > 0 and len(b_t[0]) > 0:
                composite_t[pythonic_month, :, :] += ds["t2m"][
                    day_of_month,
                    int(a_t[0][0]) : int(b_t[0][0]),
                    c_i:d_i,
                ]

        # 850 hPa specific humidity
        with netCDF4.Dataset(
            str(
                q_directory
                / ("era5_q_%d_%s.6hrly.nc" % (year, month_str))
            )
        ) as ds:
            a_q = np.where(ds["latitude"][:] == lat_max)
            b_q = np.where(ds["latitude"][:] == lat_min)
            if len(a_q[0]) > 0 and len(b_q[0]) > 0:
                composite_q[pythonic_month, :, :] += ds["q"][
                    day_of_month,
                    _Q_LEVEL_INDEX,
                    int(a_q[0][0]) : int(b_q[0][0]),
                    c_i:d_i,
                ]

        # Filtered vorticity (coarse 1.5° grid)
        if lon_max_small < 360 and lon_min_small >= 0:
            a_vo_flip = np.where(vo_lats == lat_min_small)
            b_vo_flip = np.where(vo_lats == lat_max_small)
            c_vo = np.where(vo_lons == lon_min_small)
            d_vo = np.where(vo_lons == lon_max_small)
            if all(
                len(idx[0]) > 0
                for idx in [a_vo_flip, b_vo_flip, c_vo, d_vo]
            ):
                with netCDF4.Dataset(
                    str(
                        vorticity_directory
                        / ("VO.anom.T42.%d.v146.nc" % year)
                    )
                ) as ds:
                    vo_patch = ds["VO"][
                        day_of_year,
                        int(a_vo_flip[0][0]) : int(b_vo_flip[0][0]),
                        int(c_vo[0][0]) : int(d_vo[0][0]),
                    ]
                composite_vo[pythonic_month, :, :] += vo_patch[::-1, :]

        number_of_cyclones[pythonic_month] += 1

    if composite_lat_saved is None or composite_lon_saved is None:
        _LOG.info(
            "No W/m² composites built for %s %s — no qualifying cyclones",
            hemisphere,
            rotation,
        )
        return

    prefix = "Intense_" if intensity_threshold > 0 else ""
    output_name = "%sComposites_W_M_%s_%s_.nc" % (prefix, rotation, hemisphere)
    _save_wm_composite(
        output_path=output_directory / output_name,
        composite_lat=composite_lat_saved,
        composite_lon=composite_lon_saved,
        composite_te=composite_te,
        composite_energy=composite_energy,
        composite_shf=composite_shf,
        composite_dhdt=composite_dhdt,
        composite_swabs=composite_swabs,
        composite_olr=composite_olr,
        composite_z=composite_z,
        composite_t=composite_t,
        composite_q=composite_q,
        composite_vo=composite_vo,
        number_of_cyclones=number_of_cyclones,
    )


def _resolve_month(
    *,
    day_of_year: int,
) -> tuple:
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


def _save_wm_composite(
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
    composite_z: npt.NDArray,
    composite_t: npt.NDArray,
    composite_q: npt.NDArray,
    composite_vo: npt.NDArray,
    number_of_cyclones: npt.NDArray,
) -> None:
    # Interpolate VO from 20×20 to 120×120
    x_vo = np.linspace(0, 20, 20)
    y_vo = np.linspace(0, 20, 120)

    n_box = composite_te.shape[1]
    with netCDF4.Dataset(
        str(output_path), "w", format="NETCDF3_CLASSIC"
    ) as wfile:
        wfile.createDimension("lon", n_box)
        wfile.createDimension("lat", n_box)
        wfile.createDimension("month", 12)

        lon_var = wfile.createVariable("lon", "f4", ("lon",))
        lat_var = wfile.createVariable("lat", "f4", ("lat",))

        te_var = wfile.createVariable("composite_TE", "f4", ("month", "lat", "lon"))
        energy_var = wfile.createVariable("composite_energy", "f4", ("month", "lat", "lon"))
        shf_var = wfile.createVariable("composite_Shf", "f4", ("month", "lat", "lon"))
        dhdt_var = wfile.createVariable("composite_Dhdt", "f4", ("month", "lat", "lon"))
        swabs_var = wfile.createVariable("composite_Swabs", "f4", ("month", "lat", "lon"))
        olr_var = wfile.createVariable("composite_Olr", "f4", ("month", "lat", "lon"))
        z_var = wfile.createVariable("composite_Z", "f4", ("month", "lat", "lon"))
        t_var = wfile.createVariable("composite_T", "f4", ("month", "lat", "lon"))
        q_var = wfile.createVariable("composite_Q", "f4", ("month", "lat", "lon"))
        vo_var = wfile.createVariable("composite_VO", "f4", ("month", "lat", "lon"))
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
                z_var[month_idx, :, :] = composite_z[month_idx, :, :] / n
                t_var[month_idx, :, :] = composite_t[month_idx, :, :] / n
                q_var[month_idx, :, :] = composite_q[month_idx, :, :] / n
                # Interpolate VO from 20×20 to 120×120
                spline = scipy.interpolate.RectBivariateSpline(
                    x_vo, x_vo, composite_vo[month_idx, :, :], kx=1, ky=1,
                )
                vo_var[month_idx, :, :] = spline(y_vo, y_vo) / n

    _LOG.info("Saved W/m² composites: %s", output_path)
