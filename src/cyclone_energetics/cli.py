from __future__ import annotations

import logging
import pathlib

import numpy as np
import typer
import typing_extensions

import cyclone_energetics.composites as composites
import cyclone_energetics.composites_wm as composites_wm
import cyclone_energetics.condensed_composites as condensed_composites
import cyclone_energetics.constants as constants
import cyclone_energetics.flux_assignment as flux_assignment
import cyclone_energetics.flux_computation as flux_computation
import cyclone_energetics.integration as integration
import cyclone_energetics.masking as masking
import cyclone_energetics.smoothing as smoothing
import cyclone_energetics.storage_computation as storage_computation
import cyclone_energetics.track_processing as track_processing
import cyclone_energetics.zonal_advection as zonal_advection

app = typer.Typer(help="Cyclone energetics processing pipeline.")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


# -----------------------------------------------------------------------
# Step 1a — transient-eddy flux divergence
# -----------------------------------------------------------------------
@app.command()
def compute_te(
    era5_base_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Base ERA5 data directory")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for TE flux files")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
) -> None:
    """Step 1a: Compute transient-eddy (meridional) MSE flux divergence."""
    _setup_logging()
    print("Computing transient eddy fluxes from ERA5")
    flux_computation.compute_transient_eddy_flux(
        year_start=year_start,
        year_end=year_end,
        era5_base_directory=era5_base_directory,
        output_directory=output_directory,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 1b — MSE storage term (dh/dt)
# -----------------------------------------------------------------------
@app.command()
def compute_dhdt(
    era5_base_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Base ERA5 data directory")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for dh/dt files")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
) -> None:
    """Step 1b: Compute the MSE storage term dh/dt."""
    _setup_logging()
    print("Computing MSE storage term (dh/dt)")
    storage_computation.compute_storage_term(
        year_start=year_start,
        year_end=year_end,
        era5_base_directory=era5_base_directory,
        output_directory=output_directory,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 1c — zonal MSE advection divergence
# -----------------------------------------------------------------------
@app.command()
def compute_zonal_mse(
    era5_base_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Base ERA5 data directory")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for zonal advection files")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
) -> None:
    """Step 1c: Compute the zonal MSE advection divergence."""
    _setup_logging()
    print("Computing zonal MSE advection divergence")
    zonal_advection.compute_zonal_advection(
        year_start=year_start,
        year_end=year_end,
        era5_base_directory=era5_base_directory,
        output_directory=output_directory,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 2 — Hoskins spectral filter (drives NCL)
# -----------------------------------------------------------------------
@app.command()
def smooth_hoskins(
    dhdt_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with raw dh/dt files")
    ],
    vint_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with raw ERA5 vint files")
    ],
    output_dhdt_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for filtered dh/dt files")
    ],
    output_vint_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for filtered vint files")
    ],
    adv_mse_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with raw advective MSE files (Adv_YYYY_MM.nc)")
    ] = None,
    output_adv_mse_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for filtered advective MSE files")
    ] = None,
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
    ntrunc: typing_extensions.Annotated[
        int, typer.Option(help="Maximum total wavenumber for Hoskins filter")
    ] = constants.HOSKINS_NTRUNC,
) -> None:
    """Step 2: Apply the Hoskins spectral filter to dh/dt, vint, and advective MSE fields.

    The transient-eddy (TE) divergence is NOT smoothed; the monthly
    anomaly product is already sufficiently smooth.  Only dh/dt, the
    ERA5 vertically-integrated energy terms (vigd, vimdf, vithed), and
    the advective MSE (u_mse, v_mse) are filtered before meridional
    integration.
    """
    _setup_logging()
    print("Applying Hoskins spectral filter to dh/dt, vint, and advective MSE fields")
    smoothing.smooth_all_pipeline_fields(
        year_start=year_start,
        year_end=year_end,
        dhdt_directory=dhdt_directory,
        vint_directory=vint_directory,
        output_dhdt_directory=output_dhdt_directory,
        output_vint_directory=output_vint_directory,
        adv_mse_directory=adv_mse_directory,
        output_adv_mse_directory=output_adv_mse_directory,
        ntrunc=ntrunc,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 3 — process raw TRACK output
# -----------------------------------------------------------------------
@app.command()
def process_tracks(
    track_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory containing TRACK output .nc files")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for processed .npz files")
    ],
) -> None:
    """Step 3: Process raw TRACK algorithm output into .npz arrays."""
    _setup_logging()
    print("Processing cyclone track data")
    track_processing.process_track_data(
        track_directory=track_directory,
        output_directory=output_directory,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 4 — cyclone / anticyclone masks
# -----------------------------------------------------------------------
@app.command()
def create_masks(
    hemisphere: typing_extensions.Annotated[
        str, typer.Option(help="Hemisphere: NH or SH")
    ],
    vorticity_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with filtered vorticity data")
    ],
    track_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with processed track .npz files")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for mask files")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
    vorticity_threshold: typing_extensions.Annotated[
        float, typer.Option(help="Vorticity threshold in CVU")
    ] = 0.225,
) -> None:
    """Step 4: Build cyclone/anticyclone masks from vorticity contours."""
    _setup_logging()
    print("Creating cyclone masks for hemisphere=%s" % hemisphere)
    masking.create_cyclone_masks(
        hemisphere=hemisphere,
        year_start=year_start,
        year_end=year_end,
        vorticity_directory=vorticity_directory,
        track_directory=track_directory,
        output_directory=output_directory,
        vorticity_threshold=vorticity_threshold,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 5 — poleward flux integration
# -----------------------------------------------------------------------
@app.command()
def integrate_fluxes(
    te_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with raw TE divergence files (not smoothed)")
    ],
    dhdt_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with Hoskins-filtered dh/dt files")
    ],
    vint_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with Hoskins-filtered vint fields")
    ],
    radiation_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with radiation data")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for integrated fluxes")
    ],
    adv_mse_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with Hoskins-filtered advective MSE files (Adv_YYYY_MM_filtered.nc)")
    ] = None,
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
) -> None:
    """Step 5: Poleward-integrate energy flux fields."""
    _setup_logging()
    print("Integrating fluxes poleward")
    integration.integrate_fluxes_poleward(
        year_start=year_start,
        year_end=year_end,
        te_directory=te_directory,
        dhdt_directory=dhdt_directory,
        vint_directory=vint_directory,
        radiation_directory=radiation_directory,
        output_directory=output_directory,
        adv_mse_directory=adv_mse_directory,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 6 — assign fluxes to cyclone/anticyclone categories
# -----------------------------------------------------------------------
@app.command()
def assign_fluxes(
    integrated_flux_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with integrated flux files")
    ],
    mask_directory_nh: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="NH mask directory")
    ],
    mask_directory_sh: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="SH mask directory")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
    area_cut: typing_extensions.Annotated[
        str, typer.Option(help="Area cut threshold")
    ] = "0.225",
) -> None:
    """Step 6: Assign integrated fluxes to cyclone/anticyclone categories."""
    _setup_logging()
    print("Assigning fluxes to cyclones and anticyclones")
    flux_assignment.assign_fluxes_with_intensity(
        year_start=year_start,
        year_end=year_end,
        integrated_flux_directory=integrated_flux_directory,
        mask_directory_nh=mask_directory_nh,
        mask_directory_sh=mask_directory_sh,
        output_directory=output_directory,
        area_cut=area_cut,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 7a — cyclone-centred composites (integrated fluxes)
# -----------------------------------------------------------------------
@app.command()
def build_composites(
    track_path: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Track .nc file")
    ],
    integrated_flux_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with integrated flux files")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for composites")
    ],
    hemisphere: typing_extensions.Annotated[
        str, typer.Option(help="Hemisphere: SH or NH")
    ] = "SH",
    intensity_min: typing_extensions.Annotated[
        int, typer.Option(help="Minimum intensity (inclusive)")
    ] = 1,
    intensity_max: typing_extensions.Annotated[
        int, typer.Option(help="Maximum intensity (inclusive)")
    ] = 5,
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2015,
) -> None:
    """Step 7a: Build cyclone-centred composites of integrated budget terms."""
    _setup_logging()
    storm_lat = (
        constants.STORM_LAT_SH if hemisphere == "SH"
        else constants.STORM_LAT_NH
    )
    print(
        "Building composites: %s intensity=[%s,%s]"
        % (hemisphere, intensity_min, intensity_max)
    )
    composites.build_cyclone_composites(
        year_start=year_start,
        year_end=year_end,
        hemisphere=hemisphere,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        track_path=track_path,
        integrated_flux_directory=integrated_flux_directory,
        output_directory=output_directory,
        storm_lat=storm_lat,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 7b — cyclone-centred composites (W/m² budget terms)
# -----------------------------------------------------------------------
@app.command()
def build_wm_composites(
    track_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with track .nc files")
    ],
    integrated_flux_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with integrated flux files")
    ],
    vint_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with smoothed vint files")
    ],
    dhdt_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with smoothed dh/dt files")
    ],
    radiation_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with radiation data")
    ],
    z_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with geopotential height data")
    ],
    t2m_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with 2-m temperature data")
    ],
    q_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with specific humidity data")
    ],
    vorticity_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with filtered T42 vorticity files")
    ],
    mask_directory_nh: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="NH mask directory")
    ],
    mask_directory_sh: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="SH mask directory")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for W/m² composites")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2015,
    intensity_threshold: typing_extensions.Annotated[
        int, typer.Option(help="Minimum track intensity (0=all, 6=intense only)")
    ] = 0,
) -> None:
    """Step 7b: Build cyclone-centred composites of local (W/m²) budget terms."""
    _setup_logging()
    print("Building W/m² cyclone-centred composites")
    composites_wm.build_wm_composites(
        year_start=year_start,
        year_end=year_end,
        hemispheres=["SH", "NH"],
        track_types=["TRACK", "ANTIC"],
        track_directory=track_directory,
        integrated_flux_directory=integrated_flux_directory,
        vint_directory=vint_directory,
        dhdt_directory=dhdt_directory,
        radiation_directory=radiation_directory,
        z_directory=z_directory,
        t2m_directory=t2m_directory,
        q_directory=q_directory,
        vorticity_directory=vorticity_directory,
        mask_directory_sh=mask_directory_sh,
        mask_directory_nh=mask_directory_nh,
        output_directory=output_directory,
        intensity_threshold=intensity_threshold,
    )
    print("Done.")


# -----------------------------------------------------------------------
# Step 8 — condensed monthly composites
# -----------------------------------------------------------------------
@app.command()
def condense_composites(
    intense_composite: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Intense composites .nc file")
    ],
    regular_composite: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Regular composites .nc file")
    ],
    intense_full: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Intense full-field composites .nc file")
    ],
    all_full: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="All full-field composites .nc file")
    ],
    stormtrack: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Stormtrack sampled fluxes .nc file")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory")
    ],
) -> None:
    """Step 8: Create a single condensed monthly composite file."""
    _setup_logging()
    print("Creating condensed monthly composites")
    output_path = output_directory / "Composites_monthly_condensed.nc"
    output_directory.mkdir(parents=True, exist_ok=True)
    condensed_composites.create_condensed_composites(
        intense_composite_path=intense_composite,
        regular_composite_path=regular_composite,
        intense_full_path=intense_full,
        all_full_path=all_full,
        stormtrack_path=stormtrack,
        output_path=output_path,
    )
    print("Done.")


if __name__ == "__main__":
    app()
