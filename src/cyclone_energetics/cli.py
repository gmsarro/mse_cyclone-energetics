import logging
import pathlib
import typing

import typer
import typing_extensions

import cyclone_energetics.composites as composites
import cyclone_energetics.condensed_composites as condensed_composites
import cyclone_energetics.flux_assignment as flux_assignment
import cyclone_energetics.flux_computation as flux_computation
import cyclone_energetics.integration as integration
import cyclone_energetics.masking as masking
import cyclone_energetics.smoothing as smoothing
import cyclone_energetics.track_processing as track_processing

app = typer.Typer(help="Cyclone energetics processing pipeline.")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


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
    _setup_logging()
    print("Computing transient eddy fluxes from ERA5")
    flux_computation.compute_transient_eddy_flux(
        year_start=year_start,
        year_end=year_end,
        era5_base_directory=era5_base_directory,
        output_directory=output_directory,
    )
    print("Done.")


@app.command()
def smooth(
    input_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="ERA5 input directory")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for smoothed files")
    ],
    variables: typing_extensions.Annotated[
        typing.Optional[str], typer.Option(help="Comma-separated variable names")
    ] = "TE",
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2015,
    target_grid: typing_extensions.Annotated[
        str, typer.Option(help="CDO target grid specification")
    ] = "r360x180",
) -> None:
    _setup_logging()
    var_list = [v.strip() for v in (variables or "TE").split(",")]
    print("Smoothing ERA5 fields: %s" % var_list)
    smoothing.smooth_era5_fields(
        year_start=year_start,
        year_end=year_end,
        variables=var_list,
        input_directory=input_directory,
        output_directory=output_directory,
        target_grid=target_grid,
    )
    print("Done.")


@app.command()
def process_tracks(
    track_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory containing TRACK output .nc files")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for processed .npz files")
    ],
) -> None:
    _setup_logging()
    print("Processing cyclone track data")
    track_processing.process_track_data(
        track_directory=track_directory,
        output_directory=output_directory,
    )
    print("Done.")


@app.command()
def create_masks(
    hemisphere: typing_extensions.Annotated[
        str, typer.Option(help="Hemisphere: NH or SH")
    ],
    vorticity_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with vorticity data")
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
    ] = 2015,
    vorticity_threshold: typing_extensions.Annotated[
        float, typer.Option(help="Vorticity threshold in CVU")
    ] = 0.225,
) -> None:
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


@app.command()
def integrate_fluxes(
    te_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with TE flux files")
    ],
    dhdt_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with dh/dt files")
    ],
    vint_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with vertically integrated fields")
    ],
    radiation_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with radiation data")
    ],
    output_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Output directory for integrated fluxes")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2016,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2023,
) -> None:
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
    )
    print("Done.")


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


@app.command()
def build_composites(
    track_directory: typing_extensions.Annotated[
        pathlib.Path, typer.Option(help="Directory with track .nc files")
    ],
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
        pathlib.Path, typer.Option(help="Output directory for composites")
    ],
    year_start: typing_extensions.Annotated[
        int, typer.Option(help="Start year (inclusive)")
    ] = 2000,
    year_end: typing_extensions.Annotated[
        int, typer.Option(help="End year (exclusive)")
    ] = 2015,
    intensity_threshold: typing_extensions.Annotated[
        int, typer.Option(help="Minimum intensity for composites")
    ] = 6,
) -> None:
    _setup_logging()
    print("Building cyclone-centered composites")
    composites.build_cyclone_composites(
        year_start=year_start,
        year_end=year_end,
        hemispheres=["SH", "NH"],
        track_types=["TRACK", "ANTIC"],
        track_directory=track_directory,
        integrated_flux_directory=integrated_flux_directory,
        mask_directory_sh=mask_directory_sh,
        mask_directory_nh=mask_directory_nh,
        output_directory=output_directory,
        intensity_threshold=intensity_threshold,
    )
    print("Done.")


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
