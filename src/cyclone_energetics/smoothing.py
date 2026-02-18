"""Smooth / regrid ERA5 fields using CDO (Climate Data Operators).

CDO performs bilinear interpolation on the native grid with high
throughput and minimal memory overhead, making it the most efficient
choice for regridding large 6-hourly ERA5 datasets.
"""

import logging
import pathlib
import subprocess

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)


def smooth_era5_fields(
    *,
    year_start: int,
    year_end: int,
    variables: list[str],
    input_directory: pathlib.Path,
    output_directory: pathlib.Path,
    target_grid: str = "r360x180",
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for variable in variables:
        for year in range(year_start, year_end):
            for month in constants.MONTH_STRINGS:
                _LOG.info(
                    "Smoothing variable=%s year=%s month=%s",
                    variable,
                    year,
                    month,
                )
                infile = (
                    input_directory
                    / variable
                    / ("era5_%s_%d_%s.6hrly.nc" % (variable, year, month))
                )
                outfile = (
                    output_directory
                    / variable
                    / (
                        "smoothed_era5_%s_%d_%s.6hrly.nc"
                        % (variable, year, month)
                    )
                )
                outfile.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "cdo",
                    "remapbil,%s" % target_grid,
                    str(infile),
                    str(outfile),
                ]
                _LOG.info("Running CDO command: %s", " ".join(cmd))
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )
                if result.returncode != 0:
                    _LOG.error(
                        "CDO smoothing failed for %s: %s",
                        infile,
                        result.stderr,
                    )
                    raise RuntimeError(
                        "CDO smoothing failed for %s" % infile
                    )
