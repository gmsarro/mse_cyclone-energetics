from __future__ import annotations

import logging
import os
import pathlib
import subprocess

import netCDF4

import cyclone_energetics.constants as constants

_LOG = logging.getLogger(__name__)

_NCL_SCRIPT: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent.parent / "ncl" / "hoskins_filter.ncl"
)


def hoskins_spectral_smooth(
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    variable_names: list[str],
    output_variable_names: list[str] | None = None,
    ntrunc: int = constants.HOSKINS_NTRUNC,
    n0: int = constants.HOSKINS_N0,
    r: int = constants.HOSKINS_R,
    ncl_script: pathlib.Path | None = None,
) -> None:
    if ncl_script is None:
        ncl_script = _NCL_SCRIPT

    if not ncl_script.exists():
        raise FileNotFoundError(
            "NCL script not found: %s" % ncl_script
        )

    if output_variable_names is None:
        output_variable_names = ["%s_filtered" % v for v in variable_names]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["SMOOTH_INPUT_FILE"] = str(input_path)
    env["SMOOTH_OUTPUT_FILE"] = str(output_path)
    env["SMOOTH_VARNAMES"] = ",".join(variable_names)
    env["SMOOTH_SUFFIXED"] = ",".join(output_variable_names)
    env["SMOOTH_TLOW"] = "0"
    env["SMOOTH_THIGH"] = str(ntrunc)
    env["SMOOTH_N0"] = str(n0)
    env["SMOOTH_R"] = str(r)

    cmd = ["ncl", str(ncl_script)]
    _LOG.info(
        "Running NCL Hoskins filter: %s → %s  vars=%s  T=%s  n0=%s  r=%s",
        input_path.name,
        output_path.name,
        variable_names,
        ntrunc,
        n0,
        r,
    )

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env
    )

    if result.returncode != 0:
        _LOG.error(
            "NCL Hoskins filter failed for %s:\nSTDOUT:\n%s\nSTDERR:\n%s",
            input_path,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(
            "NCL Hoskins filter failed for %s (exit %d)"
            % (input_path, result.returncode)
        )

    _LOG.info("Saved Hoskins-filtered file: %s", output_path)


def _detect_vint_variable_names(
    vint_path: pathlib.Path,
) -> tuple[list[str], list[str]]:
    with netCDF4.Dataset(str(vint_path)) as ds:
        keys = list(ds.variables.keys())

    if "vigd" in keys:
        return (
            ["vigd", "vimdf", "vithed"],
            ["vigd_filtered", "vimdf_filtered", "vithed_filtered"],
        )
    if "p85.162" in keys:
        return (
            ["p85.162", "p84.162", "p83.162"],
            ["p85.162_filtered", "p84.162_filtered", "p83.162_filtered"],
        )
    raise KeyError(
        "Cannot detect vint variable names in %s. Available: %s"
        % (vint_path, keys)
    )


def smooth_all_pipeline_fields(
    *,
    year_start: int,
    year_end: int,
    dhdt_directory: pathlib.Path,
    vint_directory: pathlib.Path,
    output_dhdt_directory: pathlib.Path,
    output_vint_directory: pathlib.Path,
    adv_mse_directory: pathlib.Path | None = None,
    output_adv_mse_directory: pathlib.Path | None = None,
    ntrunc: int = constants.HOSKINS_NTRUNC,
) -> None:
    output_dhdt_directory.mkdir(parents=True, exist_ok=True)
    output_vint_directory.mkdir(parents=True, exist_ok=True)
    if output_adv_mse_directory is not None:
        output_adv_mse_directory.mkdir(parents=True, exist_ok=True)

    for year in range(year_start, year_end):
        for month in constants.MONTH_STRINGS:
            dhdt_in = dhdt_directory / ("tend_%d_%s_2.nc" % (year, month))
            dhdt_out = output_dhdt_directory / ("tend_%d_%s_filtered_2.nc" % (year, month))
            if dhdt_in.exists():
                hoskins_spectral_smooth(
                    input_path=dhdt_in,
                    output_path=dhdt_out,
                    variable_names=["tend"],
                    output_variable_names=["tend_filtered"],
                    ntrunc=ntrunc,
                )

            vint_in = vint_directory / ("era5_vint_%d_%s.6hrly.nc" % (year, month))
            vint_out = output_vint_directory / (
                "era5_vint_%d_%s_filtered.nc" % (year, month)
            )
            if vint_in.exists():
                vint_varnames, vint_outnames = _detect_vint_variable_names(vint_in)
                hoskins_spectral_smooth(
                    input_path=vint_in,
                    output_path=vint_out,
                    variable_names=vint_varnames,
                    output_variable_names=vint_outnames,
                    ntrunc=ntrunc,
                )

            if adv_mse_directory is not None and output_adv_mse_directory is not None:
                adv_in = adv_mse_directory / ("Adv_%d_%s.nc" % (year, month))
                adv_out = output_adv_mse_directory / (
                    "Adv_%d_%s_filtered.nc" % (year, month)
                )
                if adv_in.exists():
                    hoskins_spectral_smooth(
                        input_path=adv_in,
                        output_path=adv_out,
                        variable_names=["v_mse", "u_mse"],
                        output_variable_names=["v_mse_filtered", "u_mse_filtered"],
                        ntrunc=ntrunc,
                    )
