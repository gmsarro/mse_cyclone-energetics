#!/usr/bin/env python3
"""Numerical identity validation for refactored computation modules.

Computes one month of TE and dh/dt with the refactored code and compares
against reference output produced by the original code.  Exits with
code 0 if the maximum absolute difference is within tolerance, 1 otherwise.

Usage (from the repo root):
    python tests/validate_numerics.py \
        --data-directory /project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5 \
        --reference-te-directory /project2/tas1/gmsarro/TE_dry_ERA5 \
        --reference-dhdt-directory /project2/tas1/gmsarro/dh_dt_data_ERA5 \
        --output-directory /scratch/midway2/gmsarro/validation_output \
        --year 2000 --month 01
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import netCDF4
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
_LOG = logging.getLogger(__name__)


def _compare(
    *,
    new_path: pathlib.Path,
    ref_path: pathlib.Path,
    variable: str,
    tolerance: float,
) -> bool:
    with netCDF4.Dataset(str(new_path)) as ds_new, \
         netCDF4.Dataset(str(ref_path)) as ds_ref:
        new_vals = np.asarray(ds_new[variable][:], dtype=np.float64)
        ref_vals = np.asarray(ds_ref[variable][:], dtype=np.float64)

    if new_vals.shape != ref_vals.shape:
        _LOG.error(
            "  Shape mismatch: new=%s ref=%s", new_vals.shape, ref_vals.shape,
        )
        return False

    max_abs_diff = float(np.nanmax(np.abs(new_vals - ref_vals)))
    max_rel_diff = float(
        np.nanmax(np.abs(new_vals - ref_vals) / (np.abs(ref_vals) + 1e-30))
    )
    _LOG.info("  max |diff|     = %e", max_abs_diff)
    _LOG.info("  max |rel diff| = %e", max_rel_diff)

    passed = max_abs_diff <= tolerance
    if passed:
        _LOG.info("  PASS (tolerance=%e)", tolerance)
    else:
        _LOG.error("  FAIL (tolerance=%e)", tolerance)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerical validation")
    parser.add_argument("--data-directory", type=pathlib.Path, required=True)
    parser.add_argument("--reference-te-directory", type=pathlib.Path, required=True)
    parser.add_argument("--reference-dhdt-directory", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--month", type=str, default="01")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    all_passed = True

    import cyclone_energetics.flux_computation as flux_computation
    import cyclone_energetics.storage_computation as storage_computation

    te_out_dir = args.output_directory / "te"
    te_out_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("Computing TE for %d/%s with refactored code...", args.year, args.month)
    flux_computation.compute_transient_eddy_flux(
        year_start=args.year,
        year_end=args.year + 1,
        data_directory=args.data_directory,
        output_directory=te_out_dir,
    )

    te_new = te_out_dir / ("TE_%d_%s.nc" % (args.year, args.month))
    te_ref = args.reference_te_directory / ("TE_%d_%s.nc" % (args.year, args.month))

    _LOG.info("Comparing TE: new=%s vs ref=%s", te_new, te_ref)
    if te_ref.exists():
        passed = _compare(
            new_path=te_new, ref_path=te_ref,
            variable="TE", tolerance=args.tolerance,
        )
        if not passed:
            all_passed = False
    else:
        _LOG.warning("Reference TE file not found: %s", te_ref)

    dhdt_out_dir = args.output_directory / "dhdt"
    dhdt_out_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("Computing dh/dt for %d/%s with refactored code...", args.year, args.month)
    storage_computation.compute_storage_term(
        year_start=args.year,
        year_end=args.year + 1,
        data_directory=args.data_directory,
        output_directory=dhdt_out_dir,
    )

    dhdt_new = dhdt_out_dir / ("tend_%d_%s_2.nc" % (args.year, args.month))
    dhdt_ref = args.reference_dhdt_directory / ("tend_%d_%s_2.nc" % (args.year, args.month))

    _LOG.info("Comparing dh/dt: new=%s vs ref=%s", dhdt_new, dhdt_ref)
    if dhdt_ref.exists():
        passed = _compare(
            new_path=dhdt_new, ref_path=dhdt_ref,
            variable="tend", tolerance=args.tolerance,
        )
        if not passed:
            all_passed = False
    else:
        _LOG.warning("Reference dh/dt file not found: %s", dhdt_ref)

    if all_passed:
        _LOG.info("ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        _LOG.error("SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
