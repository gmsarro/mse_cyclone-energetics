#!/usr/bin/env python3
"""Run TE computation for one year (no typer). For cluster job to verify memory-safe code."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cyclone_energetics.computation.flux as flux_computation

flux_computation.compute_transient_eddy_flux(
    year_start=2000,
    year_end=2001,
    data_directory=pathlib.Path("/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"),
    output_directory=pathlib.Path("/scratch/midway2/gmsarro/te_check"),
)
print("TE run finished successfully.")
