"""Regression tests that compare new pipeline output against reference data.

These tests are designed to be run via SLURM on a compute node
(not on the login node).  They run the actual pipeline code on January
2022 input data and compare the output against existing reference
NetCDF files to ensure the reorganised code produces identical results.
"""

import pathlib
import shutil
import tempfile

import netCDF4
import numpy as np
import numpy.typing as npt
import pytest

import cyclone_energetics.constants as constants
import cyclone_energetics.flux_computation as flux_computation
import cyclone_energetics.integration as integration

ERA5_BASE = pathlib.Path(
    "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5"
)
REFERENCE_TE_DIR = pathlib.Path("/project2/tas1/gmsarro/TE_ERA5")
REFERENCE_INTFLUX_DIR = pathlib.Path(
    "/project2/tas1/gmsarro/cyclone_centered/Integrated_TE"
)
DHDT_DIR = pathlib.Path("/project2/tas1/gmsarro/smoothed_dh_dt_ERA5")
VINT_DIR = pathlib.Path("/project2/tas1/gmsarro/smoothed_vint")
RAD_DIR = pathlib.Path(
    "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/rad"
)

TEST_YEAR = 2022
TEST_MONTH = "01"

TOLERANCE_ABSOLUTE = 1e-2
TOLERANCE_RELATIVE = 1e-3


def _array_close(
    *,
    actual: npt.NDArray,
    expected: npt.NDArray,
    label: str,
    atol: float = TOLERANCE_ABSOLUTE,
    rtol: float = TOLERANCE_RELATIVE,
) -> None:
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    denom = np.maximum(np.abs(expected), 1e-30)
    max_rel = float(np.max(diff / denom))
    mean_abs = float(np.mean(diff))
    print(
        "  %s: max_abs=%.6e  mean_abs=%.6e  max_rel=%.6e"
        % (label, max_abs, mean_abs, max_rel)
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        err_msg="Mismatch for %s" % label,
    )


class TestPolewardIntegration:
    """Verify that the vectorised integration matches a naive per-column loop."""

    def test_1d_roundtrip(self) -> None:
        lat = np.linspace(-89.5, 89.5, 360)
        field = np.cos(np.deg2rad(lat))
        result = integration.poleward_integration(field, latitude=lat)
        assert result.shape[0] == lat.shape[0] - 2

    def test_batch_matches_loop(self) -> None:
        lat = np.linspace(-89.5, 89.5, 100)
        rng = np.random.default_rng(42)
        field_3d = rng.standard_normal((4, 100, 10))

        batch = integration._poleward_integrate_batch(
            fields={"test": field_3d}, latitude=lat
        )
        batch_result = batch["test"]

        loop_result = np.zeros((4, 98, 10))
        for t in range(4):
            for ll in range(10):
                loop_result[t, :, ll] = integration.poleward_integration(
                    field_3d[t, :, ll], latitude=lat
                )
        _array_close(
            actual=batch_result,
            expected=loop_result,
            label="batch_vs_loop",
            atol=1e-10,
            rtol=1e-10,
        )


class TestTEAgainstReference:
    """Run the new TE computation on Jan 2022 and compare to reference."""

    @pytest.fixture(scope="class")
    def te_output_dir(self) -> pathlib.Path:
        ref_te = REFERENCE_TE_DIR / ("TE_%d_%s.nc" % (TEST_YEAR, TEST_MONTH))
        t_input = ERA5_BASE / "t" / (
            "era5_t_%d_%s.6hrly.nc" % (TEST_YEAR, TEST_MONTH)
        )
        if not ref_te.exists():
            pytest.skip("Reference TE file not found: %s" % ref_te)
        if not t_input.exists():
            pytest.skip("ERA5 input not found: %s" % t_input)
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_te_"))
        flux_computation.compute_transient_eddy_flux(
            year_start=TEST_YEAR,
            year_end=TEST_YEAR + 1,
            era5_base_directory=ERA5_BASE,
            output_directory=tmpdir,
        )
        yield tmpdir
        shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_te_matches_reference_jan2022(
        self, te_output_dir: pathlib.Path
    ) -> None:
        new_path = te_output_dir / ("TE_%d_%s.nc" % (TEST_YEAR, TEST_MONTH))
        ref_path = REFERENCE_TE_DIR / ("TE_%d_%s.nc" % (TEST_YEAR, TEST_MONTH))

        assert new_path.exists(), "New TE file was not created"

        with netCDF4.Dataset(str(ref_path)) as ds_ref:
            te_ref = np.array(ds_ref["TE"][:])
            lat_ref = np.array(ds_ref["latitude"][:])
            lon_ref = np.array(ds_ref["longitude"][:])

        with netCDF4.Dataset(str(new_path)) as ds_new:
            te_new = np.array(ds_new["TE"][:])
            lat_new = np.array(ds_new["latitude"][:])
            lon_new = np.array(ds_new["longitude"][:])

        print("\nTE reference shape: %s, new shape: %s" % (te_ref.shape, te_new.shape))
        assert te_ref.shape == te_new.shape, (
            "Shape mismatch: ref=%s new=%s" % (te_ref.shape, te_new.shape)
        )

        _array_close(actual=lat_new, expected=lat_ref, label="latitude")
        _array_close(actual=lon_new, expected=lon_ref, label="longitude")
        _array_close(
            actual=te_new,
            expected=te_ref,
            label="TE_field",
            atol=1e-1,
            rtol=1e-3,
        )


class TestIntegrationAgainstReference:
    """Run the new integration on Jan 2022 and compare to reference."""

    @pytest.fixture(scope="class")
    def intflux_output_dir(self) -> pathlib.Path:
        ref_path = REFERENCE_INTFLUX_DIR / (
            "Integrated_Fluxes_%d_%s_.nc" % (TEST_YEAR, TEST_MONTH)
        )
        te_input = REFERENCE_TE_DIR / (
            "TE_%d_%s.nc" % (TEST_YEAR, TEST_MONTH)
        )
        dhdt_input = DHDT_DIR / (
            "tend_%d_%s_filtered_2.nc" % (TEST_YEAR, TEST_MONTH)
        )
        vint_input = VINT_DIR / (
            "era5_vint_%d_%s_filtered.nc" % (TEST_YEAR, TEST_MONTH)
        )
        rad_input = RAD_DIR / (
            "era5_rad_%d_%s.6hrly.nc" % (TEST_YEAR, TEST_MONTH)
        )

        for p in [ref_path, te_input, dhdt_input, vint_input, rad_input]:
            if not p.exists():
                pytest.skip("Required file not found: %s" % p)

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="test_intflux_"))
        integration.integrate_fluxes_poleward(
            year_start=TEST_YEAR,
            year_end=TEST_YEAR + 1,
            te_directory=REFERENCE_TE_DIR,
            dhdt_directory=DHDT_DIR,
            vint_directory=VINT_DIR,
            radiation_directory=RAD_DIR,
            output_directory=tmpdir,
        )
        yield tmpdir
        shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_integrated_fluxes_match_reference_jan2022(
        self, intflux_output_dir: pathlib.Path
    ) -> None:
        new_path = intflux_output_dir / (
            "Integrated_Fluxes_%d_%s_.nc" % (TEST_YEAR, TEST_MONTH)
        )
        ref_path = REFERENCE_INTFLUX_DIR / (
            "Integrated_Fluxes_%d_%s_.nc" % (TEST_YEAR, TEST_MONTH)
        )

        assert new_path.exists(), "New integrated flux file was not created"

        with netCDF4.Dataset(str(ref_path)) as ds_ref:
            te_ref = np.array(ds_ref["F_TE_final"][:])
            tot_ref = np.array(ds_ref["tot_energy_final"][:])
            shf_ref = np.array(ds_ref["F_Shf_final"][:])
            swabs_ref = np.array(ds_ref["F_Swabs_final"][:])
            olr_ref = np.array(ds_ref["F_Olr_final"][:])
            lat_ref = np.array(ds_ref["lat"][:])

        with netCDF4.Dataset(str(new_path)) as ds_new:
            te_new = np.array(ds_new["F_TE_final"][:])
            tot_new = np.array(ds_new["tot_energy_final"][:])
            shf_new = np.array(ds_new["F_Shf_final"][:])
            swabs_new = np.array(ds_new["F_Swabs_final"][:])
            olr_new = np.array(ds_new["F_Olr_final"][:])
            lat_new = np.array(ds_new["lat"][:])

        print(
            "\nIntFlux ref shape: %s, new shape: %s"
            % (te_ref.shape, te_new.shape)
        )
        assert te_ref.shape == te_new.shape, (
            "Shape mismatch: ref=%s new=%s" % (te_ref.shape, te_new.shape)
        )

        _array_close(actual=lat_new, expected=lat_ref, label="lat")
        _array_close(actual=te_new, expected=te_ref, label="F_TE_final")
        _array_close(actual=tot_new, expected=tot_ref, label="tot_energy_final")
        _array_close(actual=shf_new, expected=shf_ref, label="F_Shf_final")
        _array_close(actual=swabs_new, expected=swabs_ref, label="F_Swabs_final")
        _array_close(actual=olr_new, expected=olr_ref, label="F_Olr_final")


class TestConstants:
    """Verify physical constants are sensible."""

    def test_earth_radius(self) -> None:
        assert 6.0e6 < constants.EARTH_RADIUS < 7.0e6

    def test_gravity(self) -> None:
        assert 9.7 < constants.GRAVITY < 10.0

    def test_month_boundaries(self) -> None:
        assert constants.MONTH_BOUNDARIES[0] == 0
        assert constants.MONTH_BOUNDARIES[-1] == constants.TIMESTEPS_PER_YEAR
        assert len(constants.MONTH_BOUNDARIES) == 13
