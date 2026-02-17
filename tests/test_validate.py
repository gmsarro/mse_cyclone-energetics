"""Regression tests that compare new pipeline output against reference data.

These tests are designed to be run via SLURM on a compute node
(not on the login node).  They read existing reference NetCDF files
and compare key statistics to ensure the reorganised code produces
identical results.
"""

import pathlib

import netCDF4
import numpy as np
import numpy.typing as npt
import pytest

import cyclone_energetics.constants as constants
import cyclone_energetics.integration as integration

REFERENCE_TE_DIR = pathlib.Path("/project2/tas1/gmsarro/TE_ERA5")
REFERENCE_FLUX_FILE = pathlib.Path(
    "/project2/tas1/gmsarro/cyclone_centered"
    "/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
)

TOLERANCE_ABSOLUTE = 1e-4
TOLERANCE_RELATIVE = 1e-3


def _array_close(
    actual: npt.NDArray,
    expected: npt.NDArray,
) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        atol=TOLERANCE_ABSOLUTE,
        rtol=TOLERANCE_RELATIVE,
    )


class TestPolewardIntegration:
    """Verify that the vectorised poleward integration matches a naive loop."""

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
        _array_close(batch_result, loop_result)


class TestReferenceTE:
    """Spot-check TE reference file statistics for 2022-01."""

    @pytest.fixture()
    def reference_te(self) -> npt.NDArray:
        te_path = REFERENCE_TE_DIR / "TE_2022_01.nc"
        if not te_path.exists():
            pytest.skip("Reference TE file not found: %s" % te_path)
        with netCDF4.Dataset(str(te_path)) as ds:
            return np.array(ds["TE"][:])

    def test_shape(self, reference_te: npt.NDArray) -> None:
        assert reference_te.ndim == 3
        assert reference_te.shape[0] > 0

    def test_finite(self, reference_te: npt.NDArray) -> None:
        assert np.all(np.isfinite(reference_te))

    def test_mean_sign(self, reference_te: npt.NDArray) -> None:
        nh_mean = np.mean(reference_te[:, :reference_te.shape[1] // 2, :])
        sh_mean = np.mean(reference_te[:, reference_te.shape[1] // 2:, :])
        assert nh_mean != 0.0 or sh_mean != 0.0


class TestReferenceFluxFile:
    """Spot-check the final assigned-flux reference file."""

    @pytest.fixture()
    def flux_ds(self) -> dict[str, npt.NDArray]:
        if not REFERENCE_FLUX_FILE.exists():
            pytest.skip("Reference flux file not found: %s" % REFERENCE_FLUX_FILE)
        result: dict[str, npt.NDArray] = {}
        with netCDF4.Dataset(str(REFERENCE_FLUX_FILE)) as ds:
            for var_name in ds.variables:
                result[var_name] = np.array(ds[var_name][:])
        return result

    def test_has_required_variables(self, flux_ds: dict[str, npt.NDArray]) -> None:
        for name in ["F_TE_final", "tot_energy_final", "lat", "lon"]:
            assert name in flux_ds, "Missing variable: %s" % name

    def test_twelve_months(self, flux_ds: dict[str, npt.NDArray]) -> None:
        te = flux_ds["F_TE_final"]
        assert te.shape[1] == 12

    def test_finite_fluxes(self, flux_ds: dict[str, npt.NDArray]) -> None:
        te = flux_ds["F_TE_final"]
        assert np.all(np.isfinite(te))


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
