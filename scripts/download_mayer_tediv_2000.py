"""Download Mayer et al. (2021) mass-consistent TEDIV, monthly, year 2000.

Dataset: derived-reanalysis-energy-moisture-budget (CDS, DOI 10.24381/cds.c2451f6b)
"""
import cdsapi

c = cdsapi.Client()
c.retrieve(
    "derived-reanalysis-energy-moisture-budget",
    {
        "variable": ["divergence_of_vertical_integral_of_total_energy_flux"],
        "year": ["2000"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "data_format": "netcdf",
    },
    "/project2/tas1/gmsarro/mayer_budget/mayer_tediv_2000.nc",
)
print("download complete")
