"""Hoskins-filter the corrected storage term (tend_*_3.nc) with the SAME
spectral filter that was applied to the ERA5 energy-flux divergence fields
(smoothed_vint/*_filtered.nc), so that the residual surface flux
SHF = div - RAD + storage is consistently filtered. NCL shaeC/shseC engine via
cyclone_energetics.smoothing.hoskins (reproduces the production files exactly).

Output: smoothed_dh_dt_ERA5_v3/tend_YYYY_MM_filtered_3.nc, variable 'tend_filtered',
ascending latitude, like the production tend_*_filtered.nc.

Usage: python filter_storage_v3.py YEAR [MONTH ...]   (needs `ncl` in PATH)
Filter parameters: environment variables HOSKINS_NTRUNC, HOSKINS_N0, HOSKINS_R
(defaults below, set to the values that reproduce the production vint files).
"""

import logging
import os
import pathlib
import sys

sys.path.insert(0, "/project2/tas1/gmsarro/mse_cyclone-energetics")
import cyclone_energetics.constants as constants                    # noqa: E402
from cyclone_energetics.smoothing.hoskins import hoskins_spectral_smooth  # noqa: E402

IN_DIR = pathlib.Path("/project2/tas1/gmsarro/dh_dt_data_ERA5_v3")
OUT_DIR = pathlib.Path("/project2/tas1/gmsarro/smoothed_dh_dt_ERA5_v3")
NTRUNC = int(os.environ.get("HOSKINS_NTRUNC", 100))
N0 = int(os.environ.get("HOSKINS_N0", 27))
R = int(os.environ.get("HOSKINS_R", 1))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

if __name__ == "__main__":
    year = int(sys.argv[1])
    months = sys.argv[2:] or constants.MONTH_STRINGS
    logging.info("Hoskins filter parameters: T=%d n0=%d r=%d", NTRUNC, N0, R)
    for m in months:
        mm = "%02d" % int(m)
        src = IN_DIR / ("tend_%d_%s_3.nc" % (year, mm))
        dst = OUT_DIR / ("tend_%d_%s_filtered_3.nc" % (year, mm))
        if dst.exists():
            logging.info("exists, skipping %s", dst)
            continue
        if not src.exists():
            raise FileNotFoundError(src)
        hoskins_spectral_smooth(input_path=src, output_path=dst, variable_names=["tend"],
                                ntrunc=NTRUNC, n0=N0, r=R)
        logging.info("filtered %s", dst)
