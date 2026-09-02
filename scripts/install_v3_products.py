"""Install the corrected-storage (v3) sampled products under the canonical file names.

WITH_INT: the stored file is renamed to *_storage_v2_archive.nc and the v3 copy
(all variables identical except F_Dhdt_final* / F_Shf_final*) takes its name.
WATTS:    the stored file is copied to *_storage_v2_archive.nc, then its
F_Dhdt_final* and F_Shf_final* variables are overwritten from WATTS_v3 (the
other W/m2 variables are storage independent and stay as they are).

Idempotent: skips steps whose archive already exists.
"""
import os
import shutil
import time

import netCDF4
import numpy as np

CC = "/project2/tas1/gmsarro/cyclone_centered"
WITH_INT = f"{CC}/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
WITH_INT_V3 = f"{CC}/WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225_v3.nc"
WATTS = f"{CC}/WATTS_Cyclones_Sampled_Poleward_Fluxes_0.225.nc"
WATTS_V3 = f"{CC}/WATTS_v3_Cyclones_Sampled_SHF_0.225.nc"
VARS = [f"{b}{s}" for b in ("F_Dhdt_final", "F_Shf_final") for s in ("", "_cycl", "_ant")]


def archive_name(path):
    return path.replace(".nc", "_storage_v2_archive.nc")


def main():
    arc = archive_name(WITH_INT)
    if not os.path.exists(arc):
        with netCDF4.Dataset(WITH_INT_V3) as ds:
            assert "corrected storage term" in getattr(ds, "history", ""), "v3 WITH_INT lacks history stamp"
        os.rename(WITH_INT, arc)
        os.rename(WITH_INT_V3, WITH_INT)
        print("WITH_INT: archived old ->", arc, "; installed v3 as", WITH_INT, flush=True)
    else:
        print("WITH_INT already installed", flush=True)

    arc = archive_name(WATTS)
    if not os.path.exists(arc):
        shutil.copy2(WATTS, arc)
        print("WATTS: archived copy ->", arc, flush=True)
        with netCDF4.Dataset(WATTS_V3) as src, netCDF4.Dataset(WATTS, "a") as dst:
            assert np.allclose(src["lat"][:], dst["lat"][:]) and np.allclose(src["lon"][:], dst["lon"][:])
            for v in VARS:
                old = np.asarray(dst[v][:])
                new = np.asarray(src[v][:])
                assert old.shape == new.shape, (v, old.shape, new.shape)
                d = new - old
                print(f"  {v:20s} rms old {np.sqrt(np.nanmean(old**2)):.3f}  rms diff {np.sqrt(np.nanmean(d**2)):.3f} W/m2", flush=True)
                dst[v][:] = new
            dst.history = (getattr(dst, "history", "") + f" | {time.strftime('%Y-%m-%d')}: F_Dhdt_final* and "
                           "F_Shf_final* replaced with the corrected storage term (resample_with_int_v3.py)")
        print("WATTS: installed v3 storage/SHF variables", flush=True)
    else:
        print("WATTS already installed", flush=True)


if __name__ == "__main__":
    main()
