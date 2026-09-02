"""Single-snapshot identity check for the stored Integrated_Fluxes files.

For each stored transport F_X (OLR, SWABS, tot_energy, Dhdt, Shf), recover the
implied integrand S'_implied = dF/dphi * 1e15 / (2 pi a^2 cos) on the stored
(descending) latitude grid and correlate it against the directly computed
integrand candidate S'_direct = X - mbar(lon), where mbar is the cos-weighted
meridional mean. Each source field is tested in both latitude orientations to
detect any flip inconsistency in the production pipeline.
"""

import netCDF4
import numpy as np

A = 6.371e6
LV = 2.5008e6
T = 10  # 2005-07-03 18:00

fint = netCDF4.Dataset(
    "/project2/tas1/gmsarro/cyclone_centered/Integrated_TE/Integrated_Fluxes_2005_07_.nc")
fnew = netCDF4.Dataset(
    "/project2/tas1/gmsarro/cyclone_centered/Integrated_TE/New_Integrated_Fluxes_2005_07_.nc")
lat = np.array(fint["lat"][:], dtype=np.float64)  # 719, descending 89.75..-89.75
phi = np.deg2rad(lat)
cosl = np.cos(phi)

fvint = netCDF4.Dataset(
    "/project2/tas1/gmsarro/smoothed_vint/era5_vint_2005_07_filtered.nc")
ftend = netCDF4.Dataset(
    "/project2/tas1/gmsarro/smoothed_dh_dt_ERA5/tend_2005_07_filtered.nc")
frad = netCDF4.Dataset(
    "/project2/tas1/abacus/data1/tas/archive/Reanalysis/ERA5/rad/era5_rad_2005_07.6hrly.nc")

rad_lat = np.array(frad["latitude"][:])
print("rad lat[0], lat[-1]:", rad_lat[0], rad_lat[-1])

tsr = np.nan_to_num(np.array(frad["tsr"][T]) / 3600.0)
ssr = np.nan_to_num(np.array(frad["ssr"][T]) / 3600.0)
ttr = np.nan_to_num(np.array(frad["ttr"][T]) / 3600.0)
if rad_lat[0] < rad_lat[-1]:
    tsr, ssr, ttr = tsr[::-1], ssr[::-1], ttr[::-1]
olr_desc = ttr[1:-1]
swabs_desc = (tsr - ssr)[1:-1]

vigd = np.array(fvint["p85.162_filtered"][T])
vimdf = np.array(fvint["p84.162_filtered"][T])
vithed = np.array(fvint["p83.162_filtered"][T])
tot_stored = vigd + vimdf * LV + vithed
tend_stored = np.nan_to_num(np.array(ftend["tend_filtered"][T]))


def implied(Fvar, src):
    F = np.array(src[Fvar][T], dtype=np.float64) * 1e15
    dF = np.gradient(F, axis=0) / np.gradient(phi)[:, None]
    return dF / (2 * np.pi * A**2 * cosl[:, None])


def direct(field_desc):
    mbar = np.average(field_desc, weights=cosl, axis=0)
    return field_desc - mbar[None, :]


def corr(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


tests = [
    ("F_Olr_final", fint, olr_desc, None),
    ("F_Swabs_final", fint, swabs_desc, None),
    ("tot_energy_final", fint, tot_stored[1:-1], tot_stored[::-1][1:-1]),
    ("F_Dhdt_final", fnew, tend_stored[1:-1], tend_stored[::-1][1:-1]),
]
for name, src, cand_asis, cand_flip in tests:
    si = implied(name, src)
    inner = np.abs(lat) < 88  # avoid pole metric noise
    r_asis = corr(si[inner], direct(cand_asis)[inner])
    msg = f"{name:18s} r(as-stored)={r_asis:+.3f}"
    if cand_flip is not None:
        r_flip = corr(si[inner], direct(cand_flip)[inner])
        msg += f"  r(flipped)={r_flip:+.3f}"
    print(msg)

# SHF: both orientation hypotheses for the vint+tend part
shf_asis = tot_stored[1:-1] - swabs_desc - olr_desc + tend_stored[1:-1]
shf_flip = (tot_stored[::-1][1:-1] - swabs_desc - olr_desc
            + tend_stored[::-1][1:-1])
si = implied("F_Shf_final", fint)
inner = np.abs(lat) < 88
print(f"F_Shf_final        r(vint as-stored)={corr(si[inner], direct(shf_asis)[inner]):+.3f}"
      f"  r(vint flipped)={corr(si[inner], direct(shf_flip)[inner]):+.3f}")

# magnitude check at a sample SH midlat point for the winning hypothesis
j = np.argmin(np.abs(lat + 50.0))
k = 720
print("\nsample at lat=-50, lon=180:")
print(f"  S'_implied(SHF)      = {si[j, k]:+8.1f} W/m2")
print(f"  S'_direct (as-stored)= {direct(shf_asis)[j, k]:+8.1f} W/m2")
print(f"  S'_direct (flipped)  = {direct(shf_flip)[j, k]:+8.1f} W/m2")
print(f"  raw SHF residual (as-stored) = {shf_asis[j, k]:+8.1f} W/m2")
print(f"  mbar(lon=180) as-stored = {np.average(shf_asis[:, k], weights=cosl):+8.1f}")

for f in (fint, fnew, fvint, ftend, frad):
    f.close()
