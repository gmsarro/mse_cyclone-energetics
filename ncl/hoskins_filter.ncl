; hoskins_filter.ncl — Generic Hoskins spectral filter for ERA5 fields
;
; Applies triangular truncation and a Hoskins damping filter in spectral
; space using NCL's built-in spherical-harmonic routines (shaeC / shseC).
;
; All parameters are passed via environment variables so that the Python
; driver (smoothing.py) can control every aspect of the run without
; editing this file.
;
; Required environment variables:
;   SMOOTH_INPUT_FILE   — full path to the input NetCDF file
;   SMOOTH_OUTPUT_FILE  — full path to the output NetCDF file
;   SMOOTH_VARNAMES     — comma-separated variable names to filter
;   SMOOTH_SUFFIXED     — comma-separated output variable names (with _filtered suffix)
;   SMOOTH_TLOW         — low truncation wavenumber  (default 0)
;   SMOOTH_THIGH        — high truncation wavenumber (default 100)
;   SMOOTH_N0           — Hoskins filter damping wavenumber (default 60)
;   SMOOTH_R            — Hoskins filter exponent          (default 1)
;
; Developer: Chanil Park (Seoul National University), adapted by G. M. Sarro
; Reference: Hoskins, B. & Hodges, K. I. (2019), J. Climate, 32, 1743–1760.

begin

; ================================================================
; Read parameters from environment
; ================================================================
input_file  = getenv("SMOOTH_INPUT_FILE")
output_file = getenv("SMOOTH_OUTPUT_FILE")
varnames_str = getenv("SMOOTH_VARNAMES")
suffixed_str = getenv("SMOOTH_SUFFIXED")

Tlow_str  = getenv("SMOOTH_TLOW")
Thigh_str = getenv("SMOOTH_THIGH")
n0_str    = getenv("SMOOTH_N0")
r_str     = getenv("SMOOTH_R")

; Defaults
if (ismissing(Tlow_str)) then
    Tlow = 0
else
    Tlow = toint(Tlow_str)
end if
if (ismissing(Thigh_str)) then
    Thigh = 100
else
    Thigh = toint(Thigh_str)
end if
if (ismissing(n0_str)) then
    n0 = 60
else
    n0 = toint(n0_str)
end if
if (ismissing(r_str)) then
    r = 1
else
    r = toint(r_str)
end if

; ================================================================
; Build Hoskins filter coefficients
; ================================================================
N = 721
nn = fspan(0, N-1, N)
filter_coeff = exp(-1. * ((nn*(nn+1))/(n0*(n0+1)))^r)

; ================================================================
; Parse variable name lists
; ================================================================
varnames = str_split(varnames_str, ",")
suffixed = str_split(suffixed_str, ",")
nvars = dimsizes(varnames)

; ================================================================
; Open input file
; ================================================================
f_in = addfile(input_file, "r")

; ================================================================
; Remove existing output file and create new one
; ================================================================
system("rm -f " + output_file)
setfileoption("nc", "Format", "NetCDF4Classic")
f_out = addfile(output_file, "c")

; ================================================================
; Process each variable
; ================================================================
do iv = 0, nvars-1
    vn = varnames(iv)
    vn_out = suffixed(iv)

    print("Processing variable: " + vn + " -> " + vn_out)

    ; Read with latitude ascending (required by shaeC)
    var = f_in->$vn$(:,{-90:90},:)

    ; Spherical harmonic analysis
    spec = shaeC(var)
    print("  Spherical harmonics computed")

    ; Triangular truncation + Hoskins filter
    spec_high = tri_trunC(spec, Thigh)
    spec_low  = tri_trunC(spec, Tlow - 1)
    spec_band = spec_high - spec_low

    do t = Tlow, Thigh
        spec_band(:,:,t,:) = spec_band(:,:,t,:) * filter_coeff(t)
    end do

    ; Inverse transform back to physical domain
    nlon_var = dimsizes(var&longitude)
    var_trunc = shseC(spec_band, nlon_var)
    print("  Returned to physical domain")

    ; Copy coordinates and attributes
    copy_VarCoords(var, var_trunc)
    copy_VarAtts(var, var_trunc)

    ; Write to output
    f_out->$vn_out$ = var_trunc

    ; Clean up
    delete(var)
    delete(spec)
    delete(spec_high)
    delete(spec_low)
    delete(spec_band)
    delete(var_trunc)

end do

; ================================================================
; Copy coordinate variables that are not already written
; ================================================================
; Try to copy valid_time if present
if (isfilevar(f_in, "valid_time")) then
    vt = f_in->valid_time
    vt32 = toint(vt)
    copy_VarAtts(vt, vt32)
    filedimdef(f_out, "valid_time", dimsizes(vt), False)
    f_out->valid_time = vt32
    delete(vt)
    delete(vt32)
end if

delete(f_in)
delete(f_out)

print("Done: " + output_file)

end
