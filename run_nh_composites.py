#!/usr/bin/env python3
"""
Generate NH composite figures directly from the noleap files.
SHF is the energy-balance residual (composite_Shf_wm), NOT from SLHF.
All variables on the cyclone-centered (y, x) grid.
"""

import os
import string
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.transforms as _mt
import numpy as np
import scipy.interpolate
import xarray
from netCDF4 import Dataset as NC_P

# ---------------------------------------------------------------------------
# Bbox safety patch
# ---------------------------------------------------------------------------
_orig_union = _mt.Bbox.union
def _safe_union(bboxes):
    if not len(bboxes):
        return _mt.Bbox.unit()
    return _orig_union(bboxes)
_mt.Bbox.union = staticmethod(_safe_union)

mpl.rcParams.update({
    "font.size": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/project2/tas1/gmsarro"
OUT_DIR = os.environ.get("NH_OUT_DIR", ".")
os.makedirs(OUT_DIR, exist_ok=True)

NC_FLUX = os.path.join(BASE, "cyclone_centered", "WITH_INT_Cyclones_Sampled_Poleward_Fluxes_0.225.nc")
NH_INT = os.path.join(BASE, "cyclone_centered", "Composites_Intense_NH_noleap.nc")
NH_WK = os.path.join(BASE, "cyclone_centered", "Composites_Weak_NH_noleap.nc")

for p in [NC_FLUX, NH_INT, NH_WK]:
    assert os.path.exists(p), f"Missing: {p}"

# ---------------------------------------------------------------------------
# Build NH dataset from noleap files
# ---------------------------------------------------------------------------
print("Loading noleap files...", flush=True)
_ds_int = xarray.open_dataset(NH_INT)
_ds_wk = xarray.open_dataset(NH_WK)

_a_earth = 6.371e6

with NC_P(NC_FLUX) as _f:
    _lat_st = _f["lat"][:]
    _fte = _f["F_TE_final"][0, :, :, :]

_fte_zon = np.mean(_fte, axis=2)
_nlat_st = len(_lat_st)
_xd = np.linspace(0, _nlat_st - 1, _nlat_st)
_x3d = np.linspace(0, _nlat_st - 1, 25600)
_Lat_hi = scipy.interpolate.interp1d(_xd, _lat_st)(_x3d)
_yd = np.linspace(0, 12, 12)
_fte_int = scipy.interpolate.RectBivariateSpline(_yd, _xd, _fte_zon)(_yd, _x3d)
_st_nh = _Lat_hi[np.argmax(_fte_int, axis=1)]
print(f"NH storm-track lats: {_st_nh}", flush=True)

_y_grid = _ds_int["y"].values
_nx = _ds_int["x"].shape[0]


def _build_weight(y_base, maxpos12, nx):
    w = np.empty((12, len(y_base), nx), np.float64)
    for i in range(12):
        lat2d = np.tile((y_base + maxpos12[i])[:, None], (1, nx))
        w[i] = _a_earth * np.cos(np.deg2rad(lat2d)) * 2.0 * np.pi * 1e-15
    return w


_wgt = _build_weight(_y_grid, _st_nh, _nx)

_I_L = np.stack([
    _ds_int["composite_TE"].values / _wgt,
    _ds_wk["composite_TE"].values / _wgt,
], axis=0).astype(np.float32)

_I_SHF = np.stack([
    _ds_int["composite_Shf"].values / _wgt,
    _ds_wk["composite_Shf"].values / _wgt,
], axis=0).astype(np.float32)

_SHF = np.stack([
    _ds_int["composite_Shf_wm"].values,
    _ds_wk["composite_Shf_wm"].values,
], axis=0).astype(np.float32)

_VO = np.stack([
    _ds_int["composite_VO"].values,
    _ds_wk["composite_VO"].values,
], axis=0).astype(np.float32)

_cnt = np.stack([
    _ds_int["count"].values.astype(np.float64),
    _ds_wk["count"].values.astype(np.float64),
], axis=0)

ds_nh = xarray.Dataset(
    data_vars=dict(
        I_L=(("category", "month", "y", "x"), _I_L),
        I_SHF=(("category", "month", "y", "x"), _I_SHF),
        SHF=(("category", "month", "y", "x"), _SHF),
        VO=(("category", "month", "y", "x"), _VO),
        count_track=(("category", "month"), _cnt),
    ),
    coords=dict(
        category=(("category",), np.arange(2, dtype=np.int32)),
        month=(("month",), np.arange(1, 13, dtype=np.int32)),
        y=(("y",), _y_grid),
        x=(("x",), _ds_int["x"].values),
    ),
)
_ds_int.close()
_ds_wk.close()
print("NH dataset built (all on y,x grid, residual SHF).", flush=True)

# ---------------------------------------------------------------------------
# NH helpers
# ---------------------------------------------------------------------------
PW_factor_NH = 2 * np.pi * _a_earth / 1e15

colors_nh = ["#4B0082", "#3F51B5", "#2196F3", "dodgerblue", "skyblue",
             "#FFFFFF", "#FFFFFF",
             "lightpink", "#FF9800", "#F44336", "#B71C1C", "maroon"]
custom_cmap_nh = matplotlib.colors.LinearSegmentedColormap.from_list(
    "PurpleBlue_White_OrangeRed_NH", colors_nh, N=256)


def _nh_annual_mean(variable, C=0, factor=1):
    return (factor * ds_nh[variable].sel(category=C)).mean(dim="month")


def _nh_seasonality_total(variable, C=0, factor=1):
    DJF, JJA = [12, 1, 2], [6, 7, 8]
    y_djf = (factor * ds_nh[variable].sel(category=C).sel(month=DJF)).mean(dim="month")
    y_jja = (factor * ds_nh[variable].sel(category=C).sel(month=JJA)).mean(dim="month")
    w_djf = ds_nh.count_track.sel(category=C).sel(month=DJF).mean(dim="month")
    w_jja = ds_nh.count_track.sel(category=C).sel(month=JJA).mean(dim="month")
    w_ann = ds_nh.count_track.sel(category=C).mean(dim="month")
    return (y_djf * w_djf - y_jja * w_jja) / w_ann


def _nh_constant_weight(variable, C=0, factor=1):
    DJF, JJA = [12, 1, 2], [6, 7, 8]
    y_djf = (factor * ds_nh[variable].sel(category=C).sel(month=DJF)).mean(dim="month")
    y_jja = (factor * ds_nh[variable].sel(category=C).sel(month=JJA)).mean(dim="month")
    return y_djf - y_jja


def _nh_constant_flux(variable, C=0, factor=1):
    DJF, JJA = [12, 1, 2], [6, 7, 8]
    y_ann = (factor * ds_nh[variable].sel(category=C)).mean(dim="month")
    w_djf = ds_nh.count_track.sel(category=C).sel(month=DJF).mean(dim="month")
    w_jja = ds_nh.count_track.sel(category=C).sel(month=JJA).mean(dim="month")
    w_ann = ds_nh.count_track.sel(category=C).mean(dim="month")
    return (w_djf - w_jja) * y_ann / w_ann


# ---------------------------------------------------------------------------
# Figure 1: NH both-category seasonality
# ---------------------------------------------------------------------------
def plot_2d_composites_nh_both_category(factor=PW_factor_NH):
    variables = ['I_L', 'I_SHF']
    titles = {
        'I_L': r'${\Delta I^{L}}$',
        'I_SHF': r'$\Delta {I^{L}}_{SHF}$',
    }
    categories = {
        1: {'level': np.arange(-12, 13, 1) / 2, 'title': 'Weak cyclones'},
        0: {'level': np.arange(-12, 13, 1) / 1, 'title': 'Strong cyclones'},
    }

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 5, width_ratios=[0.05, 0.1, 1, 1, 0.7],
                          hspace=0.35, wspace=0.15)
    axes = np.empty((2, 5), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[0, 2] = fig.add_subplot(gs[0, 2])
    axes[0, 3] = fig.add_subplot(gs[0, 3], sharex=axes[0, 2], sharey=axes[0, 2])
    axes[0, 4] = fig.add_subplot(gs[0, 4], sharey=axes[0, 2])
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    axes[1, 2] = fig.add_subplot(gs[1, 2], sharex=axes[0, 2], sharey=axes[0, 2])
    axes[1, 3] = fig.add_subplot(gs[1, 3], sharex=axes[0, 3], sharey=axes[0, 3])
    axes[1, 4] = fig.add_subplot(gs[1, 4], sharey=axes[0, 2])

    conlev = 3
    for row, C in enumerate(categories.keys()):
        last_cf = None
        for col, variable in enumerate(variables):
            field_total = _nh_seasonality_total(variable, C=C, factor=factor)
            cf = axes[row, col + 2].contourf(
                ds_nh['x'], ds_nh['y'], field_total,
                levels=categories[C]['level'], extend='both', cmap=custom_cmap_nh)
            cc = axes[row, col + 2].contour(
                ds_nh['x'], ds_nh['y'],
                ds_nh.VO.sel(category=C).mean(dim='month'),
                levels=[1], colors='purple', linewidths=1, linestyles='solid')
            axes[row, col + 2].clabel(cc, fmt={1: '1 CVU'})
            axes[row, col + 2].set_title(titles[variable], fontsize=16)

            field_ann = _nh_annual_mean(variable, C=C, factor=factor)
            c = axes[row, col + 2].contour(
                ds_nh['x'], ds_nh['y'], field_ann,
                conlev, colors='k', linewidths=0.3)
            axes[row, col + 2].clabel(c, fmt='%1d')
            last_cf = cf
            if row == 0:
                axes[row, col + 2].tick_params(labelbottom=False)
            if col > 0:
                axes[row, col + 2].tick_params(labelleft=False)

        for variable, color in zip(variables, ['red', 'dodgerblue']):
            zf_total = _nh_seasonality_total(variable, C=C, factor=factor).mean(dim='x')
            zf_cw = _nh_constant_weight(variable, C=C, factor=factor).mean(dim='x')
            zf_cf = _nh_constant_flux(variable, C=C, factor=factor).mean(dim='x')
            axes[row, 4].plot(zf_total, ds_nh['y'], color=color, lw=4,
                              label=titles[variable], alpha=0.7)
            axes[row, 4].plot(zf_cw, ds_nh['y'], color=color, lw=1.5, ls='--',
                              label=titles[variable] + r'${\  (local\ flux)}$')
            axes[row, 4].plot(zf_cf, ds_nh['y'], color=color, lw=1.5, ls='dotted',
                              label=titles[variable] + r'${\  (footprint)}$')
            if variable == 'I_L':
                axes[row, 4].plot(zf_total, ds_nh['y'],
                                  color='white', lw=0.01, label='  ', alpha=0.0)
            axes[row, 4].set_xlim(-10, 10)

        axes[row, 4].axvline(0, color='k', linewidth=1.5)
        axes[row, 4].set_title('Cyclone centred\nzonal mean', fontsize=12)
        if row == 0:
            axes[row, 4].legend(fontsize=10, bbox_to_anchor=(1, 0.3), frameon=True)
            axes[row, 4].tick_params(labelbottom=False)
        axes[row, 2].set_ylabel('rlat', labelpad=-5)
        axes[row, 4].tick_params(labelleft=False)

        cbar = fig.colorbar(last_cf, cax=axes[row, 0])
        cbar.set_label('PW', fontsize=12)
        axes[row, 0].yaxis.set_ticks_position('left')
        axes[row, 0].tick_params(labelleft=True)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.tick_left()
        axes[row, 1].axis('off')

    axes[1, 2].set_xlabel('rlon')
    axes[1, 3].set_xlabel('rlon')
    axes[1, 4].set_xlabel('PW')
    fig.text(0.6, 0.97, 'Weak Cyclones (NH)', ha='center', va='top', fontsize=16)
    fig.text(0.6, 0.52, 'Strong Cyclones (NH)', ha='center', va='top', fontsize=16)

    labels = list(string.ascii_lowercase)
    AXES = axes[:, 2:]
    for i, ax in enumerate(AXES.flat):
        ax.text(0.02, 0.98, f'({labels[i]})', transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='top', ha='left')

    fig.suptitle('2D composite of cyclone seasonality (NH)',
                 fontsize=16, y=1.05, x=0.55, fontweight='bold')
    fig.savefig(os.path.join(OUT_DIR, '2D_composite_of_cyclone_seasonality_NH.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, '2D_composite_of_cyclone_seasonality_NH.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("Saved 2D_composite_of_cyclone_seasonality_NH", flush=True)


# ---------------------------------------------------------------------------
# Figure 2 & 3: NH decomposition (strong + weak)
# ---------------------------------------------------------------------------
def plot_2d_composites_nh_decomposition_all(
    C=0, factor=PW_factor_NH,
    levels=np.arange(-12, 13, 1), levels2=10,
):
    variables = ['I_L', 'I_SHF', 'SHF']
    titles = {
        'I_L': r'${\Delta I^{L}}$',
        'I_SHF': r'$\Delta {I^{L}}_{SHF}$',
        'SHF': r'$\Delta {SHF}$',
    }

    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5), sharex=True, sharey=True,
                             constrained_layout=True)
    conlev = 3
    row_cf = []

    for row, variable in enumerate(variables):
        cur_levels = levels2 if variable == 'SHF' else levels
        cur_factor = 1 if variable == 'SHF' else factor

        for col, (decomp_func, col_title) in enumerate([
            (_nh_seasonality_total, 'total seasonality'),
            (_nh_constant_weight, 'flux change'),
            (_nh_constant_flux, 'footprint change'),
        ]):
            field = decomp_func(variable, C=C, factor=cur_factor)
            cf = axes[row, col].contourf(
                ds_nh['x'], ds_nh['y'], field,
                levels=cur_levels, extend='both', cmap=custom_cmap_nh)
            field_ann = _nh_annual_mean(variable, C=C, factor=cur_factor)
            c = axes[row, col].contour(
                ds_nh['x'], ds_nh['y'], field_ann,
                conlev, colors='k', linewidths=0.5)
            axes[row, col].clabel(c, fmt='%1d')
            cc = axes[row, col].contour(
                ds_nh['x'], ds_nh['y'],
                ds_nh.VO.sel(category=C).mean(dim='month'),
                levels=[1], colors='purple', linewidths=1)
            axes[row, col].clabel(cc, fmt={1: '1 CVU'})
            if col == 0:
                axes[row, col].set_title(titles[variable] + '\n' + col_title)
            else:
                axes[row, col].set_title(titles[variable] + '\n(' + col_title + ')')
        row_cf.append(cf)
        axes[row, 0].set_ylabel('rlat')

    for c in range(3):
        axes[2, c].set_xlabel('rlon')
    for row in range(3):
        label = 'PW' if row < 2 else r'$W/m^2$'
        fig.colorbar(row_cf[row], ax=axes[row, :],
                     orientation='vertical', shrink=1, pad=0.02).set_label(label)

    strength = {0: 'strong cyclones', 1: 'weak cyclones'}
    fig.suptitle(f'2D composites {strength[C]} (NH)', y=1.02, fontweight='bold')

    labels = list(string.ascii_lowercase)
    for i, ax in enumerate(axes.flat):
        ax.text(0.02, 0.98, f'({labels[i]})', transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='top', ha='left')

    fig.savefig(
        os.path.join(OUT_DIR,
                     f'2D_composites_{strength[C].replace(" ", "_")}_all_decompose_NH.pdf'),
        bbox_inches='tight')
    fig.savefig(
        os.path.join(OUT_DIR,
                     f'2D_composites_{strength[C].replace(" ", "_")}_all_decompose_NH.png'),
        dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 2D_composites_{strength[C].replace(' ', '_')}_all_decompose_NH", flush=True)


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------
print("\n=== Generating NH both-category figure ===", flush=True)
plot_2d_composites_nh_both_category()

print("\n=== Generating NH strong-cyclone decomposition ===", flush=True)
plot_2d_composites_nh_decomposition_all(
    C=0, factor=PW_factor_NH,
    levels=np.arange(-12, 13, 1),
    levels2=np.arange(-180, 200, 20),
)

print("\n=== Generating NH weak-cyclone decomposition ===", flush=True)
plot_2d_composites_nh_decomposition_all(
    C=1, factor=PW_factor_NH,
    levels=np.arange(-12, 13, 1) / 2,
    levels2=np.arange(-180, 200, 20),
)

print("\nAll NH composite figures generated successfully.", flush=True)
