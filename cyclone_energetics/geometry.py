from __future__ import annotations

import numpy as np
import numpy.typing as npt


def lonlat_to_polar(
    *,
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    hemisphere: str,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if hemisphere == "NH":
        polar_x = (90.0 - latitude) * np.cos(longitude * np.pi / 180.0)
        polar_y = (90.0 - latitude) * np.sin(longitude * np.pi / 180.0)
    elif hemisphere == "SH":
        polar_x = (90.0 + latitude) * np.cos(longitude * np.pi / 180.0)
        polar_y = (90.0 + latitude) * np.sin(longitude * np.pi / 180.0)
    else:
        raise ValueError("hemisphere must be 'NH' or 'SH', got %s" % hemisphere)
    return polar_x, polar_y


def build_polar_mesh(
    *,
    grid_spacing: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    ys = np.arange(-90, 90 + grid_spacing, grid_spacing)
    xs = np.arange(-90, 90 + grid_spacing, grid_spacing)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def spherical_distance(
    *,
    lon0: float,
    lat0: float,
    lons: npt.NDArray[np.floating],
    lats: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    re = 6.37e6
    conv = np.pi / 180.0
    lon0r = lon0 * conv
    lat0r = lat0 * conv
    lonr = lons * conv
    latr = lats * conv
    distance = 2.0 * re * np.arcsin(
        np.sqrt(
            np.sin((latr - lat0r) / 2.0) ** 2
            + np.cos(latr) * np.cos(lat0r) * np.sin((lonr - lon0r) / 2.0) ** 2
        )
    )
    return distance


def polar_mesh_to_km_distance(
    *,
    x0: float,
    y0: float,
    grid_points: npt.NDArray[np.floating],
    hemisphere: str,
) -> npt.NDArray[np.floating]:
    ll_grids = np.zeros(np.shape(grid_points))
    if hemisphere == "NH":
        lat = 90.0 - np.sqrt(x0**2 + y0**2)
        ll_grids[:, 1] = 90.0 - np.sqrt(
            grid_points[:, 0] ** 2 + grid_points[:, 1] ** 2
        )
    elif hemisphere == "SH":
        lat = -90.0 + np.sqrt(x0**2 + y0**2)
        ll_grids[:, 1] = -90.0 + np.sqrt(
            grid_points[:, 0] ** 2 + grid_points[:, 1] ** 2
        )
    else:
        raise ValueError("hemisphere must be 'NH' or 'SH', got %s" % hemisphere)
    lon = np.arctan2(y0, x0) * 180.0 / np.pi
    ll_grids[:, 0] = np.arctan2(grid_points[:, 1], grid_points[:, 0]) * 180.0 / np.pi
    dists = spherical_distance(
        lon0=lon, lat0=lat, lons=ll_grids[:, 0], lats=ll_grids[:, 1]
    )
    return dists
