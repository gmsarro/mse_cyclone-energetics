import numpy as np

EARTH_RADIUS: float = 6.371e6
GRAVITY: float = 9.81
CPD: float = 1005.7
RD: float = 287.0
RV: float = 461.0
LATENT_HEAT_VAPORIZATION: float = 2.501e6
EPSILON: float = RD / RV

TIMESTEPS_PER_DAY: int = 4
DAYS_PER_MONTH: np.ndarray = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
)
TIMESTEPS_PER_MONTH: np.ndarray = DAYS_PER_MONTH * TIMESTEPS_PER_DAY
MONTH_BOUNDARIES: np.ndarray = np.array(
    [0, 124, 236, 360, 480, 604, 724, 848, 972, 1092, 1216, 1336, 1460]
)
TIMESTEPS_PER_YEAR: int = 1460

import typing

MONTH_STRINGS: typing.List[str] = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"
]
MONTH_NAMES: typing.List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

INTENSITY_CUTS: np.ndarray = np.array([1, 2, 3, 4, 5, 6])
VORTICITY_THRESHOLD: float = 0.225
DISTANCE_THRESHOLD: float = 1.5e6

PW_TO_WM2_FACTOR: float = 1e-15
