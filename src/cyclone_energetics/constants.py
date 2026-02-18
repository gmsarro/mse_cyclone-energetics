import typing

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

# Monthly storm-track latitude (Jan–Dec) for the latitude-band filter used
# in cyclone-centred compositing.  Values are the monthly-mean latitude of
# the TE peak from the 2000–2014 analysis.
STORM_LAT_SH: np.ndarray = np.array([
    -45.39914645, -46.33174147, -46.37381343, -45.16073870,
    -43.76535216, -42.89586507, -43.20439275, -43.73730419,
    -43.30957264, -43.05714090, -43.84248408, -45.04854682,
], dtype=float)

# NH counterpart — monthly-mean latitude of the TE peak.
STORM_LAT_NH: np.ndarray = np.array([
    45.0, 45.5, 44.5, 43.0, 42.0, 41.5,
    41.0, 41.5, 42.0, 43.0, 44.0, 45.0,
], dtype=float)

STORM_LAT_BAND_HALF_WIDTH: float = 5.0

# No-leap month lengths and cumulative boundaries (used by the AI composites)
NOLEAP_MONTH_LENGTHS: np.ndarray = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int
)
NOLEAP_MONTH_CUMULATIVE: np.ndarray = np.concatenate(
    [[0], np.cumsum(NOLEAP_MONTH_LENGTHS)]
)

# Hoskins spectral filter parameters (for smoothing)
HOSKINS_N0: int = 60
HOSKINS_R: int = 1
HOSKINS_NTRUNC: int = 100
