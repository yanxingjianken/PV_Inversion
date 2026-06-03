"""
wu_nonlinear/config.py — Standalone config for Wu (Fortran SOR) pipeline at 2.5deg.
For INLIN=1 / CNLIN=0.5 nonlinear piecewise PV inversion.
Grid: 2.5deg x 2.5deg, NY=31, NX=52 (lat 85.5->10.5N, lon -169.5->-42.0W).
"""

import numpy as np
from datetime import date

# =============================================================================
# 1. HORIZONTAL GRID -- 2.5 degree (coarse, for INLIN=1 stability)
# =============================================================================
NX, NY = 52, 31
DLAT, DLON = 2.5, 2.5
LAT_N, LAT_S = 85.5, 10.5
LON_W, LON_E = -169.5, -42.0

# =============================================================================
# 2. VERTICAL LEVELS (sigma = p / 1000 hPa)
# =============================================================================
PR = np.array([1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
NW = len(PR)
PLEVS = PR * 1000.0
PLEVS_PA = PLEVS * 100.0
NW_PV = NW - 2

# =============================================================================
# 3. EVENT DATE
# =============================================================================
EVENT_YEAR, EVENT_MONTH, EVENT_DAY = 2025, 1, 8
EVENT_DATE = date(EVENT_YEAR, EVENT_MONTH, EVENT_DAY)

# =============================================================================
# 4. CLIMATOLOGY
# =============================================================================
CLIM_WINDOW_DAYS = 30

# =============================================================================
# 5. PV PIECES
# =============================================================================
PIECES = [
    {"name": "lower",  "levels": (1, 4),   "hPa": "1000-700"},
    {"name": "middle", "levels": (5, 6),   "hPa": "600-500"},
    {"name": "upper",  "levels": (7, 10),  "hPa": "400-200"},
]
NPIECES = len(PIECES)

# =============================================================================
# 6. SOR SOLVER PARAMETERS -- INLIN=1 nonlinear, CNLIN=0.5 hardcoded in Fortran
# =============================================================================
OMEGS     = 1.85
OMEGH     = 1.75
PART      = 0.8
THRSH     = 0.1
TSCAL     = 1.0
QSCAL     = 1.0
INLIN     = 1         # nonlinear (pieces sum to total per Davis 1991)
IQD       = 0         # no external PV dependency
IBC       = 0         # homogeneous Dirichlet BC on lateral walls
MAX_ITER  = 400       # max SOR iterations per Poisson solve
MAXT      = 400       # max outer coupling cycles

# =============================================================================
# 7. PLOTTING DEFAULTS
# =============================================================================
QUIVER_SKIP = 4
REF_SPEED   = 20.0
WIND_DISPLAY_CAP = 40.0
GAUSSIAN_SIGMA = 1.5

# =============================================================================
# 8. PATHS (derived -- auto-resolve into wu_nonlinear/)
# =============================================================================
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ERA5_DIR = os.path.join(DATA_DIR, "era5")
CLIM_DIR = os.path.join(DATA_DIR, "clim")
WU_IN_DIR  = os.path.join(DATA_DIR, "wu_in")
WU_OUT_DIR = os.path.join(DATA_DIR, "wu_out")
FIG_DIR    = os.path.join(DATA_DIR, "figs")
FORT_DIR   = os.path.join(ROOT_DIR, "fortran")
STEPS_DIR  = os.path.join(ROOT_DIR, "steps")

from datetime import timedelta
HALF = CLIM_WINDOW_DAYS // 2
CLIM_START = EVENT_DATE - timedelta(days=HALF)
CLIM_END   = EVENT_DATE + timedelta(days=HALF)
CLIM_DATES = [CLIM_START + timedelta(days=d) for d in range(CLIM_WINDOW_DAYS)]

