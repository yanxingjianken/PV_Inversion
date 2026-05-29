"""
config.py — Central configuration for the Wu/Davis PPVI pipeline.

Change any value here and re-run the pipeline for a new case.
All step scripts import from this module.

Reference: Davis et al. (2022, J. Climate), Wu/Davis piecewise PV inversion.
"""

import numpy as np
from datetime import date

# ═══════════════════════════════════════════════════════════════════════════════
# 1. HORIZONTAL GRID
# ═══════════════════════════════════════════════════════════════════════════════
# Fortran PARAMETER blocks must match NX, NY exactly.
# Domain: lat_N → lat_S (north to south in Fortran array I=1..NY)
#         lon_W → lon_E (west to east in Fortran array J=1..NX)
NX, NY = 87, 51
DLAT, DLON = 1.5, 1.5
LAT_N, LAT_S = 85.5, 10.5
LON_W, LON_E = -169.5, -40.5

# ═══════════════════════════════════════════════════════════════════════════════
# 2. VERTICAL LEVELS (σ = p / 1000 hPa)
# ═══════════════════════════════════════════════════════════════════════════════
# PR[0] = lower boundary (θ fixed), PR[-1] = upper boundary (θ fixed).
# CRITICAL: top 3 levels MUST have uniform Δσ spacing (0.05)
#           or the SOR solver at K=NL-1 will diverge exponentially.
PR = np.array([1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
NW = len(PR)
PLEVS = PR * 1000.0    # pressure [hPa]
PLEVS_PA = PLEVS * 100.0  # pressure [Pa]
NW_PV = NW - 2          # interior PV levels (925–250 hPa)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVENT DATE
# ═══════════════════════════════════════════════════════════════════════════════
EVENT_YEAR, EVENT_MONTH, EVENT_DAY = 2025, 1, 8
EVENT_DATE = date(EVENT_YEAR, EVENT_MONTH, EVENT_DAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLIMATOLOGY — symmetric running-mean window
# ═══════════════════════════════════════════════════════════════════════════════
# Number of total days for the time-mean (must be odd to center on event).
# The window is symmetric: event_date ± CLIM_WINDOW_DAYS//2.
# Talia tutorial: ≥10 days. Default = 30 days for robust synoptic filtering.
CLIM_WINDOW_DAYS = 30

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PV PIECES (vertical partitioning for piecewise inversion)
# ═══════════════════════════════════════════════════════════════════════════════
# Each piece = tuple of (start_K, end_K) where K is 1-indexed Fortran level.
# K=1 → 1000 hPa, K=2 → 925 hPa, …, K=10 → 200 hPa.
PIECES = [
    {"name": "lower",  "levels": (1, 2),   "hPa": "1000–925"},
    {"name": "middle", "levels": (3, 4),   "hPa": "850–700"},
    {"name": "upper",  "levels": (5, 9),   "hPa": "600–250"},
]
NPIECES = len(PIECES)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SOR SOLVER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
OMEGS = 1.4       # SOR relaxation for streamfunction ψ
OMEGH = 1.4       # SOR relaxation for geopotential Φ
PART  = 0.5       # under-relaxation between ψ and Φ updates
THRSH = 0.01      # convergence threshold
TSCAL = 1.0       # boundary θ scale factor
QSCAL = 1.0       # PV scale factor
INLIN = 0         # 0 = linear balance (stable), 1 = nonlinear (may explode)
IQD   = 0         # 0 = no external PV dependency
IBC   = 0         # 0 = homogeneous Dirichlet BC on lateral walls
MAX_ITER = 5000   # max SOR iterations per Poisson solve
MAXT     = 500    # max outer coupling cycles

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PLOTTING DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
QUIVER_SKIP = 4        # plot every Nth grid point
REF_SPEED   = 20.0     # m/s for reference arrow
WIND_DISPLAY_CAP = 40.0  # m/s — cap displayed wind magnitude
GAUSSIAN_SIGMA = 1.5   # σ for piece-3 smoothing

# ═══════════════════════════════════════════════════════════════════════════════
# 8. PATHS (derived — do not edit)
# ═══════════════════════════════════════════════════════════════════════════════
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

# Compute derived date range for climatology
from datetime import timedelta
HALF = CLIM_WINDOW_DAYS // 2
CLIM_START = EVENT_DATE - timedelta(days=HALF)
CLIM_END   = EVENT_DATE + timedelta(days=HALF)
CLIM_DATES = [CLIM_START + timedelta(days=d) for d in range(CLIM_WINDOW_DAYS)]
