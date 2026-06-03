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
NX, NY = 134, 51
DLAT, DLON = 1.5, 1.5
LAT_N, LAT_S = 85.5, 10.5
LON_W, LON_E = 120.0, 320.0  # 120°E to 40°W (0-360 convention)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. VERTICAL LEVELS (σ = p / 1000 hPa)
# ═══════════════════════════════════════════════════════════════════════════════
# PR[0] = lower boundary (θ fixed), PR[-1] = upper boundary (θ fixed).
# CRITICAL: top 3 levels MUST have uniform Δσ spacing (0.05)
#           or the SOR solver at K=NL-1 will diverge exponentially.
PR = np.array([1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2])
NW = len(PR)
PLEVS = PR * 1000.0    # pressure [hPa]
PLEVS_PA = PLEVS * 100.0  # pressure [Pa]
NW_PV = NW - 2          # interior PV levels (850–250 hPa)

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
    {"name": "lower",  "levels": (1, 2),   "hPa": "1000–850"},
    {"name": "middle", "levels": (3, 5),   "hPa": "700–400"},
    {"name": "upper",  "levels": (6, 8),   "hPa": "300–200"},
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
INLIN = 1         # 1 = nonlinear (pieces sum to total per Davis 1991); 0 = linear
IQD   = 0         # 0 = no external PV dependency
IBC   = 0         # 0 = homogeneous Dirichlet BC on lateral walls; 1 = full perturbation BC
MAX_ITER = 8000   # max SOR iterations per Poisson solve (scaled for 51×134 grid)
MAXT     = 800    # max outer coupling cycles

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PHYSICAL CONSTANTS & UNIT CONVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
G = 9.81                 # gravity [m/s²]
PSI_SCALE = 1.0e5        # Wu ψ "per-1e5 m²/s" → m²/s multiplier
PV_SI_SCALE = 1.0e6      # ERA5 PV [SI] → PVU multiplier (1/this for PVU→SI)
                          # ERA5 CDS native PV is K·m²·kg⁻¹·s⁻¹ (~1e-6); PVU = 1e-6 SI
R_E = 6.371e6            # Earth radius [m]
OMEGA = 7.292e-5          # Earth rotation rate [s⁻¹]
RD = 287.0                # dry-air gas constant [J/(kg·K)]
CP = 1004.0               # dry-air specific heat [J/(kg·K)]
P0 = 1.0e5                # reference pressure [Pa]
KAP = RD / CP             # ≈ 2/7

# ── Wu PV → SI Ertel PV exact back-conversion (2026-06-03) ──────────────────
# Wu Q (pvpialln_94UV.f L460-468):
#   Q_wu = -COEF·Π⁻²·⁵·[(f+ζ)·∂θ/∂Π + (∂u/∂Π·∂θ/∂y − ∂v/∂Π·∂θ/∂x)]
#   COEF = 1e2·1e6·9.81·κ·Cp³·⁵/p0
# True SI Ertel PV = -g[(f+ζ)·∂θ/∂p + shear_p].
# Chain rule: ∂/∂p = (κΠ/p)·∂/∂Π → the SAME factor multiplies both terms,
# so a single per-level coefficient converts the whole scalar:
#   PV_SI = Q_wu · (g·κ/COEF)·Π³·⁵/p  =  Q_wu · (1/1e8)·(p/p0)^{3.5κ−1}
# With κ = 2/7, 3.5κ = 1 exactly → (p/p0)⁰ = 1 → c(p) = 1e-8 CONSTANT ∀ level.
# WU2SI = (G * KAP / COEF_WU) * PI_LEV^3.5 / PLEVS_PA  [per-level, ≈ 1e-8]
# Wu chose Π⁻²·⁵ weighting precisely so Π³·⁵/p cancels all pressure dependence.
# The residual Wu_Q/ERA5_PV ~3.5× beyond the true 1e8 is PHYSICAL
# (Wu coarse-FD vs ERA5 native PV discretization), not a units factor.
#
# Non-dimensionalisation (qinvert21_94.f L213-276, N_L levels):
#   DPI   = 500/N_L            ─ non-dim pressure increment
#   LL    = (2e7/π)·Δλ·π/180   ─ horizontal length scale [m]
#   FF    = 1e-4               ─ reference Coriolis [s⁻¹]
#   THO   = FF²·LL²/DPI        ─ potential temperature scale [K]
#   HND   = DPI·THO/G          ─ geopotential height scale [m]
#   PI_WU = Cp·(p/p0)^κ/DPI    ─ non-dim Exner (Δπ_wu ≈ 1)
#   PIF   = PI_WU^{2.5}         ─ PV-diminishing factor
#   QCONST = 1e6·κ·G·Cp·FF·THO/(p0·DPI)  ─ PV scale [equiv. PVU]
#   H_nd  = Φ/(THO·DPI)         ψ_nd = ψ·1e⁻⁵·G/(THO·DPI)
#   q_nd  = PIF·q/(1e2·QCONST)
# Note: N_L=8 → DPI=62.5 (NOT the old hardcoded 50 from N_L=10 examples).
# ──────────────────────────────────────────────────────────────────────────────

# SI convention for all .nc output (adopted 2026-06-03):
#   ψ (streamfunction)  → m²/s       (Wu file × PSI_SCALE)
#   Φ (geopotential)    → m²/s²      (Wu H [m] × G)
#   PV (Ertel)          → K·m²·kg⁻¹·s⁻¹  (SI native, ≡ 10⁻⁶ PVU)
#   PVadv (advection)   → K·m²·kg⁻¹·s⁻²  (SI tendency, ≡ PVU/s × 1e-6)
#
# ═══════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
QUIVER_SKIP = 4        # plot every Nth grid point
REF_SPEED   = 20.0     # m/s for reference arrow
WIND_DISPLAY_CAP = 40.0  # m/s — cap displayed wind magnitude
GAUSSIAN_SIGMA = 1.5   # σ for piece-3 smoothing

# ═══════════════════════════════════════════════════════════════════════════════
# 9. PATHS (derived — do not edit)
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
