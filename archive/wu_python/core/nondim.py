# wu_python/core/nondim.py — Wu non-dimensionalization scales
"""Non-dimensionalization constants matching qinvert21_94.f and pvpialln_94UV.f.

Copied/adapted from sh/sh_ppvi/nondim.py — the physics is identical; only the
operator backend differs (finite differences here vs spherical harmonics there).

Key references:
  - qinvert21_94.f lines ~180-220: DPI, BB/BH/BL definition
  - pvpialln_94UV.f lines ~120-140: CP, P0, KAP, PI(K)
  - Wu user memory: BB/BH/BL MUST be built from PI_WU, not raw PI_VALS
"""

import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────
G: float = 9.80665            # gravity [m/s²]
CP: float = 1004.5            # specific heat of dry air [J/(kg·K)]
RD: float = 287.0             # gas constant for dry air [J/(kg·K)]
P0: float = 1.0e5             # reference pressure [Pa]
KAP: float = 2.0 / 7.0        # Rd/Cp
PII: float = np.pi

# ── Sigma levels and Exner function ────────────────────────────────────────
PR: np.ndarray = np.array([1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
NW: int = len(PR)

# Exner function π = Cp * (p/p0)^κ  (Wu: PI)
PI_VALS: np.ndarray = CP * (PR ** KAP)

# Wu non-dimensionalization parameter (DPI in Fortran)
DPI: float = 50.0

# Wu-scaled Exner: PI_WU = CP * (p/p0)^κ / DPI ≈ 20.08 * PI_VALS
PI_WU: np.ndarray = PI_VALS / DPI

# ── Non-dimensional scales ─────────────────────────────────────────────────
# Grid spacing scale [m]
LL: float = 1.667e5            # ≈ DP

# Coriolis scale [s⁻¹]
FF: float = 1.0e-4

# Non-dimensional height scale (Wu: THO)
THO: float = FF * FF * LL * LL / DPI  # ≈ 5.5556

# ── Vertical operator coefficients BB, BH, BL ──────────────────────────────
# These are the tridiagonal coefficients for the H-equation vertical coupling.
# CRITICAL: built from PI_WU (Wu-scaled), NOT raw PI_VALS.
# If built from PI_VALS they are off by CP²/DPI² ≈ 403 → instant blow-up.

# qinvert21_94.f format: BB = diagonal, BH = super-diagonal, BL = sub-diagonal
# Indices k = 1..NW (Fortran 1-based), interior k = 2..NW-1

BB: np.ndarray = np.array([
    0.0,
    -9.637, -4.121, -2.472, -2.903,
    -2.230, -1.611, -2.268, -3.314,
    0.0,
])

BH: np.ndarray = np.array([
    0.0,
    4.679, 1.285, 1.408, 1.365,
    1.035, 0.733, 1.424, 1.538,
    0.0,
])

BL: np.ndarray = np.array([
    0.0,
    4.958, 2.836, 1.064, 1.539,
    1.195, 0.878, 0.844, 1.776,
    0.0,
])

# Boundary theta contributions (non-dimensional)
THA_B: float = 0.0   # lower BC forcing (set by θ_bottom)
THA_T: float = 0.0   # upper BC forcing (set by θ_top)

# ── Verification ───────────────────────────────────────────────────────────
# BB + BH + BL ≈ 0 at each interior level → pure 2nd-derivative, no self-damping
_row_sum = BB + BH + BL
assert np.allclose(_row_sum[1:-1], 0.0, atol=1e-2), \
    f"BB+BH+BL not zero at interior levels: {_row_sum}"
