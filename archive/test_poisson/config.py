"""Shared configuration for test_poisson benchmarks.

All paths are relative to the pv_inversion repo root so tests can run from
any working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── repo root (test_poisson sits inside pv_inversion) ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── data paths ─────────────────────────────────────────────────────────────
DATA_WU_IN = REPO_ROOT / "data" / "wu_in"
EVENT_GRID = DATA_WU_IN / "event.grid"
MEAN_GRID = DATA_WU_IN / "mean.grid"

# ── grid (Wu NH domain) ────────────────────────────────────────────────────
NX: int = 87
NY: int = 51
NW: int = 10
DLAT: float = 1.5          # degrees
DLON: float = 1.5          # degrees
LAT_N: float = 85.5        # °N
LAT_S: float = 10.5        # °N
LON_W: float = -169.5      # °E (negative = W)
LON_E: float = -40.5       # °E
R_EARTH: float = 6.371e6   # m

# ── vertical ───────────────────────────────────────────────────────────────
# σ-levels (1.0 = surface, 0.2 = top ≈ 200 hPa)
SIGMA: tuple[float, ...] = (1.0, 0.925, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2)
PLEVS_HPA: tuple[float, ...] = tuple(s * 1000.0 for s in SIGMA)

# Wu non-dimensional scales (from qinvert21_94.f)
LL: float = 1.667e5         # horizontal length scale [m]
FF: float = 1.0e-4          # Coriolis scale [s⁻¹]
DPI: float = 50.0           # Exner scale
THO: float = 5.5556         # height scale [m]  (= FF²·LL²/DPI)

# ── SOR solver defaults ────────────────────────────────────────────────────
OMEGS: float = 1.75          # SOR relaxation for ψ
OMEGH: float = 1.75          # SOR relaxation for H
PART: float = 0.5            # ψ–H under-relaxation
MAX_ITER: int = 5000         # max inner SOR iterations
MAX_OUTER: int = 500         # max outer (Picard) iterations
TOL: float = 1e-6            # relative convergence tolerance
THRSH: float = 0.01          # absolute convergence threshold [m]

# ── xinvert defaults ───────────────────────────────────────────────────────
XINVERT_IPARAMS: dict = {
    "BCs": ["fixed", "periodic"],  # lat (fixed), lon (periodic)
    "undef": float("nan"),
    "mxLoop": 5000,
    "tolerance": 1e-6,
    "optArg": None,               # None → auto-compute optimal ω
}

# ── output ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
