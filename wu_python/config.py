# wu_python/config.py — Configuration for the pure-Python Wu PPVI track
"""Thin wrapper re-exporting root config.py, plus Python-specific settings."""

import sys
from pathlib import Path

# Import root config (from pv_inversion/config.py)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import *  # noqa: E402, F403 — re-export all root config

# ── Python-specific settings ──────────────────────────────────────────────

# Output directory for wu_python pipeline results
PYTHON_OUT_DIR = Path(DATA_DIR) / "wu_python_out"
PYTHON_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parallelism ───────────────────────────────────────────────────────────

# Number of worker processes for pass/piece-level parallelism
# (383 cores available on this machine; 32 is a safe default)
N_WORKERS: int = 32

# Use numba JIT for hot loops (finite differences, SOR sweeps)
USE_NUMBA: bool = True

# ── SOR Solver Parameters (mirrors Fortran config) ────────────────────────

OMEGS: float = 1.4        # SOR relaxation factor for ψ
OMEGH: float = 1.4        # SOR relaxation factor for Φ
PART: float = 0.5         # Under-relaxation for ψ–Φ coupling
THRSH: float = 0.01       # Convergence threshold [m]
INLIN: int = 0            # 0 = linear balance (stable); 1 = nonlinear (diverges)
MAX_ITER: int = 5000      # Max SOR iterations per level
MAXT: int = 500           # Max outer ψ–Φ coupling cycles

# ── Cross-validation tolerance ────────────────────────────────────────────

# Acceptable RMS difference vs Fortran reference at 500 hPa
CROSS_VALIDATION_TOL: float = 0.05  # 5% RMS
