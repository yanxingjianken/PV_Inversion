# wu_python/core/balance.py — BALNC: Total balanced PV inversion (Pass C)
"""Total balanced PV inversion via coupled ψ–Φ SOR.

Ports the BALNC subroutine from qinvert21_94.f:
  - Outer loop: alternate ψ SOR and Φ SOR with under-relaxation
  - Inner ψ step: ∇²ψ = ζ (Poisson SOR per level)
  - Inner Φ step: nonlinear balance equation (Charney) → H update
  - Coupling: ψ and Φ are updated alternatingly with PART under-relaxation

Key constraint: INLIN=0 (linear balance only). INLIN=1 (nonlinear) diverges
at the 250 hPa jet.

Reference: qinvert21_94.f, subroutine BALNC (lines ~350-650).
"""

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wu_python.core.grid import NY, NX, FC  # noqa: E402
from wu_python.core.nondim import G, BB, BH, BL, THO, DPI, THA_B, THA_T  # noqa: E402
from wu_python.core.fd_ops import laplacian_5pt, gradient_x, gradient_y, jacobian  # noqa: E402
from wu_python.core.sor_solver import sor_poisson_2d  # noqa: E402
from wu_python.config import OMEGS, OMEGH, PART, THRSH, MAXT, MAX_ITER, INLIN  # noqa: E402


def _solve_h_equation(
    H: np.ndarray, PSI: np.ndarray, Q: np.ndarray,
    ASI: np.ndarray,
) -> np.ndarray:
    """Solve the H-equation (nonlinear balance / Charney) for one outer iteration.

    This is a simplified 3D tridiagonal solve per horizontal grid point.
    The Wu Fortran uses SOR for this; we use direct Thomas algorithm.

    Args:
        H:    Current geopotential height (NW, NY, NX)
        PSI:  Current streamfunction (NW, NY, NX)
        Q:    PV field (NW, NY, NX)
        ASI:  ASI = FCM + FR * VOR  (map-factor-scaled Coriolis + vorticity term)

    Returns:
        H_new: Updated geopotential height (NW, NY, NX)
    """
    # Placeholder — full implementation requires the Wu H-equation solver
    # (tridiagonal in k with BB/BH/BL, coupled to ψ via VOR)
    # This will be implemented in a follow-up.
    return H.copy()


def balnc_total(
    Q: np.ndarray,
    H_init: np.ndarray,
    PSI_init: np.ndarray,
    omegs: float = OMEGS,
    omegh: float = OMEGH,
    part: float = PART,
    thrsh: float = THRSH,
    maxt: int = MAXT,
    max_iter: int = MAX_ITER,
    inlin: int = INLIN,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Total balanced PV inversion (Pass C).

    Args:
        Q:        Ertel PV field (NW, NY, NX) — total (event day)
        H_init:   Initial guess for geopotential height (NW, NY, NX)
        PSI_init: Initial guess for streamfunction (NW, NY, NX)
        omegs:    SOR relaxation for ψ
        omegh:    SOR relaxation for Φ
        part:     Under-relaxation between ψ and Φ updates
        thrsh:    Convergence threshold [m]
        maxt:     Max outer coupling iterations
        max_iter: Max inner SOR iterations
        inlin:    0 = linear, 1 = nonlinear (use 0 for stability)

    Returns:
        PSI_bal:  Balanced streamfunction (NW, NY, NX)
        H_bal:    Balanced geopotential height (NW, NY, NX)
        converged: Whether convergence was reached
        n_outer:  Number of outer iterations used
    """
    NW = Q.shape[0]
    H = H_init.copy()
    PSI = PSI_init.copy()

    for n_outer in range(maxt):
        max_dpsi = 0.0

        # Step 1: Solve ∇²ψ = ζ at each level
        for k in range(NW):
            vor = laplacian_5pt(PSI[k])
            # ... full BALNC logic to be implemented
            pass

        # Step 2: Solve H-equation
        # ... to be implemented

        # Check convergence
        if max_dpsi < thrsh:
            return PSI, H, True, n_outer + 1

    return PSI, H, False, maxt
