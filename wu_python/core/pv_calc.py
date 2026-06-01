# wu_python/core/pv_calc.py — Ertel PV + Balanced ψ (Pass A/B equivalent)
"""Compute Ertel Potential Vorticity and balanced streamfunction from U, V, θ, H.

Ports the logic from pvpialln_94UV.f:
  1. Ertel PV on interior σ-levels (k = 1..NW-2, i.e. 925-250 hPa)
  2. Balanced ψ via ∇²ψ = ζ with Dirichlet BC ψ = gH/f on the frame

Wu's internal PV unit includes a 100× over-scaling (COEF = 1e2 * ...).
We preserve this for cross-validation; physical PVU is computed separately.

Reference: pvpialln_94UV.f lines ~450-480 (PV), ~500-550 (ψ inversion).
"""

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wu_python.core.grid import NY, NX, FC  # noqa: E402
from wu_python.core.nondim import G, CP, P0, KAP, PI_VALS  # noqa: E402
from wu_python.core.fd_ops import gradient_x, gradient_y, d_dpi  # noqa: E402
from wu_python.core.sor_solver import sor_poisson_2d  # noqa: E402


# Wu's PV coefficient (includes 100× over-scaling for historical compatibility)
COEF: float = 1.0e2 * 1.0e6 * G * KAP * (CP ** 3.5) / P0


def compute_relative_vorticity(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute relative vorticity ζ = ∂v/∂x − ∂u/∂y on each σ-level.

    Args:
        U: Zonal wind (NW, NY, NX) [m/s]
        V: Meridional wind (NW, NY, NX) [m/s]

    Returns:
        ζ: Relative vorticity (NW, NY, NX) [s⁻¹]
    """
    NW = U.shape[0]
    zeta = np.zeros_like(U)
    for k in range(NW):
        zeta[k] = gradient_x(V[k]) - gradient_y(U[k])
    return zeta


def compute_stability(U: np.ndarray, V: np.ndarray,
                      TH: np.ndarray) -> np.ndarray:
    """Compute static stability STB = (f+ζ)·∂θ/∂π − ∂u/∂π·∂θ/∂y + ∂v/∂π·∂θ/∂x.

    This is the numerator of the Ertel PV expression (without the -g·κ·Cp³·⁵/P0 prefactor).

    Args:
        U, V: Wind components (NW, NY, NX)
        TH:   Potential temperature (NW, NY, NX)

    Returns:
        STB: Static stability × absolute vorticity term (NW, NY, NX)
    """
    zeta = compute_relative_vorticity(U, V)
    dth_dpi = d_dpi(TH)
    du_dpi = d_dpi(U)
    dv_dpi = d_dpi(V)

    abs_vor = FC[np.newaxis, :, np.newaxis] + zeta  # (NW, NY, NX)
    dth_dy = np.zeros_like(TH)
    dth_dx = np.zeros_like(TH)
    for k in range(TH.shape[0]):
        dth_dy[k] = gradient_y(TH[k])
        dth_dx[k] = gradient_x(TH[k])

    return abs_vor * dth_dpi - du_dpi * dth_dy + dv_dpi * dth_dx


def compute_ertel_pv_wu(U: np.ndarray, V: np.ndarray, TH: np.ndarray) -> np.ndarray:
    """Compute Ertel PV in Wu internal units (~340-600× PVU).

    Q = -COEF · PI(L)^(-5/2) · STB

    Args:
        U, V: Wind components (NW, NY, NX)
        TH:   Potential temperature (NW, NY, NX)

    Returns:
        Q: Ertel PV in Wu internal units (NW-2, NY, NX) — interior levels only
    """
    STB = compute_stability(U, V, TH)
    NW = U.shape[0]
    NW_PV = NW - 2  # interior levels (k=1..NW-2, Fortran 1-based k=2..NW-1)

    Q = np.zeros((NW_PV, NY, NX))
    for k in range(NW_PV):
        k_fortran = k + 1  # Fortran uses 1-based indexing
        Q[k] = -COEF * (PI_VALS[k_fortran] ** (-2.5)) * STB[k_fortran]

    # Set sentinel (9999.9) for boundaries (matching Fortran)
    Q[:, 0, :] = 9999.9
    Q[:, -1, :] = 9999.9
    Q[:, :, 0] = 9999.9
    Q[:, :, -1] = 9999.9

    return Q


def invert_vorticity_balanced(VOR: np.ndarray, H: np.ndarray,
                              omega: float = 1.75,
                              max_iter: int = 300) -> np.ndarray:
    """Compute balanced ψ from ∇²ψ = ζ with BC ψ = gH/f.

    This is the vorticity inversion step in pvpialln_94UV.f (Pass A/B).

    Args:
        VOR:      Relative vorticity (NW, NY, NX) [s⁻¹]
        H:        Geopotential height (NW, NY, NX) [m]
        omega:    SOR relaxation factor
        max_iter: Max SOR iterations

    Returns:
        PSI: Balanced streamfunction (NW, NY, NX) [m²/s]
    """
    NW = VOR.shape[0]
    PSI = np.zeros_like(VOR)

    for k in range(NW):
        # Boundary condition: ψ = gH/f on the frame
        bc = G * H[k] / FC[:, np.newaxis]
        # Avoid division by zero near equator (FC ≈ 0 at lat=0)
        bc = np.where(np.abs(FC[:, np.newaxis]) > 1e-8, bc, 0.0)

        rhs = VOR[k].copy()
        psi_k, n_iter, max_err = sor_poisson_2d(
            rhs, bc, omega=omega, max_iter=max_iter, tol=5e4
        )
        PSI[k] = psi_k

    return PSI
