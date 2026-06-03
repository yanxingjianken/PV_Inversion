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

from wu_python.core.grid import NY, NX, FC, AP, DP, DL  # noqa: E402
from wu_python.core.nondim import G, CP, P0, KAP, PI_VALS  # noqa: E402
from wu_python.core.fd_ops import gradient_x, gradient_y, d_dpi  # noqa: E402
from wu_python.core.sor_solver import sor_poisson_2d  # noqa: E402


# Wu's PV coefficient (includes 100× over-scaling for historical compatibility)
COEF: float = 1.0e2 * 1.0e6 * G * KAP * (CP ** 3.5) / P0


def compute_relative_vorticity(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute relative vorticity ζ = ∂v/∂x − ∂u/∂y on each σ-level.

    Uses the AP-weighted stencil matching pvpialln_94UV.f:
      VL  = (V[j+1] - V[j-1]) / (2*DL*AP[i])
      UPV = (AP[i-1]*U[i-1,j] - AP[i+1]*U[i+1,j]) / (2*DP*AP[i])  ← AP-weighted
      ζ = VL - UPV

    Args:
        U: Zonal wind (NW, NY, NX) [m/s]
        V: Meridional wind (NW, NY, NX) [m/s]

    Returns:
        ζ: Relative vorticity (NW, NY, NX) [s⁻¹]
    """
    NW = U.shape[0]
    zeta = np.zeros_like(U)
    for k in range(NW):
        # ∂v/∂x  (unchanged — matches Fortran VL term)
        dvdx = gradient_x(V[k])

        # AP-weighted ∂u/∂y (matches Fortran UPV term)
        # UPV[i,j] = (AP[i-1]*U[i-1,j] - AP[i+1]*U[i+1,j]) / (2*DP*AP[i])
        upv = np.zeros_like(U[k])
        upv[1:-1, :] = (
            AP[:-2, np.newaxis] * U[k][:-2, :]
            - AP[2:, np.newaxis] * U[k][2:, :]
        ) / (2.0 * DP * AP[1:-1, np.newaxis])

        zeta[k] = dvdx - upv
    return zeta


def compute_stability(U: np.ndarray, V: np.ndarray,
                      TH: np.ndarray) -> np.ndarray:
    """Compute static stability STB = (f+ζ)·∂θ/∂π + ∂u/∂π·∂θ/∂y − ∂v/∂π·∂θ/∂x.

    This matches the Fortran pvpialln_94UV.f convention:
      Q = -COEF·π^(-5/2) · [(f+ζ)·STB + DU·DTHY − DV·DTHX]
    where STB = ∂θ/∂π.

    Note: Fortran uses STB = ∂θ/∂π (stratification only), then adds
    the shear terms separately in the PV expression. We combine them
    into a single STB for efficiency, but must use the SAME SIGN convention
    as Fortran: +∂u/∂π·∂θ/∂y − ∂v/∂π·∂θ/∂x.

    Args:
        U, V: Wind components (NW, NY, NX)
        TH:   Potential temperature (NW, NY, NX)

    Returns:
        STB: Combined stability + shear term (NW, NY, NX)
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

    # Fortran convention: +∂u/∂π·∂θ/∂y − ∂v/∂π·∂θ/∂x
    return abs_vor * dth_dpi + du_dpi * dth_dy - dv_dpi * dth_dx


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
                              U: np.ndarray = None, V: np.ndarray = None,
                              omega: float = 1.75,
                              max_iter: int = 50000,
                              tol: float = 1e1) -> np.ndarray:
    """Compute balanced ψ from ∇²ψ = ζ with BC from wind integration.

    Port of pvpialln_94UV.f (Pass A/B).
    Boundary ψ is computed by integrating winds around the domain edge
    (Davis eq 2.40), NOT simple gH/f. Interior ψ refined by SOR.

    Args:
        VOR:      Relative vorticity (NW, NY, NX) [s⁻¹]
        H:        Geopotential height (NW, NY, NX) [m]
        U:        Zonal wind (NW, NY, NX) [m/s]. Required for boundary integration.
        V:        Meridional wind (NW, NY, NX) [m/s]
        omega:    SOR relaxation factor
        max_iter: Max SOR iterations
        tol:      Convergence tolerance on |Δψ|_max (in primitive units)

    Returns:
        PSI: Balanced streamfunction (NW, NY, NX) [m²/s]
    """
    NW = VOR.shape[0]
    PSI = np.zeros_like(VOR)

    # If U,V provided, use wind integration for boundary ψ (matching Fortran).
    # Otherwise fall back to simple gH/f (legacy behavior).
    use_wind_bc = U is not None and V is not None

    for k in range(NW):
        if use_wind_bc:
            bc = _boundary_psi_from_winds(H[k], U[k], V[k])
        else:
            bc = G * H[k] / FC[:, np.newaxis]
            bc = np.where(np.abs(FC[:, np.newaxis]) > 1e-8, bc, 0.0)

        # Scale RHS by DL² to match Fortran convention:
        #   Fortran: Lapsi = (1/DL²)*sum(A*ψ), update = -omegs*RS/A[3]*(DL*DL)
        #   → equation solved is: sum(A*ψ) = VOR * DL²
        rhs = VOR[k] * DL * DL
        psi_k, n_iter, max_err = sor_poisson_2d(
            rhs, bc, omega=omega, max_iter=max_iter, tol=tol
        )
        PSI[k] = psi_k

    return PSI


def _boundary_psi_from_winds(H_k: np.ndarray, U_k: np.ndarray,
                              V_k: np.ndarray) -> np.ndarray:
    """Compute ψ on the 1-cell boundary frame by integrating winds.

    Port of pvpialln_94UV.f lines 338-383 (Davis eq 2.40).
    Starts at NW corner with ψ = gH/f, then integrates along all 4 edges
    using U (zonal) and V (meridional) winds, with a divergence correction.

    Args:
        H_k: Geopotential height (NY, NX) [m]
        U_k: Zonal wind (NY, NX) [m/s]
        V_k: Meridional wind (NY, NX) [m/s]

    Returns:
        psi_bc: ψ on the full grid (NY, NX) — boundary values are
                wind-integrated, interior values are H*g/f (initial guess).
    """
    ny, nx = H_k.shape
    psi = np.zeros((ny, nx))

    # ── Step 1: divergence correction dsum ──
    # Fortran: dsum = -∫U·DP along west + ∫U·DP along east
    #                 + ∫V·DL·AP along north - ∫V·DL·AP along south
    dsum = 0.0
    for i in range(ny - 1):
        dsum -= 0.5 * (U_k[i, 0] + U_k[i + 1, 0]) * DP            # west edge
        dsum += 0.5 * (U_k[i, nx - 1] + U_k[i + 1, nx - 1]) * DP  # east edge
    for j in range(nx - 1):
        dsum += 0.5 * (V_k[0, j] + V_k[0, j + 1]) * DL * AP[0]              # north edge
        dsum -= 0.5 * (V_k[ny - 1, j] + V_k[ny - 1, j + 1]) * DL * AP[ny - 1]  # south edge

    perimeter = 2.0 * DP * (ny - 1) + DL * (nx - 1) * (AP[0] + AP[ny - 1])
    dsum /= perimeter

    # ── Step 2: start at NW corner ──
    psi[0, 0] = H_k[0, 0] * G / FC[0]

    # ── Step 3: integrate south along west edge (i=0..ny-2, j=0) ──
    for i in range(ny - 1):
        u_avg = 0.5 * (U_k[i, 0] + U_k[i + 1, 0])
        psi[i + 1, 0] = psi[i, 0] + (dsum + u_avg) * DP

    # ── Step 4: integrate east along south edge (i=ny-1, j=0..nx-2) ──
    for j in range(nx - 1):
        v_avg = 0.5 * (V_k[ny - 1, j] + V_k[ny - 1, j + 1])
        psi[ny - 1, j + 1] = psi[ny - 1, j] + (dsum + v_avg) * DL * AP[ny - 1]

    # ── Step 5: integrate north along east edge (i=ny-1..1, j=nx-1) ──
    for i in range(ny - 1, 0, -1):
        u_avg = 0.5 * (U_k[i, nx - 1] + U_k[i - 1, nx - 1])
        psi[i - 1, nx - 1] = psi[i, nx - 1] + (dsum - u_avg) * DP

    # ── Step 6: integrate west along north edge (i=0, j=nx-1..2) ──
    for j in range(nx - 1, 1, -1):
        v_avg = 0.5 * (V_k[0, j] + V_k[0, j - 1])
        psi[0, j - 1] = psi[0, j] + (dsum - v_avg) * DL * AP[0]

    # ── Fill interior with initial geostrophic guess ──
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            psi[i, j] = H_k[i, j] * G / FC[i]

    return psi
