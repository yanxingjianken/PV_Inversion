# wu_python/core/fd_ops.py — Finite-difference operators on the Wu grid
"""Finite-difference operators matching pvpialln_94UV.f and qinvert21_94.f.

All operators work on (NY, NX) 2D arrays with:
  - i = 0..NY-1 (north→south, Fortran I index)
  - j = 0..NX-1 (west→east,  Fortran J index)

Boundary conditions: Dirichlet (ψ = gH/f on frame).

Uses numba JIT for performance when USE_NUMBA=True.
"""

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wu_python.core.grid import NY, NX, DL, DP, SIGM, AP, APM, APP, A  # noqa: E402
from wu_python.config import USE_NUMBA  # noqa: E402

if USE_NUMBA:
    try:
        from numba import jit, prange
        _NUMBA_AVAILABLE = True
    except ImportError:
        _NUMBA_AVAILABLE = False
else:
    _NUMBA_AVAILABLE = False


def _maybe_jit(func):
    """Apply numba JIT if available."""
    if _NUMBA_AVAILABLE:
        return jit(nopython=True, parallel=True, cache=True)(func)
    return func


# ── 5-point Laplacian ──────────────────────────────────────────────────────

@_maybe_jit
def _laplacian_5pt_kernel(psi: np.ndarray, out: np.ndarray) -> None:
    """Numba kernel: 5-point Laplacian ∇²ψ on (NY,NX) grid.

    Wu formula (A coefficients precomputed in grid.py):
      ∇²ψ[i,j] = A[i,1]*ψ[i-1,j] + A[i,2]*ψ[i,j-1] + A[i,3]*ψ[i,j]
               + A[i,4]*ψ[i,j+1] + A[i,5]*ψ[i+1,j]

    Note: i increases southward (row +1 = next latitude south).
    """
    for i in range(1, NY - 1):
        for j in range(1, NX - 1):
            out[i, j] = (
                A[i, 0] * psi[i - 1, j]
                + A[i, 1] * psi[i, j - 1]
                + A[i, 2] * psi[i, j]
                + A[i, 3] * psi[i, j + 1]
                + A[i, 4] * psi[i + 1, j]
            )


def laplacian_5pt(psi: np.ndarray) -> np.ndarray:
    """Compute 5-point Laplacian ∇²ψ on the Wu grid.

    Args:
        psi: Streamfunction field (NY, NX) [m²/s]

    Returns:
        ∇²ψ: (NY, NX) — relative vorticity proxy [s⁻¹]
    """
    out = np.zeros_like(psi)
    if _NUMBA_AVAILABLE:
        _laplacian_5pt_kernel(psi, out)
    else:
        # Pure numpy fallback (vectorized)
        out[1:-1, 1:-1] = (
            A[1:-1, 0, np.newaxis] * psi[0:-2, 1:-1]
            + A[1:-1, 1, np.newaxis] * psi[1:-1, 0:-2]
            + A[1:-1, 2, np.newaxis] * psi[1:-1, 1:-1]
            + A[1:-1, 3, np.newaxis] * psi[1:-1, 2:]
            + A[1:-1, 4, np.newaxis] * psi[2:, 1:-1]
        )
    return out


# ── Gradient operators ─────────────────────────────────────────────────────

def gradient_x(field: np.ndarray) -> np.ndarray:
    """∂/∂x (zonal gradient) with cos(lat) scaling.

    ∂f/∂x = (f[j+1] - f[j-1]) / (2 * DL * cos(lat_i))

    Args:
        field: (NY, NX) array

    Returns:
        ∂field/∂x: (NY, NX) [field units / m]
    """
    out = np.zeros_like(field)
    for i in range(NY):
        out[i, 1:-1] = (field[i, 2:] - field[i, :-2]) / (2.0 * DL * AP[i])
    return out


def gradient_y(field: np.ndarray) -> np.ndarray:
    """∂/∂y (meridional gradient).

    ∂f/∂y = (f[i-1] - f[i+1]) / (2 * DP)
    Note: i-1 is northward (toward pole), i+1 is southward.

    Args:
        field: (NY, NX) array

    Returns:
        ∂field/∂y: (NY, NX) [field units / m]
    """
    out = np.zeros_like(field)
    out[1:-1, :] = (field[:-2, :] - field[2:, :]) / (2.0 * DP)
    return out


# ── Jacobian ───────────────────────────────────────────────────────────────

def jacobian(psi: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Horizontal Jacobian J(ψ, f) = ∂ψ/∂x·∂f/∂y − ∂ψ/∂y·∂f/∂x.

    Args:
        psi: Streamfunction (NY, NX)
        field: Any scalar field (NY, NX)

    Returns:
        J(ψ, field): (NY, NX)
    """
    dpsi_dx = gradient_x(psi)
    dpsi_dy = gradient_y(psi)
    df_dx = gradient_x(field)
    df_dy = gradient_y(field)
    return dpsi_dx * df_dy - dpsi_dy * df_dx


# ── Vertical derivative on σ ───────────────────────────────────────────────

def d_dpi(field_3d: np.ndarray) -> np.ndarray:
    """Vertical derivative ∂/∂π on sigma (Exner) coordinate.

    Uses centered differences for interior levels, one-sided at boundaries.

    Args:
        field_3d: (NW, NY, NX) — field on sigma levels

    Returns:
        ∂field/∂π: (NW, NY, NX)
    """
    from wu_python.core.nondim import PI_WU

    NW = field_3d.shape[0]
    out = np.zeros_like(field_3d)

    # Interior: centered
    for k in range(1, NW - 1):
        dpi = PI_WU[k + 1] - PI_WU[k - 1]
        out[k] = (field_3d[k + 1] - field_3d[k - 1]) / dpi

    # Boundaries: one-sided
    dpi = PI_WU[1] - PI_WU[0]
    out[0] = (field_3d[1] - field_3d[0]) / dpi
    dpi = PI_WU[-1] - PI_WU[-2]
    out[-1] = (field_3d[-1] - field_3d[-2]) / dpi

    return out
