# wu_python/core/sor_solver.py — Red-black SOR Poisson solver
"""Red-black Successive Over-Relaxation (SOR) for 2D Poisson problems.

Solves ∇²ψ = ζ on the Wu (NY, NX) grid with Dirichlet boundary conditions
ψ = gH/f on the 1-cell frame.

Red-black ordering colours the grid like a checkerboard:
  - Red cells (i+j even): depend only on black neighbour values
  - Black cells (i+j odd): depend only on red neighbour values
  → Each colour has ~2200 independent cells → parallelizable via numba prange.

The algorithm is mathematically identical to sequential SOR (same fixed point,
same convergence rate). Only floating-point summation order differs slightly.

Reference: qinvert21_94.f subroutine BALNC, SOR loop at ~L400-500.
"""

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wu_python.core.grid import NY, NX, A  # noqa: E402
from wu_python.config import USE_NUMBA, OMEGS, MAX_ITER  # noqa: E402

_NUMBA_AVAILABLE = False
if USE_NUMBA:
    try:
        from numba import jit, prange
        _NUMBA_AVAILABLE = True
    except ImportError:
        pass


if _NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _sor_red_black_numba(
        psi: np.ndarray, rhs: np.ndarray, omega: float,
        max_iter: int, tol: float, A_arr: np.ndarray,
    ) -> tuple:
        """Numba-accelerated red-black SOR sweep."""
        ny, nx = psi.shape
        for n in range(max_iter):
            max_err = 0.0
            # Red sweep: i+j even
            for i in prange(1, ny - 1):
                for j in range(1, nx - 1):
                    if (i + j) % 2 == 0:
                        lap = (
                            A_arr[i, 0] * psi[i - 1, j]
                            + A_arr[i, 1] * psi[i, j - 1]
                            + A_arr[i, 2] * psi[i, j]
                            + A_arr[i, 3] * psi[i, j + 1]
                            + A_arr[i, 4] * psi[i + 1, j]
                        )
                        residual = rhs[i, j] - lap
                        delta = omega * residual / A_arr[i, 2]
                        psi[i, j] += delta
                        if abs(delta) > max_err:
                            max_err = abs(delta)
            # Black sweep: i+j odd
            for i in prange(1, ny - 1):
                for j in range(1, nx - 1):
                    if (i + j) % 2 == 1:
                        lap = (
                            A_arr[i, 0] * psi[i - 1, j]
                            + A_arr[i, 1] * psi[i, j - 1]
                            + A_arr[i, 2] * psi[i, j]
                            + A_arr[i, 3] * psi[i, j + 1]
                            + A_arr[i, 4] * psi[i + 1, j]
                        )
                        residual = rhs[i, j] - lap
                        delta = omega * residual / A_arr[i, 2]
                        psi[i, j] += delta
                        if abs(delta) > max_err:
                            max_err = abs(delta)
            if max_err < tol:
                return n + 1, max_err
        return max_iter, max_err


def sor_poisson_2d(
    rhs: np.ndarray,
    bc: np.ndarray,
    omega: float = OMEGS,
    max_iter: int = MAX_ITER,
    tol: float = 1e-6,
) -> tuple:
    """Red-black SOR for 2D Poisson equation ∇²ψ = rhs.

    Solves on interior points (1..NY-2, 1..NX-2) with ψ = bc on the frame.

    Args:
        rhs:   Right-hand side (NY, NX) — the vorticity/forcing field
        bc:    Boundary condition values (NY, NX) — ψ on the frame
        omega: SOR relaxation factor (1.0 = Gauss-Seidel, >1 = over-relaxed)
        max_iter: Maximum iterations
        tol:   Convergence tolerance on |Δψ|_max

    Returns:
        psi:     Solution (NY, NX)
        n_iter:  Number of iterations used
        max_err: Final maximum absolute change |Δψ|_max
    """
    psi = bc.copy().astype(np.float64)
    rhs = rhs.astype(np.float64)
    A_arr = A.astype(np.float64)

    if _NUMBA_AVAILABLE:
        n_iter, max_err = _sor_red_black_numba(psi, rhs, omega, max_iter, tol, A_arr)
        return psi, n_iter, max_err

    # Pure Python fallback
    for n in range(max_iter):
        max_err = 0.0
        # Red sweep
        for i in range(1, NY - 1):
            for j in range(1, NX - 1):
                if (i + j) % 2 != 0:
                    continue
                lap = (
                    A[i, 0] * psi[i - 1, j]
                    + A[i, 1] * psi[i, j - 1]
                    + A[i, 2] * psi[i, j]
                    + A[i, 3] * psi[i, j + 1]
                    + A[i, 4] * psi[i + 1, j]
                )
                residual = rhs[i, j] - lap
                delta = omega * residual / A[i, 2]
                psi[i, j] += delta
                max_err = max(max_err, abs(delta))
        # Black sweep
        for i in range(1, NY - 1):
            for j in range(1, NX - 1):
                if (i + j) % 2 != 1:
                    continue
                lap = (
                    A[i, 0] * psi[i - 1, j]
                    + A[i, 1] * psi[i, j - 1]
                    + A[i, 2] * psi[i, j]
                    + A[i, 3] * psi[i, j + 1]
                    + A[i, 4] * psi[i + 1, j]
                )
                residual = rhs[i, j] - lap
                delta = omega * residual / A[i, 2]
                psi[i, j] += delta
                max_err = max(max_err, abs(delta))
        if max_err < tol:
            return psi, n + 1, max_err

    return psi, max_iter, max_err
