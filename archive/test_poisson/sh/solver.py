"""Spectral (spherical-harmonic) Poisson/Helmholtz solver backends.

These wrappers call :mod:`pvtend.sh_ops` with standardised interfaces
so they can be benchmarked alongside Wu SOR and xinvert in the
``test_poisson`` harness.  All solvers here close cleanly at both
poles and (via parity mirroring) at the equator for NH-only inputs.

.. note::
   The spectral H-equation solver (``solve_helmholtz_3d_sh``) uses
   per-level Helmholtz inversion with **explicitly lagged** vertical
   coupling terms — identical to the approach in the archived SH-PPVI
   method.  This limits the under-relaxation parameter ``part`` to
   ≲ 0.07 in the full nonlinear BALNC iteration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import R_EARTH


def solve_poisson_2d_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    **_kwargs,
) -> np.ndarray:
    """Solve ∇²ψ = *rhs* on the sphere (mean-zero gauge).

    Thin wrapper around :func:`pvtend.sh_ops.invert_laplacian_sh`.

    Args:
        rhs: RHS field, shape ``(nlat, nlon)``.
        lat: Latitudes [°], ascending.
        lon: Longitudes [°].

    Returns:
        ψ of the same shape as *rhs*.
    """
    from pvtend.sh_ops import invert_laplacian_sh
    return invert_laplacian_sh(
        np.asarray(rhs, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        R_earth=R_EARTH,
        parity="scalar",
    )


def solve_helmholtz_2d_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    c: float = 0.0,
    **_kwargs,
) -> np.ndarray:
    """Solve (∇² + *c*) χ = *rhs* on the sphere.

    Thin wrapper around :func:`pvtend.sh_ops.invert_helmholtz_sh`.

    Args:
        rhs: RHS field, shape ``(nlat, nlon)``.
        lat: Latitudes [°], ascending.
        lon: Longitudes [°].
        c: Helmholtz constant (can be negative).

    Returns:
        χ of the same shape as *rhs*.
    """
    from pvtend.sh_ops import invert_helmholtz_sh
    return invert_helmholtz_sh(
        np.asarray(rhs, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        c=float(c),
        R_earth=R_EARTH,
        parity="scalar",
    )


def solve_helmholtz_3d_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    c_k: np.ndarray,
    bh_k: np.ndarray,
    bl_k: np.ndarray,
    H_init: np.ndarray | None = None,
    max_outer: int = 50,
    part: float = 0.05,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, int, bool]:
    """Per-level spectral Helmholtz with lagged vertical coupling.

    Solves the 3-D H-equation from the nonlinear balance system:

        (∇² + c_k) H_k + bh_k·H_{k+1} + bl_k·H_{k-1} = rhs_k

    by alternating horizontal spectral solves with explicit vertical
    coupling terms on the RHS.  This is the same algorithm used in the
    archived ``sh_ppvi/balnc.py``.

    Args:
        rhs: RHS per level, shape ``(nlev, nlat, nlon)``.
        lat: Latitudes [°], ascending.
        lon: Longitudes [°].
        c_k: Helmholtz coefficients per level, shape ``(nlev,)``.
        bh_k: Super-diagonal coupling per level, shape ``(nlev,)``.
        bl_k: Sub-diagonal coupling per level, shape ``(nlev,)``.
        H_init: Initial guess, shape ``(nlev, nlat, nlon)``.
                If ``None``, zeros are used.
        max_outer: Maximum outer iterations.
        part: Under-relaxation factor for H update.
        tol: Relative convergence tolerance.

    Returns:
        ``(H, n_iter, converged)`` where *H* shape matches *rhs*.
    """
    from pvtend.sh_ops import invert_helmholtz_sh

    nlev, nlat, nlon = rhs.shape
    rhs = np.asarray(rhs, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    c_k = np.asarray(c_k, dtype=np.float64)
    bh_k = np.asarray(bh_k, dtype=np.float64)
    bl_k = np.asarray(bl_k, dtype=np.float64)

    H = (
        np.asarray(H_init, dtype=np.float64).copy()
        if H_init is not None
        else np.zeros_like(rhs)
    )

    norm_init = float(np.linalg.norm(rhs))
    if norm_init < 1e-30:
        return H, 0, True

    converged = False
    for outer in range(max_outer):
        H_old = H.copy()
        max_dh = 0.0

        for k in range(nlev):
            # Build effective RHS:  rhs_k − bh·H_{k+1} − bl·H_{k-1}
            rhs_k = rhs[k].copy()
            if k > 0:
                rhs_k -= bl_k[k] * H[k - 1]
            if k < nlev - 1:
                rhs_k -= bh_k[k] * H[k + 1]

            H_new_k = invert_helmholtz_sh(
                rhs_k, lat, lon,
                c=float(c_k[k]),
                R_earth=R_EARTH,
                parity="scalar",
            )

            dh = np.abs(H_new_k - H[k]).max()
            max_dh = max(max_dh, dh)

            # Under-relax
            H[k] = part * H_new_k + (1.0 - part) * H[k]

        rel_change = max_dh / (np.abs(H).max() + 1e-30)
        if rel_change < tol:
            converged = True
            break

    return H, outer + 1, converged
