"""sh_ppvi.wu_ops — Wu non-dimensionalised horizontal operators.

Purpose
-------
Wu's balance equations (qinvert21_94.f / qinvertp21_94.f) are written in
non-dimensional *grid-index* coordinates where Δx = LL (metres) and
derivatives are per grid-step.  The spherical-harmonic operators in
``operators.py`` return *physical-unit* results (m⁻¹, m⁻², etc.).

This module provides thin wrappers that absorb the LL (and LL²) scale
factors so that the rest of BALNC/BALP can be written as a 1-for-1
translation of the Fortran:

    Wu's Laplacian  ≡  LL² · ∇²_phys
    Wu's gradient   ≡  LL  · (∂/∂x_phys, ∂/∂y_phys)
    Wu's Jacobian   ≡  LL² · J_phys

The *only* methodological difference from Wu is that we use the spherical-
harmonic operator instead of his 5-point finite-difference stencil with
cos²(lat) metric factors.  All non-dimensional scale factors (DPI, THO,
FF, ZPL, etc.) are identical to the Fortran.

Public API
----------
lap_wu(f, lat, lon, LL)            → LL²·∇²f        (NW, nlat, nlon)
inv_lap_wu(rhs, lat, lon, LL)      → f s.t. lap_wu(f)=rhs
grad_wu(f, lat, lon, LL)           → (LL·∂f/∂x, LL·∂f/∂y)
jac_wu(a, b, lat, lon, LL)         → LL²·J(a,b)
"""

from __future__ import annotations

import numpy as np

from .operators import laplacian, laplacian_inv, gradient, jacobian
from pvtend.sh_ops import invert_helmholtz_sh
from .coords import R_EARTH

__all__ = [
    "lap_wu",
    "inv_lap_wu",
    "inv_helm_wu",
    "grad_wu",
    "jac_wu",
]


def lap_wu(
    f: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> np.ndarray:
    """Wu non-dim Laplacian: LL² · ∇²f.

    Parameters
    ----------
    f   : (..., nlat, nlon) field in Wu non-dim units.
    lat : (nlat,) latitudes [degrees], ascending.
    lon : (nlon,) longitudes [degrees].
    LL  : zonal grid spacing [m] (= ``scales.LL``).

    Returns
    -------
    ndarray of same shape, Wu-non-dim.  O(1) when f is O(1) and LL matches
    the actual grid spacing.
    """
    return LL * LL * laplacian(f, lat, lon)


def inv_lap_wu(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> np.ndarray:
    """Inverse of lap_wu: returns f such that lap_wu(f) = rhs.

    Equivalent to ``laplacian_inv(rhs / LL²)``.

    Parameters
    ----------
    rhs : (..., nlat, nlon) right-hand side in Wu non-dim units.
    lat, lon, LL : same as ``lap_wu``.

    Returns
    -------
    ndarray of same shape in Wu non-dim units.
    """
    return laplacian_inv(rhs / (LL * LL), lat, lon)


def inv_helm_wu(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
    c: float,
) -> np.ndarray:
    """Solve ``(lap_wu + c) H = rhs``, i.e. ``(LL²·∇²_sph + c) H = rhs``.

    This is the modified-Helmholtz solve required for Wu's H-equation, whose
    spatial operator is ``lap_wu + ASI·BB_k``.  The eigenvalue at total
    wavenumber n is ``-n(n+1)·LL²/R² + c``.  For the PPVI H-equation
    ``c = ASI_mean·BB[k] < 0``, which keeps all eigenvalues negative and
    well-posed (no near-zero eigenvalue issues).

    Derivation::

        (LL²·∇²_sph + c) H = rhs
        (∇²_sph + c/LL²)  H = rhs / LL²
        H = L⁻¹_{c/LL²}(rhs / LL²) × LL²   [via invert_helmholtz_sh]

    But ``invert_helmholtz_sh`` solves ``(∇²_sph + c_sh) H = rhs_sh``, so::

        c_sh  = c / LL²
        rhs_sh = rhs / LL²
        H     = LL² · (result)   ← cancelled by LL² outside

    Parameters
    ----------
    rhs : (..., nlat, nlon)  right-hand side in Wu non-dim units.
    lat, lon : coordinate arrays [degrees].
    LL  : Wu grid-spacing scale [m].
    c   : constant Helmholtz coefficient in Wu non-dim units
          (typically ``mean(ASI[k]) * BB[k]``, a negative number).

    Returns
    -------
    ndarray same shape as *rhs*, Wu non-dim.
    """
    # (∇²_sph + c/LL²) H = rhs/LL²  → same solution H as (LL²·∇²_sph + c)H = rhs
    rhs_sh = rhs / (LL * LL)
    c_sh   = c   / (LL * LL)
    out    = np.empty_like(rhs_sh, dtype=np.float64)
    for k in range(rhs_sh.shape[0]):
        out[k] = invert_helmholtz_sh(
            rhs_sh[k].astype(np.float64), lat, lon, c=c_sh, R_earth=R_EARTH
        )
    return out


def grad_wu(
    f: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wu non-dim gradient: (LL·∂f/∂x, LL·∂f/∂y).

    The returned components are dimensionless when *f* is dimensionless and
    LL equals the physical grid spacing; magnitude is O(1) per grid step.

    Parameters
    ----------
    f   : (..., nlat, nlon)
    lat, lon, LL : as above.

    Returns
    -------
    (fx_wu, fy_wu) : tuple of ndarrays of the same shape as f.
    """
    fx, fy = gradient(f, lat, lon)
    return LL * fx, LL * fy


def jac_wu(
    a: np.ndarray,
    b: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> np.ndarray:
    """Wu non-dim Jacobian: LL² · J(a, b).

    J(a,b) = ∂a/∂x · ∂b/∂y − ∂a/∂y · ∂b/∂x  (physical units in m⁻²)
    Wu's non-dim grid gives J_wu = LL² · J_phys.

    Parameters
    ----------
    a, b : (..., nlat, nlon) Wu non-dim fields.
    lat, lon, LL : as above.

    Returns
    -------
    ndarray of same shape, Wu non-dim.
    """
    return LL * LL * jacobian(a, b, lat, lon)
