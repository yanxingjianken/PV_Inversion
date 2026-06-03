"""sh_ppvi.operators — horizontal SH operators applied level-by-level.

All functions accept NH-only grids (lat ascending 0→90, lon 0→358.5).
pvtend.sh_ops handles parity-mirroring to the global sphere internally.

Level axis convention: axis 0 of 3-D arrays = pressure level (k),
axis 1 = latitude (ascending), axis 2 = longitude.

Public API
----------
laplacian(f, lat, lon)            → ∇²f  (NW, nlat, nlon)
laplacian_inv(rhs, lat, lon)      → χ such that ∇²χ = rhs
gradient(f, lat, lon)             → (df_dx, df_dy) each (NW, nlat, nlon)
vortdiv(u, v, lat, lon)           → (ζ, δ)
helmholtz(u, v, lat, lon)         → dict with psi, chi, u_rot, v_rot, ...
jacobian(a, b, lat, lon)          → J(a,b) = ∂a/∂x·∂b/∂y − ∂a/∂y·∂b/∂x
div_f_grad(f, psi, lat, lon)      → ∇·(f·∇psi)  where f = Coriolis array
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from pvtend.sh_ops import (
    gradient_sh,
    laplacian_sh,
    invert_laplacian_sh,
    vortdiv_sh,
    helmholtz_sh,
)
from .coords import R_EARTH, coriolis

__all__ = [
    "laplacian", "laplacian_inv", "gradient", "vortdiv",
    "helmholtz", "jacobian", "div_f_grad",
]


# ──────────────────────────────────────────────────────────
# Internal: apply 2-D SH operator to every level of a 3-D field
# ──────────────────────────────────────────────────────────

def _apply_2d(fn, field3d: np.ndarray, lat, lon, **kw):
    """Apply fn(field2d, lat, lon, **kw) level-by-level to (NW, nlat, nlon)."""
    out = np.empty_like(field3d, dtype=np.float64)
    for k in range(field3d.shape[0]):
        out[k] = fn(field3d[k].astype(np.float64), lat, lon, R_earth=R_EARTH, **kw)
    return out


def _apply_2d_pair(fn, a3d, b3d, lat, lon, **kw):
    """Apply fn(a2d, b2d, lat, lon) level-by-level; fn returns (out_a, out_b)."""
    nk = a3d.shape[0]
    out_a = np.empty_like(a3d, dtype=np.float64)
    out_b = np.empty_like(b3d, dtype=np.float64)
    for k in range(nk):
        ra, rb = fn(a3d[k].astype(np.float64), b3d[k].astype(np.float64),
                    lat, lon, R_earth=R_EARTH, **kw)
        out_a[k] = ra
        out_b[k] = rb
    return out_a, out_b


# ──────────────────────────────────────────────────────────
# Public operators
# ──────────────────────────────────────────────────────────

def laplacian(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Spherical Laplacian ∇²f at every level.

    Args:
        f: (NW, nlat, nlon), NH ascending.
    Returns: ∇²f, same shape.
    """
    return _apply_2d(laplacian_sh, f, lat, lon)


def laplacian_inv(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    parity: str = "scalar",
) -> np.ndarray:
    """Invert ∇²χ = rhs at every level (mean-zero gauge).

    Args:
        rhs: (NW, nlat, nlon).
        parity: "scalar" (even) for ψ/Φ; "v" (odd) for divergence-like RHS.
    Returns: χ, same shape.
    """
    out = np.empty_like(rhs, dtype=np.float64)
    for k in range(rhs.shape[0]):
        out[k] = invert_laplacian_sh(
            rhs[k].astype(np.float64), lat, lon,
            R_earth=R_EARTH, parity=parity,
        )
    return out


def gradient(
    f: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(∂f/∂x, ∂f/∂y) in m⁻¹ × [f] at every level.

    Returns: (df_dx, df_dy) each (NW, nlat, nlon).
    """
    df_dx = np.empty_like(f, dtype=np.float64)
    df_dy = np.empty_like(f, dtype=np.float64)
    for k in range(f.shape[0]):
        fx, fy = gradient_sh(f[k].astype(np.float64), lat, lon, R_earth=R_EARTH)
        df_dx[k] = fx
        df_dy[k] = fy
    return df_dx, df_dy


def vortdiv(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Relative vorticity ζ and divergence δ at every level.

    Returns: (zeta, div) each (NW, nlat, nlon).
    """
    return _apply_2d_pair(vortdiv_sh, u, v, lat, lon)


def helmholtz(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> Dict[str, np.ndarray]:
    """Helmholtz decomposition at every level.

    Returns dict with keys: psi, chi, u_rot, v_rot, u_div, v_div,
    u_har, v_har, vorticity, divergence. Each array (NW, nlat, nlon).
    """
    nk = u.shape[0]
    keys = ("psi", "chi", "u_rot", "v_rot", "u_div", "v_div",
            "u_har", "v_har", "vorticity", "divergence")
    result: Dict[str, np.ndarray] = {k: np.empty_like(u, dtype=np.float64) for k in keys}
    for k in range(nk):
        d = helmholtz_sh(u[k].astype(np.float64), v[k].astype(np.float64),
                         lat, lon, R_earth=R_EARTH)
        for key in keys:
            result[key][k] = d[key]
    return result


def jacobian(
    a: np.ndarray, b: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """J(a,b) = ∂a/∂x·∂b/∂y − ∂a/∂y·∂b/∂x at every level.

    Uses SH gradients; result in m⁻² × [a] × [b].
    Returns: (NW, nlat, nlon).
    """
    da_dx, da_dy = gradient(a, lat, lon)
    db_dx, db_dy = gradient(b, lat, lon)
    return da_dx * db_dy - da_dy * db_dx


def div_f_grad(
    f_coriolis: np.ndarray,
    psi: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Compute ∇·(f·∇ψ) = f·∇²ψ + ∇f·∇ψ, where f is the Coriolis field.

    This is the leading term in the Charney balance equation.
    f_coriolis: (nlat,) or (NW, nlat, nlon) broadcast-compatible with psi.
    psi: (NW, nlat, nlon).

    Returns: (NW, nlat, nlon).
    """
    # f varies only in latitude
    if f_coriolis.ndim == 1:
        f2d = f_coriolis[np.newaxis, :, np.newaxis]   # (1, nlat, 1)
    else:
        f2d = f_coriolis

    lap_psi = laplacian(psi, lat, lon)             # ∇²ψ
    _, df_dy = gradient(                            # ∂f/∂y (∂f/∂x = 0 for zonal f)
        np.broadcast_to(f2d, psi.shape).copy(), lat, lon
    )
    _, dpsi_dy = gradient(psi, lat, lon)

    return f2d * lap_psi + df_dy * dpsi_dy
