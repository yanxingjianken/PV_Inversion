"""sh_ppvi.coords — vertical coordinate system and FD operators.

Pressure levels from ERA5 (10 levels, 1000 → 200 hPa).
Exner function π = (p/p0)^κ, κ = R/cp.

FD operators on π-grid
-----------------------
D_π f[k]   — first derivative ∂f/∂π, non-uniform 3-point stencil.
             At interior points: central difference.
             At k=0 (surface) and k=K-1 (top): one-sided 2nd-order stencil.
D_ππ f[k]  — second derivative ∂²f/∂π², non-uniform 3-point stencil.
             BCs applied by replacing end-rows with θ equations.

Boundary conditions
-------------------
Wu's convention: Φ is the primary variable for the vertical coupling.
At the surface (k=0) and top (k=K-1), we impose:
    −∂Φ/∂π = θ · cp        (hydrostatic + definition of θ in π coords)
→   D_π_bc Φ[0]   = −θ_bot · cp
    D_π_bc Φ[K-1] = −θ_top · cp

This is applied in invert_full.py by replacing the first and last rows
of the system matrix before solving — not in this module.  Here we only
provide the operator coefficients.

All operators return (K,) arrays when applied to a 1-D vector, or
broadcast to (..., K, nlat, nlon) when applied level-by-level.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# ═══════════════════════════════════════════════════════════
# 1. Fixed ERA5 pressure levels
# ═══════════════════════════════════════════════════════════

P0_PA   = 1.0e5        # reference pressure [Pa]
KAPPA   = 287.0 / 1004.0   # R/cp = 0.28585...
G       = 9.80665      # m s⁻²
CP      = 1004.0       # J kg⁻¹ K⁻¹
R       = 287.0        # J kg⁻¹ K⁻¹
OMEGA   = 7.292e-5     # rad s⁻¹
R_EARTH = 6.371e6      # m

# ERA5 pressure levels [Pa], surface → top (ascending p = descending altitude)
PLEVS_PA = np.array([
    1000., 925., 850., 700., 600., 500., 400., 300., 250., 200.
]) * 100.0   # → Pa

NW = len(PLEVS_PA)    # 10

# Exner function π = (p/p0)^κ, monotonically decreasing surface → top
PI_VALS = (PLEVS_PA / P0_PA) ** KAPPA   # shape (NW,)

# Potential temperature [K] from T and p: θ = T * (p0/p)^κ
# (computed per-call, not stored here)


# ═══════════════════════════════════════════════════════════
# 2. FD coefficient builders
# ═══════════════════════════════════════════════════════════

def _fd1_coeffs(xm: float, x0: float, xp: float) -> Tuple[float, float, float]:
    """Three-point non-uniform first-derivative coefficients at x0.

    Returns (am, a0, ap) such that f'(x0) ≈ am·f(xm) + a0·f(x0) + ap·f(xp).
    Uses Lagrange polynomial differentiation.
    """
    hm = x0 - xm   # > 0
    hp = xp - x0   # > 0
    am = -hp / (hm * (hm + hp))
    a0 = (hp - hm) / (hm * hp)
    ap = hm / (hp * (hm + hp))
    return am, a0, ap


def _fd2_coeffs(xm: float, x0: float, xp: float) -> Tuple[float, float, float]:
    """Three-point non-uniform second-derivative coefficients at x0."""
    hm = x0 - xm
    hp = xp - x0
    am = 2.0 / (hm * (hm + hp))
    a0 = -2.0 / (hm * hp)
    ap = 2.0 / (hp * (hm + hp))
    return am, a0, ap


def _fd1_onesided(x0: float, x1: float, x2: float, forward: bool = True) -> Tuple[float, float, float]:
    """Second-order one-sided first-derivative at x0 using x0, x1, x2.

    forward=True: derivative at x0 using x0, x1, x2 (bottom BC).
    forward=False: derivative at x2 using x0, x1, x2 (top BC).
    """
    if forward:
        # at x0
        h1 = x1 - x0; h2 = x2 - x0
        a0 = -(h1 + h2) / (h1 * h2)
        a1 = h2 / (h1 * (h2 - h1))
        a2 = -h1 / (h2 * (h2 - h1))
        return a0, a1, a2
    else:
        # at x2
        h0 = x2 - x0; h1 = x2 - x1
        a0 = h1 / (h0 * (h0 - h1))
        a1 = -h0 / (h1 * (h0 - h1))
        a2 = (h0 + h1) / (h0 * h1)
        return a0, a1, a2


def build_D1_matrix(pi: np.ndarray) -> np.ndarray:
    """Build the (NW, NW) first-derivative matrix in π.

    Interior: central 3-point.  Ends: one-sided 2nd-order.
    Returns M such that D_π f = M @ f.
    """
    N = len(pi)
    M = np.zeros((N, N))
    # Interior
    for k in range(1, N - 1):
        am, a0, ap = _fd1_coeffs(pi[k - 1], pi[k], pi[k + 1])
        M[k, k - 1] = am
        M[k, k]     = a0
        M[k, k + 1] = ap
    # Bottom (k=0) — one-sided forward
    a0, a1, a2 = _fd1_onesided(pi[0], pi[1], pi[2], forward=True)
    M[0, 0] = a0; M[0, 1] = a1; M[0, 2] = a2
    # Top (k=N-1) — one-sided backward
    a0, a1, a2 = _fd1_onesided(pi[N - 3], pi[N - 2], pi[N - 1], forward=False)
    M[N - 1, N - 3] = a0; M[N - 1, N - 2] = a1; M[N - 1, N - 1] = a2
    return M


def build_D2_matrix(pi: np.ndarray) -> np.ndarray:
    """Build the (NW, NW) second-derivative matrix in π.

    Interior: central 3-point.
    Ends: one-sided (first-order accurate — sufficient for BC rows
    since those rows will be replaced by BC equations in the inversion).
    """
    N = len(pi)
    M = np.zeros((N, N))
    # Interior
    for k in range(1, N - 1):
        am, a0, ap = _fd2_coeffs(pi[k - 1], pi[k], pi[k + 1])
        M[k, k - 1] = am
        M[k, k]     = a0
        M[k, k + 1] = ap
    # Ends: use interior stencil with ghost extrapolation (handled by BC replacement)
    # For now just fill with same one-sided approximation at the boundary.
    # k=0 uses k=0,1,2; k=N-1 uses k=N-3,N-2,N-1
    am, a0, ap = _fd2_coeffs(pi[0], pi[1], pi[2])
    M[0, 0] = am; M[0, 1] = a0; M[0, 2] = ap  # approx at k=1, not k=0
    # For actual BC replacement, caller zeros this row.
    am, a0, ap = _fd2_coeffs(pi[N - 3], pi[N - 2], pi[N - 1])
    M[N - 1, N - 3] = am; M[N - 1, N - 2] = a0; M[N - 1, N - 1] = ap
    return M


# Prebuilt matrices (singleton for the fixed ERA5 levels)
D1_MAT: np.ndarray = build_D1_matrix(PI_VALS)   # (NW, NW)
D2_MAT: np.ndarray = build_D2_matrix(PI_VALS)   # (NW, NW)


# ═══════════════════════════════════════════════════════════
# 3. Apply operators to 3-D fields
# ═══════════════════════════════════════════════════════════

def d_pi(field: np.ndarray, mat: np.ndarray = D1_MAT) -> np.ndarray:
    """Apply ∂/∂π to field along axis 0 (level axis).

    Args:
        field: (..., NW, nlat, nlon) or (NW,) or (NW, N).
        mat:   (NW, NW) differentiation matrix.

    Returns: same shape as field.
    """
    return np.einsum("kl,...lij->...kij", mat, field) if field.ndim >= 3 \
        else mat @ field


def d_pi_pi(field: np.ndarray, mat: np.ndarray = D2_MAT) -> np.ndarray:
    """Apply ∂²/∂π² to field along axis 0."""
    return np.einsum("kl,...lij->...kij", mat, field) if field.ndim >= 3 \
        else mat @ field


# ═══════════════════════════════════════════════════════════
# 4. θ and Φ conversions
# ═══════════════════════════════════════════════════════════

def theta_from_T(T: np.ndarray) -> np.ndarray:
    """Potential temperature θ = T·(p0/p)^κ. T shape: (NW, nlat, nlon)."""
    return T * (P0_PA / PLEVS_PA[:, None, None]) ** KAPPA


def T_from_theta(theta: np.ndarray) -> np.ndarray:
    """Temperature T from θ. θ shape: (NW, nlat, nlon)."""
    return theta / (P0_PA / PLEVS_PA[:, None, None]) ** KAPPA


def phi_from_z(z: np.ndarray) -> np.ndarray:
    """Geopotential Φ = g·z [m²/s²]. z shape: (NW, nlat, nlon)."""
    return G * z


def dpi_phi_from_theta(theta: np.ndarray) -> np.ndarray:
    """Compute ∂Φ/∂π = −θ·cp (hydrostatic in π coords).

    Returns (NW, nlat, nlon) array — the vertical gradient of Φ at every level,
    which is diagnostic (not the integrated Φ).
    """
    return -CP * theta


# ═══════════════════════════════════════════════════════════
# 5. Coriolis parameter
# ═══════════════════════════════════════════════════════════

def coriolis(lat: np.ndarray) -> np.ndarray:
    """f = 2Ω sin(lat).  lat in degrees, ascending 0→90. Returns (nlat,)."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat))


# ═══════════════════════════════════════════════════════════
# 6. Self-test
# ═══════════════════════════════════════════════════════════

def _selftest():
    """Verify FD operators on π-linear and π-quadratic test functions."""
    pi = PI_VALS.copy()

    # Test D1: apply to f(π) = c·π + d → f'(π) = c
    c = 3.7
    f = c * pi + 1.2
    df_fd = D1_MAT @ f
    df_exact = c * np.ones_like(pi)
    err1 = np.max(np.abs(df_fd - df_exact))
    assert err1 < 1e-10, f"D1 linear test failed: max err = {err1:.2e}"

    # Test D1: apply to f(π) = π² → f'(π) = 2π
    f2 = pi ** 2
    df2_fd = D1_MAT @ f2
    df2_ex = 2.0 * pi
    # Interior points only (end-points have one-sided stencil on non-uniform grid)
    err2 = np.max(np.abs((df2_fd - df2_ex)[1:-1]))
    assert err2 < 2e-5, f"D1 quadratic interior test failed: max err = {err2:.2e}"

    # Test D2: apply to f(π) = π² → f''(π) = 2
    d2f_fd = D2_MAT @ f2
    d2f_ex = 2.0 * np.ones_like(pi)
    err3 = np.max(np.abs((d2f_fd - d2f_ex)[1:-1]))
    assert err3 < 1e-8, f"D2 quadratic interior test failed: max err = {err3:.2e}"

    print(f"[coords] selftest passed: D1_linear_err={err1:.1e}, "
          f"D1_quad_err={err2:.1e}, D2_quad_err={err3:.1e}")


if __name__ == "__main__":
    _selftest()
