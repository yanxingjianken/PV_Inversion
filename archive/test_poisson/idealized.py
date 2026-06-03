"""Analytical test functions for Poisson/Helmholtz solver benchmarks.

Provides known solutions ψ_exact(x, y) whose Laplacian ∇²ψ_exact can be
computed analytically, enabling direct error measurement against any solver.

The three test cases span:
  A — Smooth global field (spherical harmonic) → tests pole closure
  B — Localised NH feature (Gaussian bump)   → tests regional performance
  C — Sharp gradient (tanh front)             → tests spectral ringing
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import R_EARTH

# ═══════════════════════════════════════════════════════════════════════════
#  Spherical-harmonic test (global grid, smooth)
# ═══════════════════════════════════════════════════════════════════════════


def sph_harm_test(
    l: int = 5,
    m: int = 3,
    nlat: int = 181,
    nlon: int = 360,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a spherical harmonic test field and its analytical Laplacian.

    On the sphere, ∇² Y_l^m = -l(l+1)/R² · Y_l^m, giving an exact
    reference for any Poisson solver.

    Args:
        l: Total wavenumber (≥ 0).
        m: Zonal wavenumber (0 ≤ m ≤ l).
        nlat: Number of latitude points (global, poles included).
        nlon: Number of longitude points.
        amplitude: Peak amplitude of the field.

    Returns:
        ``(lat, lon, psi_exact, zeta_exact)`` where:
        - *lat*: latitudes [°] ascending -90 → 90, shape ``(nlat,)``
        - *lon*: longitudes [°] 0 → 360, shape ``(nlon,)``
        - *psi_exact*: Y_l^m field, shape ``(nlat, nlon)``
        - *zeta_exact*: ∇²ψ = -l(l+1)/R² · ψ, same shape
    """
    if m > l:
        raise ValueError(f"Require m ≤ l; got l={l}, m={m}")

    lat_deg = np.linspace(-90, 90, nlat)
    lon_deg = np.linspace(0, 360, nlon, endpoint=False)

    colat = np.deg2rad(90.0 - lat_deg)  # (nlat,)  co-latitude
    lon_rad = np.deg2rad(lon_deg)        # (nlon,)

    # Associated Legendre P_l^m(cos θ) via scipy if available, else pure NumPy.
    try:
        from scipy.special import lpmv
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    if _has_scipy:
        # lpmv(m, l, x) — note argument order
        plm = lpmv(m, l, np.cos(colat))  # (nlat,)
    else:
        # Fallback: use the standard recurrence (stable for moderate l).
        plm = _legendre_plm(l, m, np.cos(colat))

    # Normalise so max|ψ| = amplitude
    plm *= amplitude / max(abs(plm).max(), 1e-30)

    psi = plm[:, None] * np.cos(m * lon_rad)[None, :]   # (nlat, nlon)

    # Analytical Laplacian
    eig = -l * (l + 1) / (R_EARTH * R_EARTH)
    zeta = eig * psi

    return lat_deg, lon_deg, psi.astype(np.float64), zeta.astype(np.float64)


def _legendre_plm(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Pure-NumPy associated Legendre P_l^m(x) for |x| ≤ 1."""
    # Recurrence: (l-m+1)P_{l+1}^m = (2l+1)x P_l^m - (l+m)P_{l-1}^m
    # Start from P_m^m = (-1)^m (2m-1)!! (1-x²)^{m/2}
    x = np.asarray(x, dtype=np.float64)
    # P_m^m
    p_mm = 1.0
    if m > 0:
        fact = 1.0
        for _ in range(m):
            p_mm *= fact * np.sqrt(1.0 - x * x)
            fact += 2.0
        if m % 2 == 1:
            p_mm = -p_mm
    if l == m:
        return p_mm
    # P_{m+1}^m = (2m+1) x P_m^m
    p_prev = p_mm
    p_curr = (2 * m + 1) * x * p_mm
    if l == m + 1:
        return p_curr
    for k in range(m + 2, l + 1):
        p_next = ((2 * k - 1) * x * p_curr - (k + m - 1) * p_prev) / (k - m)
        p_prev, p_curr = p_curr, p_next
    return p_curr


# ═══════════════════════════════════════════════════════════════════════════
#  Gaussian-bump test (regional / NH, localised)
# ═══════════════════════════════════════════════════════════════════════════


def gaussian_bump_test(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float = 45.0,
    lon0: float = -100.0,
    sigma_deg: float = 10.0,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a Gaussian bump and its analytical Cartesian Laplacian.

    The bump is placed at *(lat0, lon0)* with Gaussian half-width
    *sigma_deg* (converted to metres).  The Laplacian is computed
    analytically on the tangent plane — accurate when σ ≪ R_earth.

    Args:
        lat: Latitudes [°], ascending.
        lon: Longitudes [°].
        lat0, lon0: Bump centre [°].
        sigma_deg: Gaussian half-width [°].
        amplitude: Peak amplitude.

    Returns:
        ``(psi, zeta)`` each shape ``(len(lat), len(lon))``.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Convert to metres on the tangent plane at lat0
    sigma_m = np.deg2rad(sigma_deg) * R_EARTH
    dy_m = np.deg2rad(lat - lat0) * R_EARTH               # (nlat,)
    # Account for convergence of meridians at lat0
    dx_m = np.deg2rad(lon - lon0) * R_EARTH * np.cos(np.deg2rad(lat0))  # (nlon,)

    dy_2d = dy_m[:, None]    # (nlat, 1)
    dx_2d = dx_m[None, :]    # (1, nlon)

    r2 = (dx_2d**2 + dy_2d**2) / (2.0 * sigma_m**2)
    psi = amplitude * np.exp(-r2)

    # ∇²ψ = (r²/σ⁴ − 2/σ²) · ψ  (Cartesian Laplacian of a Gaussian)
    zeta = amplitude * np.exp(-r2) * (r2 / (sigma_m**2) - 1.0 / sigma_m**2) * 2.0

    return psi.astype(np.float64), zeta.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
#  Tanh-front test (sharp gradient — stresses spectral methods)
# ═══════════════════════════════════════════════════════════════════════════


def tanh_front_test(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_front: float = 50.0,
    width_deg: float = 2.0,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a tanh front (zonal band) and its analytical Laplacian.

    The solution is a function of latitude only, so ∂²/∂x² = 0.

    Args:
        lat: Latitudes [°].
        lon: Longitudes [°].
        lat_front: Centre latitude of the front [°].
        width_deg: Front half-width [°].
        amplitude: Half the total jump across the front.

    Returns:
        ``(psi, zeta)`` each shape ``(len(lat), len(lon))``.
    """
    lat = np.asarray(lat, dtype=np.float64)

    width_m = np.deg2rad(width_deg) * R_EARTH
    dy_m = np.deg2rad(lat - lat_front) * R_EARTH   # (nlat,)

    arg = dy_m / width_m
    psi = amplitude * np.tanh(arg)                  # (nlat,)
    psi = np.tile(psi[:, None], (1, len(lon)))      # (nlat, nlon)

    # ∂²ψ/∂y² = -2·amp/w² · tanh(arg) · sech²(arg)
    d2psi_dy2 = (
        -2.0 * amplitude / width_m**2
        * np.tanh(arg)
        * (1.0 / np.cosh(arg)) ** 2
    )
    zeta = np.tile(d2psi_dy2[:, None], (1, len(lon)))

    return psi.astype(np.float64), zeta.astype(np.float64)
