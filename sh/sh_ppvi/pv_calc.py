"""sh_ppvi.pv_calc — PV computation and stream-function inversion.

Mirrors pvpialln_94UV.f (Wu/Davis 1994), replacing finite-difference
horizontal operators with spherical-harmonic equivalents.

Mathematical content
--------------------
Following pvpialln_94UV.f line-by-line:

L172–182  θ = T · CP / π(k)  (potential temperature via Exner function)
L190–194  θ_B = 0.5·[θ(0)+θ(1)],  θ_T = 0.5·[θ(NW-2)+θ(NW-1)]
L200–204  ∂θ/∂λ, ∂θ/∂φ via SH gradient;  ∂θ/∂π via D1_MAT
L210–217  ζ = vortdiv(u,v) [spherical]
L222–321  ψ via helmholtz(u,v) [spherical; replaces Davis boundary integration
          + 2-D Poisson SOR in Fortran]
L326–334  ∂u/∂π, ∂v/∂π via D1_MAT
L337–354  Ertel PV:
            Q = -COEF·π^(−2.5)·[(f+ζ)·∂θ/∂π] − COEF·π^(−2.5)·[∂u/∂π·∂θ/∂y
                                                                  −∂v/∂π·∂θ/∂x]
          with COEF = 1e2 · 1e6 · 9.81 · κ · CP^3.5 / P0

COEF computation (pvpialln L463):
    COEF = 1.E2 * 1.E6 * 9.81 * KAP * (CP ** 3.5) / P0
    where KAP=2/7, CP=1004.5, P0=1e5.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from .coords import (
    PLEVS_PA, PI_VALS, NW, P0_PA, KAPPA, CP, G,
    D1_MAT, d_pi,
)
from .operators import gradient, vortdiv, helmholtz  # noqa: F401

__all__ = ["pvpialln_sh", "ertel_pv_sh"]

# ──────────────────────────────────────────────────────────────────────────────
# COEF — adapted for our PI_VALS = (p/p0)^κ  (dimensionless, no CP factor)
#
# Fortran: PI(K) = CP*(p/p0)^κ,  COEF = 1e2·1e6·g·κ·CP^3.5/P0,
#          Q = -COEF · PI^(-2.5) · [...]
#
# Python:  PI_VALS = (p/p0)^κ = PI_fortran/CP
#          STB_python = d(θ)/d(PI_VALS) = CP · STB_fortran
#          → COEF_python = COEF_fortran / CP^3.5 = 1e2·1e6·g·κ/P0
#
# Verified: COEF_python · PI_VALS^(-2.5) · (f+ζ) · STB_python
#           = COEF_fortran · PI_fortran^(-2.5) · (f+ζ) · STB_fortran  ✓
# ──────────────────────────────────────────────────────────────────────────────
_KAP_WU  = 2.0 / 7.0
_CP_WU   = 1004.5
_G_WU    = 9.81
_P0_WU   = 1.0e5
COEF_WU  = 1.0e2 * 1.0e6 * _G_WU * _KAP_WU / _P0_WU   # ≈ 2803.0


def pvpialln_sh(
    H: np.ndarray,
    T: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute PV and stream function mirroring pvpialln_94UV.f.

    Parameters
    ----------
    H   : (NW, nlat, nlon)  geopotential [m² s⁻²]
    T   : (NW, nlat, nlon)  temperature [K]
    u,v : (NW, nlat, nlon)  wind [m s⁻¹]
    lat : (nlat,)  ascending latitudes [°N]
    lon : (nlon,)  longitudes [°E]

    Returns
    -------
    dict with keys:
        theta   : (NW, nlat, nlon)  potential temperature [K]    (L172)
        theta_B : (nlat, nlon)      lower boundary θ [K]          (L190)
        theta_T : (nlat, nlon)      upper boundary θ [K]          (L191)
        dth_dx  : (NW, nlat, nlon)  ∂θ/∂x  [K m⁻¹]              (L200)
        dth_dy  : (NW, nlat, nlon)  ∂θ/∂y  [K m⁻¹]              (L200)
        dth_dpi : (NW, nlat, nlon)  ∂θ/∂π  [K / (J kg⁻¹ K⁻¹)]  (L204)
        zeta    : (NW, nlat, nlon)  relative vorticity [s⁻¹]      (L210)
        psi     : (NW, nlat, nlon)  stream function [m² s⁻¹]      (L222)
        du_dpi  : (NW, nlat, nlon)  ∂u/∂π                        (L326)
        dv_dpi  : (NW, nlat, nlon)  ∂v/∂π                        (L326)
        pv      : (NW, nlat, nlon)  Ertel PV [10⁻⁶ K m² s⁻¹ kg⁻¹ × COEF] (L337)
        f       : (nlat,)           Coriolis parameter [s⁻¹]
    """
    nlat, nlon = lat.size, lon.size

    # ── L172: θ = T / (p/p0)^κ  ─────────────────────────────────────────────
    # Fortran: TH = T * CP / PI  where PI = CP*(p/p0)^κ  → θ = T/(p/p0)^κ
    # Python:  PI_VALS = (p/p0)^κ  (no CP factor)        → θ = T / PI_VALS
    pi_col = PI_VALS[:, np.newaxis, np.newaxis]   # (NW,1,1)
    theta  = T / pi_col                            # (NW, nlat, nlon)

    # ── L190–194: boundary θ ─────────────────────────────────────────────────
    theta_B = 0.5 * (theta[0] + theta[1])         # (nlat, nlon)
    theta_T = 0.5 * (theta[NW - 2] + theta[NW - 1])

    # ── L200–204: ∂θ/∂λ, ∂θ/∂φ (SH), ∂θ/∂π (vertical FD) ──────────────────
    dth_dx, dth_dy = gradient(theta, lat, lon)    # (NW, nlat, nlon)
    dth_dpi = d_pi(theta)                          # (NW, nlat, nlon)

    # ── L210–217: relative vorticity ζ (SH, spherical) ───────────────────────
    zeta, _ = vortdiv(u, v, lat, lon)              # (NW, nlat, nlon)

    # ── L222–321: ψ from u,v via Helmholtz decomposition ────────────────────
    # Replaces Davis boundary integration (L322–349) + 2-D Poisson SOR (L383–420)
    hd     = helmholtz(u, v, lat, lon)
    psi    = hd["psi"]                             # (NW, nlat, nlon)

    # ── L326–334: ∂u/∂π, ∂v/∂π ───────────────────────────────────────────────
    du_dpi = d_pi(u)                               # (NW, nlat, nlon)
    dv_dpi = d_pi(v)                               # (NW, nlat, nlon)

    # ── L337–354: Ertel PV ────────────────────────────────────────────────────
    # Coriolis parameter  f = 2·Ω·sin(φ)
    # pvpialln L158: FC(I) = 1.458e-4 * sin(lat_I * π/180)
    # 1.458e-4 ≈ 2Ω  (Wu uses 1.458e-4; standard 2Ω ≈ 1.4584e-4)
    f_1d  = 1.458e-4 * np.sin(np.deg2rad(lat))    # (nlat,)
    f_3d  = f_1d[np.newaxis, :, np.newaxis]        # (1, nlat, 1)

    # π^(-2.5) per level
    pi_m25 = PI_VALS[:, np.newaxis, np.newaxis] ** (-2.5)  # (NW,1,1)

    # Vorticity term: -(f+ζ)·∂θ/∂π
    pv_vort = -COEF_WU * pi_m25 * (f_3d + zeta) * dth_dpi

    # Wind-shear term: +COEF·π^(-2.5)·[∂u/∂π·∂θ/∂y − ∂v/∂π·∂θ/∂x]
    # pvpialln L468–470:
    #   ZSHR = COEF·π^(-2.5)·[DU·DTHY − DV·DTHX]
    #   Q = -COEF·π^(-2.5)·[(FC+VOR)·STB] − ZSHR
    # Note sign: subtraction of the shear term from the vorticity term.
    pv_shear = COEF_WU * pi_m25 * (du_dpi * dth_dy - dv_dpi * dth_dx)
    pv = pv_vort - pv_shear

    return {
        "theta":   theta,
        "theta_B": theta_B,
        "theta_T": theta_T,
        "dth_dx":  dth_dx,
        "dth_dy":  dth_dy,
        "dth_dpi": dth_dpi,
        "zeta":    zeta,
        "psi":     psi,
        "du_dpi":  du_dpi,
        "dv_dpi":  dv_dpi,
        "pv":      pv,
        "f":       f_1d,
    }


def ertel_pv_sh(
    psi: np.ndarray,
    theta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Compute Ertel PV from ψ (and u,v for shear) and θ.

    Convenience wrapper that avoids recomputing θ from T.  Used in
    the outer loop to re-evaluate PV from updated balanced fields.
    """
    dth_dx, dth_dy = gradient(theta, lat, lon)
    dth_dpi = d_pi(theta)
    zeta, _ = vortdiv(u, v, lat, lon)
    du_dpi  = d_pi(u)
    dv_dpi  = d_pi(v)

    f_3d    = 1.458e-4 * np.sin(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    pi_m25  = PI_VALS[:, np.newaxis, np.newaxis] ** (-2.5)

    pv_vort  = -COEF_WU * pi_m25 * (f_3d + zeta) * dth_dpi
    pv_shear = COEF_WU * pi_m25 * (du_dpi * dth_dy - dv_dpi * dth_dx)
    return pv_vort - pv_shear
