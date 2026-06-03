"""sh_ppvi.nondim — Wu/Davis non-dimensionalisation scales and helpers.

Mirrors qinvert21_94.f L213–276 exactly.

  DPI    = 500/NL                       (dimensionless π scale)
  LL     = R_EARTH · Δλ · π/180         (zonal grid spacing, m)
  FF     = 1e-4 s⁻¹                     (reference Coriolis)
  THO    = FF²·LL²/DPI                  (potential-temperature scale, K)
  FRC    = DPI·THO/(FF²·LL²) = 1        (Froude-like ratio, identity)
  QCONST = 1e6·κ·g·CP·FF·THO/(P0·DPI)   (PV scale in PVU)
  UC     = DPI·THO/(FF·LL)              (velocity scale, m/s)

Non-dimensionalisation (Fortran L280–367 / L401–412):

  PI_nd    = CP · (p/p0)^κ / DPI        (≡ CP·PI_VALS/DPI in our notation)
  H_nd     = H · g / (THO·DPI)          (geopotential m²/s² → non-dim)
  ψ_nd     = ψ · g / (THO·DPI)          (same scale as H)
  θ_nd     = θ / THO                    (K)
  q_nd     = PIF(k) · q_PVU / (1e2 · QCONST)    [MF=1 in our SH grid]
            where PIF(k) = (CP·(p/p0)^κ / CP)^2.5 = (p/p0)^(2.5·κ) = PI_VALS^2.5
  thr_nd   = g · thr_phys / (DPI·THO)   (convergence threshold)

Re-dimensionalisation after BALNC (Fortran L501–512):

  H_phys = H_nd · DPI·THO / g
  ψ_phys = ψ_nd · DPI·THO / g

Notes
-----
* In Wu's code `Q` (PV) is read in *PVU* (10⁻⁶ K m² kg⁻¹ s⁻¹).  Our `pv_calc`
  produces PV in the units output by `COEF_WU · π^(-2.5) · ...`.  With our
  rescaled `COEF_WU = 1e2·1e6·g·κ/P0 ≈ 2803`, the Python PV is *already*
  Wu's `Q` in PVU (verified ≈ tens to hundreds in the troposphere).
* We therefore use `q_nd = PIF · q_python / (1e2 · QCONST)` directly,
  *without* the per-grid-point map factor (MF=1 on a regular lat-lon grid).
"""

from __future__ import annotations
import numpy as np
from .coords import PI_VALS, NW, G, CP, R_EARTH, P0_PA, KAPPA

__all__ = [
    "Scales", "make_scales",
    "nondim_H", "redim_H",
    "nondim_psi", "redim_psi",
    "nondim_theta",
    "nondim_q",
    "nondim_thr",
    "PI_ND", "PI_WU", "PIF",
]


class Scales:
    """Container for Wu non-dimensionalisation scales.

    Attributes
    ----------
    NL     : int       number of vertical levels
    DPI    : float     500/NL
    LL     : float     zonal grid spacing in metres
    FF     : float     reference Coriolis 1e-4
    THO    : float     potential temperature scale [K]
    FRC    : float     = 1 by construction
    QCONST : float     PV scale [PVU]
    UC     : float     velocity scale [m/s]
    PI_nd  : (NW,)     non-dim Wu PI = CP·(p/p0)^κ/DPI
    PIF    : (NW,)     (p/p0)^(2.5·κ) factor for q non-dim
    H_factor   : float = g/(THO·DPI)   (H_nd = H_phys · H_factor)
    inv_H_factor : float = DPI·THO/g
    q_factor_per_k : (NW,) PIF/(1e2·QCONST)
    """

    def __init__(self, dlon_deg: float, nl: int = NW):
        self.NL  = nl
        self.DPI = 500.0 / nl
        # AA = 2e7/π (Wu Fortran L216) — Earth circumference / 2π ≈ R_E
        AA = 2.0e7 / np.pi
        self.LL  = AA * dlon_deg * np.pi / 180.0
        self.FF  = 1.0e-4
        self.THO = self.FF**2 * self.LL**2 / self.DPI
        self.FRC = self.DPI * self.THO / (self.FF**2 * self.LL**2)  # ≡ 1
        self.QCONST = 1.0e6 * KAPPA * G * CP * self.FF * self.THO \
                      / (P0_PA * self.DPI)
        self.UC  = self.DPI * self.THO / (self.FF * self.LL)

        # PI_VALS = (p/p0)^κ (dimensionless). Wu: PI = CP·(p/p0)^κ / DPI.
        self.PI_nd = CP * PI_VALS / self.DPI
        # PIF(k) = (PI_phys/CP)^2.5 = PI_VALS^2.5
        self.PIF   = PI_VALS ** 2.5

        self.H_factor      = G / (self.THO * self.DPI)
        self.inv_H_factor  = self.DPI * self.THO / G
        # q_nd = PIF(k) · q_phys / (1e2 · QCONST)
        self.q_factor_per_k = self.PIF / (1.0e2 * self.QCONST)

    def __repr__(self) -> str:
        return (f"Scales(NL={self.NL}, DPI={self.DPI:.3f}, "
                f"LL={self.LL:.3e}, THO={self.THO:.4f}, "
                f"QCONST={self.QCONST:.4e}, UC={self.UC:.3f}, "
                f"H_factor={self.H_factor:.4e})")


def make_scales(dlon_deg: float, nl: int = NW) -> Scales:
    """Build a Scales object for the given zonal grid spacing."""
    return Scales(dlon_deg=dlon_deg, nl=nl)


# Module-level scales for the standard 1.5° grid (NX=240).  Use make_scales()
# for other grids.
_DEFAULT = Scales(dlon_deg=1.5)
PI_ND = _DEFAULT.PI_nd   # CP·PI_VALS/DPI — Wu non-dim π, O(1) per level
PIF   = _DEFAULT.PIF
# Convenient alias for the vertical-operator builders in sor.py / balnc.py.
# PI_WU[k] = CP*(p_k/p0)^κ / DPI  ≈  1 per grid step in pressure (Δπ_wu ≈ 1).
PI_WU = PI_ND


# ─── helpers ────────────────────────────────────────────────────────────────

def nondim_H(H_phys: np.ndarray, sc: Scales) -> np.ndarray:
    """H_phys [m²/s²] (geopotential Φ = g·z) → H_nd.

    Wu reads heights in *metres* and multiplies by ``G/(THO·DPI)``.
    Our ``H_phys`` is geopotential (m²/s²), i.e. ``Φ = g·z``, so
    ``z = Φ/g`` and the correct non-dim is
    ``H_nd = z · G/(THO·DPI) = Φ / (THO·DPI)``.
    Applying ``sc.H_factor = G/(THO·DPI)`` directly to geopotential would
    introduce a spurious extra factor of ``g`` — hence this function divides
    by ``G`` first (or equivalently uses ``1/(THO·DPI)``).
    """
    return H_phys / (sc.THO * sc.DPI)


def redim_H(H_nd: np.ndarray, sc: Scales) -> np.ndarray:
    """H_nd → H_phys [m²/s²] (geopotential Φ)."""
    return H_nd * (sc.THO * sc.DPI)


def nondim_psi(psi_phys: np.ndarray, sc: Scales) -> np.ndarray:
    """ψ_phys [m²/s] → ψ_nd.

    Wu's SI input is in units of 10^5 m²/s (L57 comment) and non-dimmed as
    SI_nd = SI_10e5 * GG/(THO*DPI) (L292).  So from SI_phys [m²/s]:
        ψ_nd = ψ_phys * 1e-5 * H_factor.
    This gives ψ_nd ~ O(10–50) for typical mid-latitude stream function.
    """
    return psi_phys * 1.0e-5 * sc.H_factor


def redim_psi(psi_nd: np.ndarray, sc: Scales) -> np.ndarray:
    return psi_nd / (1.0e-5 * sc.H_factor)


def nondim_theta(theta_phys: np.ndarray, sc: Scales) -> np.ndarray:
    """θ_phys [K] → θ_nd."""
    return theta_phys / sc.THO


def nondim_q(q_phys: np.ndarray, sc: Scales) -> np.ndarray:
    """PV in our Python (PVU-like) units → non-dim q.

    q_nd(k) = PIF(k) · q_phys(k) / (1e2 · QCONST)
    """
    return q_phys * sc.q_factor_per_k[:, np.newaxis, np.newaxis]


def nondim_thr(thr_phys_m: float, sc: Scales) -> float:
    """Convergence threshold [m] → non-dim (Fortran L275)."""
    return G * thr_phys_m / (sc.DPI * sc.THO)
